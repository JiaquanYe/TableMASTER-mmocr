import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmocr.models.builder import DECODERS
from .base_decoder import BaseDecoder
from ..encoders.positional_encoding import PositionalEncoding

from mmocr.models.builder import DECODERS

class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, *input):
        x = input[0]
        return self.lut(x) * math.sqrt(self.d_model)


def clones(module, N):
    """ Produce N identical layers """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class SubLayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SubLayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        #tmp = self.norm(x)
        #tmp = sublayer(tmp)
        return x + self.dropout(sublayer(self.norm(x)))


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

def self_attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scale Dot Product Attention'
    """

    d_k = value.size(-1)
    score = torch.matmul(query, key.transpose(-2, -1) / math.sqrt(d_k))
    if mask is not None:
        #score = score.masked_fill(mask == 0, -1e9) # b, h, L, L
        score = score.masked_fill(mask == 0, -6.55e4) # for fp16
    p_attn = F.softmax(score, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):

    def __init__(self, headers, d_model, dropout):
        super(MultiHeadAttention, self).__init__()

        assert d_model % headers == 0
        self.d_k = int(d_model / headers)
        self.headers = headers
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.headers, self.d_k).transpose(1, 2)
             for l,x in zip(self.linears, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = self_attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.headers * self.d_k)
        return self.linears[-1](x)


class DecoderLayer(nn.Module):
    """
    Decoder is made of self attention, srouce attention and feed forward.
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = MultiHeadAttention(**self_attn)
        self.src_attn = MultiHeadAttention(**src_attn)
        self.feed_forward = FeedForward(**feed_forward)
        self.sublayer = clones(SubLayerConnection(size, dropout), 3)

    def forward(self, x, feature, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, feature, feature, src_mask))
        return self.sublayer[2](x, self.feed_forward)


@DECODERS.register_module()
class MasterDecoder(BaseDecoder):

    def __init__(self,
                 N,
                 decoder,
                 d_model,
                 num_classes,
                 start_idx,
                 padding_idx,
                 max_seq_len,
                 ):
        super(MasterDecoder, self).__init__()
        self.layers = clones(DecoderLayer(**decoder), N)
        self.norm = nn.LayerNorm(decoder.size)
        self.fc = nn.Linear(d_model, num_classes)

        self.embedding = Embeddings(d_model=d_model, vocab=num_classes)
        self.positional_encoding = PositionalEncoding(d_model=d_model)

        self.SOS = start_idx
        self.PAD = padding_idx
        self.max_length = max_seq_len

    def make_mask(self, src, tgt):
        """
        Make mask for self attention.
        :param src: [b, c, h, l_src]
        :param tgt: [b, l_tgt]
        :return:
        """
        trg_pad_mask = (tgt != self.PAD).unsqueeze(1).unsqueeze(3).byte()

        tgt_len = tgt.size(1)
        trg_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.uint8, device=src.device))

        tgt_mask = trg_pad_mask & trg_sub_mask
        return None, tgt_mask

    def decode(self, input, feature, src_mask, tgt_mask):
        # main process of transformer decoder.
        x = self.embedding(input)
        x = self.positional_encoding(x)
        for i, layer in enumerate(self.layers):
            x = layer(x, feature, src_mask, tgt_mask)
        x = self.norm(x)
        return self.fc(x)

    def greedy_forward(self, SOS, feature, mask):
        input = SOS
        output = None
        for i in range(self.max_length+1):
            _, target_mask = self.make_mask(feature, input)
            out = self.decode(input, feature, None, target_mask)
            #out = self.decoder(input, feature, None, target_mask)
            output = out
            prob = F.softmax(out, dim=-1)
            _, next_word = torch.max(prob, dim=-1)
            input = torch.cat([input, next_word[:, -1].unsqueeze(-1)], dim=1)
        return output

    def forward_train(self, feat, out_enc, targets_dict, img_metas=None):
        # x is token of label
        # feat is feature after backbone before pe.
        # out_enc is feature after pe.
        device = feat.device
        if isinstance(targets_dict, dict):
            padded_targets = targets_dict['padded_targets'].to(device)
        else:
            padded_targets = targets_dict.to(device)

        src_mask = None
        _, tgt_mask = self.make_mask(out_enc, padded_targets[:,:-1])
        return self.decode(padded_targets[:, :-1], out_enc, src_mask, tgt_mask)

    def forward_test(self, feat, out_enc, img_metas):
        src_mask = None
        batch_size = out_enc.shape[0]
        SOS = torch.zeros(batch_size).long().to(out_enc.device)
        SOS[:] = self.SOS
        SOS = SOS.unsqueeze(1)
        output = self.greedy_forward(SOS, out_enc, src_mask)
        return output

    def forward(self,
                feat,
                out_enc,
                targets_dict=None,
                img_metas=None,
                train_mode=True):
        self.train_mode = train_mode
        if train_mode:
            return self.forward_train(feat, out_enc, targets_dict, img_metas)

        return self.forward_test(feat, out_enc, img_metas)


@DECODERS.register_module()
class TableMasterDecoder(BaseDecoder):
    """
    Split to two transformer header at the last layer.
    Cls_layer is used to structure token classification.
    Bbox_layer is used to regress bbox coord.
    """
    def __init__(self,
                 N,
                 decoder,
                 d_model,
                 num_classes,
                 start_idx,
                 padding_idx,
                 max_seq_len,
                 ):
        super(TableMasterDecoder, self).__init__()
        self.layers = clones(DecoderLayer(**decoder), N-1)
        self.cls_layer = clones(DecoderLayer(**decoder), 1)
        self.bbox_layer = clones(DecoderLayer(**decoder), 1)
        self.cls_fc = nn.Linear(d_model, num_classes)
        self.bbox_fc = nn.Sequential(
            nn.Linear(d_model, 4),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(decoder.size)
        self.embedding = Embeddings(d_model=d_model, vocab=num_classes)
        self.positional_encoding = PositionalEncoding(d_model=d_model)

        self.SOS = start_idx
        self.PAD = padding_idx
        self.max_length = max_seq_len

    def make_mask(self, src, tgt):
        """
        Make mask for self attention.
        :param src: [b, c, h, l_src]
        :param tgt: [b, l_tgt]
        :return:
        """
        trg_pad_mask = (tgt != self.PAD).unsqueeze(1).unsqueeze(3).byte()

        tgt_len = tgt.size(1)
        trg_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.uint8, device=src.device))

        tgt_mask = trg_pad_mask & trg_sub_mask
        return None, tgt_mask

    def decode(self, input, feature, src_mask, tgt_mask):
        # main process of transformer decoder.
        x = self.embedding(input)
        x = self.positional_encoding(x)

        # origin transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x, feature, src_mask, tgt_mask)

        # cls head
        for layer in self.cls_layer:
            cls_x = layer(x, feature, src_mask, tgt_mask)
        cls_x = self.norm(cls_x)

        # bbox head
        for layer in self.bbox_layer:
            bbox_x = layer(x, feature, src_mask, tgt_mask)
        bbox_x = self.norm(bbox_x)

        return self.cls_fc(cls_x), self.bbox_fc(bbox_x)

    def greedy_forward(self, SOS, feature, mask):
        input = SOS
        output = None
        for i in range(self.max_length+1):
            _, target_mask = self.make_mask(feature, input)
            out, bbox_output = self.decode(input, feature, None, target_mask)
            output = out
            prob = F.softmax(out, dim=-1)
            _, next_word = torch.max(prob, dim=-1)
            input = torch.cat([input, next_word[:, -1].unsqueeze(-1)], dim=1)
        return output, bbox_output

    def forward_train(self, feat, out_enc, targets_dict, img_metas=None):
        # x is token of label
        # feat is feature after backbone before pe.
        # out_enc is feature after pe.
        device = feat.device
        if isinstance(targets_dict, dict):
            padded_targets = targets_dict['padded_targets'].to(device)
        else:
            padded_targets = targets_dict.to(device)

        src_mask = None
        _, tgt_mask = self.make_mask(out_enc, padded_targets[:,:-1])
        return self.decode(padded_targets[:, :-1], out_enc, src_mask, tgt_mask)

    def forward_test(self, feat, out_enc, img_metas):
        src_mask = None
        batch_size = out_enc.shape[0]
        SOS = torch.zeros(batch_size).long().to(out_enc.device)
        SOS[:] = self.SOS
        SOS = SOS.unsqueeze(1)
        output, bbox_output = self.greedy_forward(SOS, out_enc, src_mask)
        return output, bbox_output

    def forward(self,
                feat,
                out_enc,
                targets_dict=None,
                img_metas=None,
                train_mode=True):
        self.train_mode = train_mode
        if train_mode:
            return self.forward_train(feat, out_enc, targets_dict, img_metas)

        return self.forward_test(feat, out_enc, img_metas)


@DECODERS.register_module()
class TableMasterConcatDecoder(BaseDecoder):
    """
    Split to two transformer header at the last layer.
    Cls_layer is used to structure token classification.
    Bbox_layer is used to regress bbox coord.
    """
    def __init__(self,
                 N,
                 decoder,
                 d_model,
                 num_classes,
                 start_idx,
                 padding_idx,
                 max_seq_len,
                 ):
        super(TableMasterConcatDecoder, self).__init__()
        self.layers = clones(DecoderLayer(**decoder), N-1)
        self.cls_layer = clones(DecoderLayer(**decoder), 1)
        self.bbox_layer = clones(DecoderLayer(**decoder), 1)
        self.cls_fc = nn.Linear(d_model, num_classes)
        self.bbox_fc = nn.Sequential(
            nn.Linear(d_model, 4),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(decoder.size)
        self.embedding = Embeddings(d_model=d_model, vocab=num_classes)
        self.positional_encoding = PositionalEncoding(d_model=d_model)

        self.SOS = start_idx
        self.PAD = padding_idx
        self.max_length = max_seq_len

    def make_mask(self, src, tgt):
        """
        Make mask for self attention.
        :param src: [b, c, h, l_src]
        :param tgt: [b, l_tgt]
        :return:
        """
        trg_pad_mask = (tgt != self.PAD).unsqueeze(1).unsqueeze(3).byte()

        tgt_len = tgt.size(1)
        trg_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.uint8, device=src.device))

        tgt_mask = trg_pad_mask & trg_sub_mask
        return None, tgt_mask

    def decode(self, input, feature, src_mask, tgt_mask):
        # main process of transformer decoder.
        x = self.embedding(input)
        x = self.positional_encoding(x)

        # x_list = []
        cls_x_list = []
        bbox_x_list = []

        # origin transformer layers
        for i, layer in enumerate(self.layers):
            x = layer(x, feature, src_mask, tgt_mask)

        # cls head
        for layer in self.cls_layer:
            cls_x = layer(x, feature, src_mask, tgt_mask)
            cls_x_list.append(cls_x)
        cls_x = torch.cat(cls_x_list, dim=-1)
        cls_x = self.norm(cls_x)

        # bbox head
        for layer in self.bbox_layer:
            bbox_x = layer(x, feature, src_mask, tgt_mask)
            bbox_x_list.append(bbox_x)
        bbox_x = torch.cat(bbox_x_list, dim=-1)
        bbox_x = self.norm(bbox_x)

        return self.cls_fc(cls_x), self.bbox_fc(bbox_x)

    def greedy_forward(self, SOS, feature, mask):
        input = SOS
        output = None
        for i in range(self.max_length+1):
            _, target_mask = self.make_mask(feature, input)
            out, bbox_output = self.decode(input, feature, None, target_mask)
            output = out
            prob = F.softmax(out, dim=-1)
            _, next_word = torch.max(prob, dim=-1)
            input = torch.cat([input, next_word[:, -1].unsqueeze(-1)], dim=1)
        return output, bbox_output

    def forward_train(self, feat, out_enc, targets_dict, img_metas=None):
        # x is token of label
        # feat is feature after backbone before pe.
        # out_enc is feature after pe.
        device = feat.device
        if isinstance(targets_dict, dict):
            padded_targets = targets_dict['padded_targets'].to(device)
        else:
            padded_targets = targets_dict.to(device)

        src_mask = None
        _, tgt_mask = self.make_mask(out_enc, padded_targets[:,:-1])
        return self.decode(padded_targets[:, :-1], out_enc, src_mask, tgt_mask)

    def forward_test(self, feat, out_enc, img_metas):
        src_mask = None
        batch_size = out_enc.shape[0]
        SOS = torch.zeros(batch_size).long().to(out_enc.device)
        SOS[:] = self.SOS
        SOS = SOS.unsqueeze(1)
        output, bbox_output = self.greedy_forward(SOS, out_enc, src_mask)
        return output, bbox_output

    def forward(self,
                feat,
                out_enc,
                targets_dict=None,
                img_metas=None,
                train_mode=True):
        self.train_mode = train_mode
        if train_mode:
            return self.forward_train(feat, out_enc, targets_dict, img_metas)

        return self.forward_test(feat, out_enc, img_metas)