import torch

import mmocr.utils as utils
from mmocr.models.builder import CONVERTORS
from .base import BaseConvertor

import numpy as np

@CONVERTORS.register_module()
class MasterConvertor(BaseConvertor):
    """Convert between text, index and tensor for encoder-decoder based
    pipeline.

    Args:
        dict_type (str): Type of dict, should be one of {'DICT36', 'DICT90'}.
        dict_file (None|str): Character dict file path. If not none,
            higher priority than dict_type.
        dict_list (None|list[str]): Character list. If not none, higher
            priority than dict_type, but lower than dict_file.
        with_unknown (bool): If True, add `UKN` token to class.
        max_seq_len (int): Maximum sequence length of label.
        lower (bool): If True, convert original string to lower case.
        start_end_same (bool): Whether use the same index for
            start and end token or not. Default: True.
    """

    def __init__(self,
                 dict_type='DICT90',
                 dict_file=None,
                 dict_list=None,
                 with_unknown=True,
                 max_seq_len=40,
                 lower=False,
                 start_end_same=True,
                 **kwargs):
        super().__init__(dict_type, dict_file, dict_list)
        assert isinstance(with_unknown, bool)
        assert isinstance(max_seq_len, int)
        assert isinstance(lower, bool)

        self.with_unknown = with_unknown
        self.max_seq_len = max_seq_len
        self.lower = lower
        self.start_end_same = start_end_same

        self.update_dict()

    def update_dict(self):
        start_token = '<SOS>'
        end_token = '<EOS>'
        unknown_token = '<UKN>'
        padding_token = '<PAD>'

        # unknown
        self.unknown_idx = None
        if self.with_unknown:
            self.idx2char.append(unknown_token)
            self.unknown_idx = len(self.idx2char) - 1

        # SOS/EOS
        self.idx2char.append(start_token)
        self.start_idx = len(self.idx2char) - 1
        if not self.start_end_same:
            self.idx2char.append(end_token)
        self.end_idx = len(self.idx2char) - 1

        # padding
        self.idx2char.append(padding_token)
        self.padding_idx = len(self.idx2char) - 1

        # update char2idx
        self.char2idx = {}
        for idx, char in enumerate(self.idx2char):
            self.char2idx[char] = idx

    def str2tensor(self, strings):
        """
        Convert text-string into tensor.
        Args:
            strings (list[str]): ['hello', 'world']
        Returns:
            dict (str: Tensor | list[tensor]):
                tensors (list[Tensor]): [torch.Tensor([1,2,3,3,4]),
                                                    torch.Tensor([5,4,6,3,7])]
                padded_targets (Tensor(bsz * max_seq_len))
        """
        # ordinary OCR task strings is list of str, but table master is list of list.
        assert utils.is_type_list(strings, str) or utils.is_type_list(strings, list)

        tensors, padded_targets = [], []
        indexes = self.str2idx(strings)
        for index in indexes:
            tensor = torch.LongTensor(index)
            tensors.append(tensor)
            # target tensor for loss
            src_target = torch.LongTensor(tensor.size(0) + 2).fill_(0)
            src_target[-1] = self.end_idx
            src_target[0] = self.start_idx
            src_target[1:-1] = tensor
            padded_target = (torch.ones(self.max_seq_len) *
                             self.padding_idx).long()
            char_num = src_target.size(0)
            if char_num > self.max_seq_len:
                # TODO:大于max_seq_len-2的，应该跳过？检查dataset有没处理。
                padded_target = src_target[:self.max_seq_len]
            else:
                # TODO:这里是最后一个是PAD token，而不是EOS，与FASTOCR不同，其最后一个是EOS.
                padded_target[:char_num] = src_target
            padded_targets.append(padded_target)
        padded_targets = torch.stack(padded_targets, 0).long()

        return {'targets': tensors, 'padded_targets': padded_targets}

    def tensor2idx(self, outputs, img_metas=None):
        """
        Convert output tensor to text-index
        Args:
            outputs (tensor): model outputs with size: N * T * C
            img_metas (list[dict]): Each dict contains one image info.
        Returns:
            indexes (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]]
            scores (list[list[float]]): [[0.9,0.8,0.95,0.97,0.94],
                                         [0.9,0.9,0.98,0.97,0.96]]
        """
        batch_size = outputs.size(0)
        ignore_indexes = [self.padding_idx]
        indexes, scores = [], []
        for idx in range(batch_size):
            seq = outputs[idx, :, :]
            seq = seq.softmax(-1)
            max_value, max_idx = torch.max(seq, -1)
            str_index, str_score = [], []
            output_index = max_idx.cpu().detach().numpy().tolist()
            output_score = max_value.cpu().detach().numpy().tolist()
            for char_index, char_score in zip(output_index, output_score):
                if char_index in ignore_indexes:
                    continue
                if char_index == self.end_idx:
                    break
                str_index.append(char_index)
                str_score.append(char_score)

            indexes.append(str_index)
            scores.append(str_score)

        return indexes, scores


@CONVERTORS.register_module()
class TableMasterConvertor(MasterConvertor):
    """Similarity with MasterConvertor, but add key 'bbox' and 'bbox_masks'.
    'bbox' and 'bbox_mask' need to the same length as 'text'.
    This convert use the alphabet extract by data_preprocess.py of table_recognition.

    Args:
        dict_type (str): Type of dict, should be one of {'DICT36', 'DICT90'}.
        dict_file (None|str): Character dict file path. If not none,
            higher priority than dict_type.
        dict_list (None|list[str]): Character list. If not none, higher
            priority than dict_type, but lower than dict_file.
        with_unknown (bool): If True, add `UKN` token to class.
        max_seq_len (int): Maximum sequence length of label.
        lower (bool): If True, convert original string to lower case.
        start_end_same (bool): Whether use the same index for
            start and end token or not. Default: True.
    """
    def __init__(self,
                 dict_type='DICT90',
                 dict_file=None,
                 dict_list=None,
                 with_unknown=True,
                 max_seq_len=500,
                 lower=False,
                 start_end_same=False,
                 **kwargs
                 ):
        self.start_end_same = start_end_same
        self.checker()
        super().__init__(dict_type, dict_file, dict_list, with_unknown, max_seq_len, lower, start_end_same)
        # self.deal_alphabet_span_token()

    def checker(self):
        try:
            assert self.start_end_same is False
        except AssertionError:
            raise

    # def deal_alphabet_span_token(self):
    #     """
    #     Modify the self.idx2char in base, which read by alphabet file.
    #     Reading alphabet will strip space char in the head, eg. ' colspan' -> 'colspan'.
    #     This function will modify self.idx2char and self.char2idx,
    #     to add space char in span-style after reading alphabet.
    #
    #     PS:
    #         If use line.strip('\n') in reading alphabet file in base.py, comment this function.
    #     :return:
    #     """
    #     # modify idx2char
    #     new_alphabet = []
    #     for char in self.idx2char:
    #         char = char.replace('colspan=', ' colspan=')
    #         char = char.replace('rowspan=', ' rowspan=')
    #         new_alphabet.append(char)
    #     self.idx2char = new_alphabet
    #     # modify char2idx
    #     new_dict = {}
    #     for idx, char in enumerate(self.idx2char):
    #         new_dict[char] = idx
    #     self.char2idx = new_dict
    #     import pdb;pdb.set_trace()

    def _pad_bbox(self, bboxes):
        padded_bboxes = []
        for bbox in bboxes:
            bbox = torch.from_numpy(bbox)
            bbox_pad = torch.Tensor([0., 0., 0., 0.]).float()
            padded_bbox = torch.zeros(self.max_seq_len, 4)
            padded_bbox[:] = bbox_pad
            padded_bbox[1:len(bbox)+1] = bbox
            padded_bboxes.append(padded_bbox)
        padded_bboxes = torch.stack(padded_bboxes, 0).float()
        return padded_bboxes

    def _pad_bbox_mask(self, bbox_masks):
        padded_bbox_masks = []
        for bbox_mask in bbox_masks:
            bbox_mask = torch.from_numpy(bbox_mask)
            bbox_mask_pad = torch.Tensor([0])
            padded_bbox_mask = torch.zeros(self.max_seq_len)
            padded_bbox_mask[:] = bbox_mask_pad
            padded_bbox_mask[1:len(bbox_mask)+1] = bbox_mask
            padded_bbox_masks.append(padded_bbox_mask)
        padded_bbox_masks = torch.stack(padded_bbox_masks, 0).long()
        return padded_bbox_masks

    def idx2str(self, indexes):
        """
        Similar with the 'idx2str' function of base, but use ',' to join the token list.
        :param indexes: (list[list[int]]): [[1,2,3,3,4], [5,4,6,3,7]].
        :return:
        """
        assert isinstance(indexes, list)
        strings = []
        for index in indexes:
            string = [self.idx2char[i] for i in index]
            # use ',' to join char list.
            string = ','.join(string)
            strings.append(string)

        return strings

    def _get_pred_bbox_mask(self, strings):
        """
        get the bbox mask by the pred strings results, where 1 means to output.
        <SOS>, <EOS>, <PAD> and <eb></eb> series set to 0, <td></td>, <td set to 1, others set to 0.
        :param strings: pred text list by cls branch.
        :return: pred bbox_mask
        """
        assert isinstance(strings, list)
        pred_bbox_masks = []
        SOS = self.idx2char[self.start_idx]
        EOS = self.idx2char[self.end_idx]
        PAD = self.idx2char[self.padding_idx]

        for string in strings:
            pred_bbox_mask = []
            char_list = string.split(',')
            for char in char_list:
                if char == EOS:
                    pred_bbox_mask.append(0)
                    break
                elif char == PAD:
                    pred_bbox_mask.append(0)
                    continue
                elif char == SOS:
                    pred_bbox_mask.append(0)
                    continue
                else:
                    if char == '<td></td>' or char == '<td':
                        pred_bbox_mask.append(1)
                    else:
                        pred_bbox_mask.append(0)
            pred_bbox_masks.append(pred_bbox_mask)

        return np.array(pred_bbox_masks)

    def _filter_invalid_bbox(self, output_bbox, pred_bbox_mask):
        """
        filter the invalid bboxes, use pred_bbox_masks and value not in [0,1].
        :param output_bbox:
        :param pred_bbox_mask:
        :return:
        """
        # filter bboxes coord out of [0,1]
        low_mask = (output_bbox >= 0.) * 1
        high_mask = (output_bbox <= 1.) * 1
        mask = np.sum((low_mask + high_mask), axis=1)
        value_mask = np.where(mask == 2*4, 1, 0)

        output_bbox_len = output_bbox.shape[0]
        pred_bbox_mask_len = pred_bbox_mask.shape[0]
        padded_pred_bbox_mask = np.zeros(output_bbox_len, dtype='int64')
        padded_pred_bbox_mask[:pred_bbox_mask_len] = pred_bbox_mask
        filtered_output_bbox = \
            output_bbox * np.expand_dims(value_mask, 1) * np.expand_dims(padded_pred_bbox_mask, 1)

        return filtered_output_bbox


    def _decode_bboxes(self, outputs_bbox, pred_bbox_masks, img_metas):
        """
        De-normalize and scale back the bbox coord.
        :param outputs_bbox:
        :param pred_bbox_masks:
        :param img_metas:
        :return:
        """
        pred_bboxes = []
        for output_bbox, pred_bbox_mask, img_meta in zip(outputs_bbox, pred_bbox_masks, img_metas):
            output_bbox = output_bbox.cpu().numpy()
            scale_factor = img_meta['scale_factor']
            pad_shape = img_meta['pad_shape']
            ori_shape = img_meta['ori_shape']

            output_bbox = self._filter_invalid_bbox(output_bbox, pred_bbox_mask)

            # de-normalize to pad shape
            # output_bbox[:, 0], output_bbox[:, 2] = output_bbox[:, 0] * pad_shape[1], output_bbox[:, 2] * pad_shape[1]
            # output_bbox[:, 1], output_bbox[:, 3] = output_bbox[:, 1] * pad_shape[0], output_bbox[:, 3] * pad_shape[0]
            output_bbox[:, 0::2] = output_bbox[:, 0::2] * pad_shape[1]
            output_bbox[:, 1::2] = output_bbox[:, 1::2] * pad_shape[0]

            # scale to origin shape
            # output_bbox[:, 0], output_bbox[:, 2] = output_bbox[:, 0] / scale_factor[1], output_bbox[:, 2] / scale_factor[1]
            # output_bbox[:, 1], output_bbox[:, 3] = output_bbox[:, 1] / scale_factor[0], output_bbox[:, 3] / scale_factor[0]
            output_bbox[:, 0::2] = output_bbox[:, 0::2] / scale_factor[1]
            output_bbox[:, 1::2] = output_bbox[:, 1::2] / scale_factor[0]

            pred_bboxes.append(output_bbox)

            return pred_bboxes

    def _adjsut_bboxes_len(self, bboxes, strings):
        new_bboxes = []
        for bbox, string in zip(bboxes, strings):
            string = string.split(',')
            string_len = len(string)
            bbox = bbox[:string_len, :]
            new_bboxes.append(bbox)
        return new_bboxes

    def _get_strings_scores(self, str_scores):
        """
        Calculate strings scores by averaging str_scores
        :param str_scores: softmax score of each char.
        :return:
        """
        strings_scores = []
        for str_score in str_scores:
            score = sum(str_score) / len(str_score)
            strings_scores.append(score)
        return strings_scores

    def str_bbox_format(self, img_metas):
        """
        Convert text-string into tensor.
        Pad 'bbox' and 'bbox_masks' to the same length as 'text'

        Args:
            img_metas (list[dict]):
                dict.keys() ['filename', 'ori_shape', 'img_shape', 'text', 'scale_factor', 'bbox', 'bbox_masks']
        Returns:
            dict (str: Tensor | list[tensor]):
                tensors (list[Tensor]): [torch.Tensor([1,2,3,3,4]),
                                                    torch.Tensor([5,4,6,3,7])]
                padded_targets (Tensor(bsz * max_seq_len))

                bbox (list[Tensor]):
                bbox_masks (Tensor):
        """

        # output of original str2tensor function(split by ',' in each string).
        gt_labels = [[char for char in img_meta['text'].split(',')] for img_meta in img_metas]
        tmp_dict = self.str2tensor(gt_labels)
        text_target = tmp_dict['targets']
        text_padded_target = tmp_dict['padded_targets']

        # pad bbox's length
        bboxes = [img_meta['bbox'] for img_meta in img_metas]
        bboxes = self._pad_bbox(bboxes)

        # pad bbox_mask's length
        bbox_masks = [img_meta['bbox_masks'] for img_meta in img_metas]
        bbox_masks = self._pad_bbox_mask(bbox_masks)

        format_dict = {'targets': text_target,
                        'padded_targets': text_padded_target,
                        'bbox': bboxes,
                        'bbox_masks': bbox_masks}

        return format_dict

    def output_format(self, outputs, out_bbox, img_metas=None):
        # cls_branch process
        str_indexes, str_scores = self.tensor2idx(outputs, img_metas)
        strings = self.idx2str(str_indexes)
        scores = self._get_strings_scores(str_scores)

        # bbox_branch process
        pred_bbox_masks = self._get_pred_bbox_mask(strings)
        pred_bboxes = self._decode_bboxes(out_bbox, pred_bbox_masks, img_metas)
        pred_bboxes = self._adjsut_bboxes_len(pred_bboxes, strings)

        return strings, scores, pred_bboxes
