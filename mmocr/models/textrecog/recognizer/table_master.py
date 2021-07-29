import torch
import torch.nn as nn

from mmdet.models.builder import DETECTORS, build_backbone, build_loss

from .encode_decode_recognizer import EncodeDecodeRecognizer


def visual_pred_bboxes(img_metas, results):
    """
    visual after normalized bbox in results.
    :param results:
    :return:
    """
    import os
    import cv2
    import numpy as np

    for img_meta, result in zip(img_metas, results):
        img = cv2.imread(img_meta['filename'])
        bboxes = result['bbox']
        save_path = '/data_0/cache/{}_pred_bbox.jpg'. \
            format(os.path.basename(img_meta['filename']).split('.')[0])

        # x,y,w,h to x,y,x,y
        new_bboxes = np.empty_like(bboxes)
        new_bboxes[..., 0] = bboxes[..., 0] - bboxes[..., 2] / 2
        new_bboxes[..., 1] = bboxes[..., 1] - bboxes[..., 3] / 2
        new_bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2] / 2
        new_bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3] / 2
        # draw
        for new_bbox in new_bboxes:
            img = cv2.rectangle(img, (int(new_bbox[0]), int(new_bbox[1])),
                                (int(new_bbox[2]), int(new_bbox[3])), (0, 255, 0), thickness=1)
        cv2.imwrite(save_path, img)

@DETECTORS.register_module()
class TABLEMASTER(EncodeDecodeRecognizer):
    # need to inherit BaseRecognizer or EncodeDecodeRecognizer in mmocr
    def __init__(self,
                 preprocessor=None,
                 backbone=None,
                 encoder=None,
                 decoder=None,
                 loss=None,
                 bbox_loss=None,
                 label_convertor=None,
                 train_cfg=None,
                 test_cfg=None,
                 max_seq_len=40,
                 pretrained=None):
        super(TABLEMASTER, self).__init__(preprocessor,
                                       backbone,
                                       encoder,
                                       decoder,
                                       loss,
                                       label_convertor,
                                       train_cfg,
                                       test_cfg,
                                       max_seq_len,
                                       pretrained)
        # build bbox loss
        self.bbox_loss = build_loss(bbox_loss)

    def init_weights(self, pretrained=None):
        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)

    def forward_train(self, img, img_metas):
        """
        Args:
            img (tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A list of image info dict where each dict
                contains: 'img_shape', 'filename', and may also contain
                'ori_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, tensor]: A dictionary of loss components.
        """
        feat = self.extract_feat(img)
        feat = feat[-1]

        targets_dict = self.label_convertor.str_bbox_format(img_metas)

        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat)

        out_dec, out_bbox = self.decoder(
            feat, out_enc, targets_dict, img_metas, train_mode=True)

        loss_inputs = (
            out_dec,
            targets_dict,
            img_metas,
        )
        losses = self.loss(*loss_inputs)

        bbox_loss_inputs = (
            out_bbox,
            targets_dict,
            img_metas,
        )
        bbox_losses = self.bbox_loss(*bbox_loss_inputs)

        losses.update(bbox_losses)

        return losses

    def simple_test(self, img, img_metas, **kwargs):
        """Test function with test time augmentation.

        Args:
            imgs (torch.Tensor): Image input tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """
        feat = self.extract_feat(img)
        feat = feat[-1]

        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat)

        out_dec, out_bbox = self.decoder(
            feat, out_enc, None, img_metas, train_mode=False)

        strings, scores, pred_bboxes = \
            self.label_convertor.output_format(out_dec, out_bbox, img_metas)

        # flatten batch results
        results = []
        for string, score, pred_bbox in zip(strings, scores, pred_bboxes):
            results.append(dict(text=string, score=score, bbox=pred_bbox))

        # visual_pred_bboxes(img_metas, results)

        return results