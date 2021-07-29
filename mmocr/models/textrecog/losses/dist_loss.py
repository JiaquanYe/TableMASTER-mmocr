import torch.nn as nn

from mmdet.models.builder import LOSSES

@LOSSES.register_module()
class DistLoss(nn.Module):
    """Implementation of loss module for table master bbox regression branch
    with Distance loss.

    Args:
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ('none', 'mean', 'sum').
    """
    def __init__(self, reduction='none'):
        super().__init__()
        assert isinstance(reduction, str)
        assert reduction in ['none', 'mean', 'sum']
        self.dist_loss = self.build_loss(reduction)

    def build_loss(self, reduction, **kwargs):
        raise NotImplementedError

    def format(self, outputs, targets_dict):
        raise NotImplementedError

    def forward(self, outputs, targets_dict, img_metas=None):
        outputs, targets = self.format(outputs, targets_dict)
        loss_dist = self.dist_loss(outputs, targets.to(outputs.device))
        losses = dict(loss_dist=loss_dist)
        return losses


@LOSSES.register_module()
class TableL1Loss(DistLoss):
    """Implementation of L1 loss module for table master bbox branch."""
    def __init__(self,
                 reduction='sum',
                 **kwargs):
        super().__init__(reduction)
        self.lambda_horizon = 1.
        self.lambda_vertical = 1.
        self.eps = 1e-9
        # use reduction sum, and divide bbox_mask's nums, to get mean loss.
        try:
            assert reduction == 'sum'
        except AssertionError:
            raise ('Table L1 loss in bbox branch should keep reduction is sum.')

    def build_loss(self, reduction):
        return nn.L1Loss(reduction=reduction)

    def format(self, outputs, targets_dict):
        # target in calculate loss, start from idx 1.
        bboxes = targets_dict['bbox'][:, 1:, :].to(outputs.device)  # bxLx4
        bbox_masks = targets_dict['bbox_masks'][:, 1:].unsqueeze(-1).to(outputs.device)  # bxLx1
        # mask empty-bbox or non-bbox structure token's bbox.
        masked_outputs = outputs * bbox_masks
        masked_bboxes = bboxes * bbox_masks
        return masked_outputs, masked_bboxes, bbox_masks

    def forward(self, outputs, targets_dict, img_metas=None):
        outputs, targets, bbox_masks = self.format(outputs, targets_dict)
        # horizon loss (x and width)
        horizon_sum_loss = self.dist_loss(outputs[:, :, 0::2].contiguous(), targets[:, :, 0::2].contiguous())
        horizon_loss = horizon_sum_loss / (bbox_masks.sum() + self.eps)
        # vertical loss (y and height)
        vertical_sum_loss = self.dist_loss(outputs[:, :, 1::2].contiguous(), targets[:, :, 1::2].contiguous())
        vertical_loss = vertical_sum_loss / (bbox_masks.sum() + self.eps)

        losses = {'horizon_bbox_loss': horizon_loss, 'vertical_bbox_loss': vertical_loss}

        return losses
