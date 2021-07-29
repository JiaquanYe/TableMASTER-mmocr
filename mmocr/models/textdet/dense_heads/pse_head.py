from mmdet.models.builder import HEADS
from . import PANHead


@HEADS.register_module()
class PSEHead(PANHead):
    """The class for PANet head."""

    def __init__(
            self,
            in_channels,
            out_channels,
            text_repr_type='poly',  # 'poly' or 'quad'
            downsample_ratio=0.25,
            min_kernel_confidence=0.5,
            min_text_avg_confidence=0.85,
            loss=dict(type='PSELoss'),
            train_cfg=None,
            test_cfg=None):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            text_repr_type=text_repr_type,
            downsample_ratio=downsample_ratio,
            min_kernel_confidence=min_kernel_confidence,
            min_text_avg_confidence=min_text_avg_confidence,
            loss=loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg)
