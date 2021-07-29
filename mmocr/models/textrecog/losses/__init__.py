from .ce_loss import CELoss, SARLoss, TFLoss, MASTERTFLoss
from .ctc_loss import CTCLoss
from .seg_loss import SegLoss
from .dist_loss import TableL1Loss

__all__ = ['CELoss', 'SARLoss', 'CTCLoss', 'TFLoss', 'SegLoss', 'MASTERTFLoss', 'TableL1Loss']
