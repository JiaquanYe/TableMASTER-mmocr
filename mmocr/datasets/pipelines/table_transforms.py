from mmdet.datasets.builder import PIPELINES

import os
import cv2
import random
import numpy as np

def visual_table_resized_bbox(results):
    bboxes = results['img_info']['bbox']
    img = results['img']
    for bbox in bboxes:
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), thickness=1)
    return img

def visual_table_xywh_bbox(results):
    img = results['img']
    bboxes = results['img_info']['bbox']
    for bbox in bboxes:
        draw_bbox = np.empty_like(bbox)
        draw_bbox[0] = bbox[0] - bbox[2] / 2
        draw_bbox[1] = bbox[1] - bbox[3] / 2
        draw_bbox[2] = bbox[0] + bbox[2] / 2
        draw_bbox[3] = bbox[1] + bbox[3] / 2
        img = cv2.rectangle(img, (int(draw_bbox[0]), int(draw_bbox[1])), (int(draw_bbox[2]), int(draw_bbox[3])), (0, 255, 0), thickness=1)
    return img

@PIPELINES.register_module()
class TableResize:
    """Image resizing and padding for Table Recognition OCR, Table Structure Recognition.

    Args:
        height (int | tuple(int)): Image height after resizing.
        min_width (none | int | tuple(int)): Image minimum width
            after resizing.
        max_width (none | int | tuple(int)): Image maximum width
            after resizing.
        keep_aspect_ratio (bool): Keep image aspect ratio if True
            during resizing, Otherwise resize to the size height *
            max_width.
        img_pad_value (int): Scalar to fill padding area.
        width_downsample_ratio (float): Downsample ratio in horizontal
            direction from input image to output feature.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.
    """

    def __init__(self,
                 img_scale=None,
                 min_size=None,
                 ratio_range=None,
                 interpolation=None,
                 keep_ratio=True,
                 long_size=None):
        self.img_scale = img_scale
        self.min_size = min_size
        self.ratio_range = ratio_range
        self.interpolation = cv2.INTER_LINEAR
        self.long_size = long_size
        self.keep_ratio = keep_ratio

    def _get_resize_scale(self, w, h):
        if self.keep_ratio:
            if self.img_scale is None and isinstance(self.ratio_range, list):
                choice_ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])
                return (int(w * choice_ratio), int(h * choice_ratio))
            elif isinstance(self.img_scale, tuple) and -1 in self.img_scale:
                if self.img_scale[0] == -1:
                    resize_w = w / h * self.img_scale[1]
                    return (int(resize_w), self.img_scale[1])
                else:
                    resize_h = h / w * self.img_scale[0]
                    return (self.img_scale[0], int(resize_h))
            else:
                return (int(w), int(h))
        else:
            if isinstance(self.img_scale, tuple):
                return self.img_scale
            else:
                raise NotImplementedError

    def _resize_bboxes(self, results):
        img_shape = results['img_shape']
        if 'img_info' in results.keys():
            # train and validate phase
            if results['img_info'].get('bbox', None) is not None:
                bboxes = results['img_info']['bbox']
                scale_factor = results['scale_factor']
                # bboxes[..., 0::2], bboxes[..., 1::2] = \
                #     bboxes[..., 0::2] * scale_factor[1], bboxes[..., 1::2] * scale_factor[0]
                bboxes[..., 0::2] = np.clip(bboxes[..., 0::2] * scale_factor[1], 0, img_shape[1]-1)
                bboxes[..., 1::2] = np.clip(bboxes[..., 1::2] * scale_factor[0], 0, img_shape[0]-1)
                results['img_info']['bbox'] = bboxes
            else:
                raise ValueError('results should have bbox keys.')
        else:
            # testing phase
            pass

    def _resize_img(self, results):
        img = results['img']
        h, w, _ = img.shape

        if self.min_size is not None:
            if w > h:
                w = self.min_size / h * w
                h = self.min_size
            else:
                h = self.min_size / w * h
                w = self.min_size

        if self.long_size is not None:
            if w < h:
                w = self.long_size / h * w
                h = self.long_size
            else:
                h = self.long_size / w * h
                w = self.long_size

        img_scale = self._get_resize_scale(w, h)
        resize_img = cv2.resize(img, img_scale, interpolation=self.interpolation)
        scale_factor = (resize_img.shape[0] / img.shape[0], resize_img.shape[1] / img.shape[1])

        results['img'] = resize_img
        results['img_shape'] = resize_img.shape
        results['pad_shape'] = resize_img.shape
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def __call__(self, results):
        self._resize_img(results)
        self._resize_bboxes(results)
        return results


@PIPELINES.register_module()
class TablePad:
    """Pad the image & mask.
    Two padding modes:
    (1) pad to fixed size.
    (2) pad to the minium size that is divisible by some number.
    """
    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=None,
                 keep_ratio=False,
                 return_mask=False,
                 mask_ratio=2,
                 train_state=True,
                 ):
        self.size = size[::-1]
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.keep_ratio = keep_ratio
        self.return_mask = return_mask
        self.mask_ratio = mask_ratio
        self.training = train_state
        # only one of size or size_divisor is valid.
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad(self, img, size, pad_val):
        if not isinstance(size, tuple):
            raise NotImplementedError

        if len(size) < len(img.shape):
            shape = size + (img.shape[-1], )
        else:
            shape = size

        pad = np.empty(shape, dtype=img.dtype)
        pad[...] = pad_val

        h, w = img.shape[:2]
        size_w, size_h = size[:2]
        if h > size_h or w > size_w:
            if self.keep_ratio:
                if h / size_h > w / size_w:
                    size = (int(w / h * size_h), size_h)
                else:
                    size = (size_w, int(h / w * size_w))
            img = cv2.resize(img, size[::-1], cv2.INTER_LINEAR)
        pad[:img.shape[0], :img.shape[1], ...] = img
        if self.return_mask:
            mask = np.empty(size, dtype=img.dtype)
            mask[...] = 0
            mask[:img.shape[0], :img.shape[1]] = 1

            # mask_ratio is mean stride of backbone in (height, width)
            if isinstance(self.mask_ratio, int):
                mask = mask[::self.mask_ratio, ::self.mask_ratio]
            elif isinstance(self.mask_ratio, tuple):
                mask = mask[::self.mask_ratio[0], ::self.mask_ratio[1]]
            else:
                raise NotImplementedError

            mask = np.expand_dims(mask, axis=0)
        else:
            mask = None
        return pad, mask

    def _divisor(self, img, size_divisor, pad_val):
        pass

    def _pad_img(self, results):
        if self.size is not None:
            padded_img, mask = self._pad(results['img'], self.size, self.pad_val)
        elif self.size_divisor is not None:
            raise NotImplementedError
        results['img'] = padded_img
        results['mask'] = mask
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        self._pad_img(results)
        #visual_img = visual_table_resized_bbox(results)
        #cv2.imwrite('/data_0/cache/{}_visual.jpg'.format(os.path.basename(results['filename']).split('.')[0]), visual_img)
        # if self.training:
            # scaleBbox(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(size={}, size_divisor={}, pad_val={})'.format(
            self.size, self.size_divisor, self.pad_val)
        return repr_str


def xyxy2xywh(bboxes):
    """
    Convert coord (x1,y1,x2,y2) to (x,y,w,h).
    where (x1,y1) is top-left, (x2,y2) is bottom-right.
    (x,y) is bbox center and (w,h) is width and height.
    :param bboxes: (x1, y1, x2, y2)
    :return:
    """
    new_bboxes = np.empty_like(bboxes)
    new_bboxes[..., 0] = (bboxes[..., 0] + bboxes[..., 2]) / 2 # x center
    new_bboxes[..., 1] = (bboxes[..., 1] + bboxes[..., 3]) / 2 # y center
    new_bboxes[..., 2] = bboxes[..., 2] - bboxes[..., 0] # width
    new_bboxes[..., 3] = bboxes[..., 3] - bboxes[..., 1] # height
    return new_bboxes


def normalize_bbox(bboxes, img_shape):
    bboxes[..., 0], bboxes[..., 2] = bboxes[..., 0] / img_shape[1], bboxes[..., 2] / img_shape[1]
    bboxes[..., 1], bboxes[..., 3] = bboxes[..., 1] / img_shape[0], bboxes[..., 3] / img_shape[0]
    return bboxes


@PIPELINES.register_module()
class TableBboxEncode:
    """Encode table bbox for training.
    convert coord (x1,y1,x2,y2) to (x,y,w,h)
    normalize to (0,1)
    adjust key 'bbox' and 'bbox_mask' location in dictionary 'results'
    """
    def __init__(self):
        pass

    def __call__(self, results):
        bboxes = results['img_info']['bbox']
        bboxes = xyxy2xywh(bboxes)
        img_shape = results['img'].shape
        bboxes = normalize_bbox(bboxes, img_shape)
        flag = self.check_bbox_valid(bboxes)
        if not flag:
            print('Box invalid in {}'.format(results['filename']))
        results['img_info']['bbox'] = bboxes
        self.adjust_key(results)
        # self.visual_normalized_bbox(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

    def check_bbox_valid(self, bboxes):
        low = (bboxes >= 0.) * 1
        high = (bboxes <= 1.) * 1
        matrix = low + high
        for idx, m in enumerate(matrix):
            if m.sum() != 8:
                return False
        return True

    def visual_normalized_bbox(self, results):
        """
        visual after normalized bbox in results.
        :param results:
        :return:
        """
        save_path = '/data_0/cache/{}_normalized.jpg'.\
            format(os.path.basename(results['filename']).split('.')[0])
        img = results['img']
        img_shape = img.shape
        # x,y,w,h
        bboxes = results['img_info']['bbox']
        bboxes[..., 0::2] = bboxes[..., 0::2] * img_shape[1]
        bboxes[..., 1::2] = bboxes[..., 1::2] * img_shape[0]
        # x,y,x,y
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

    def adjust_key(self, results):
        """
        Adjust key 'bbox' and 'bbox_mask' location in dictionary 'results'.
        :param results:
        :return:
        """
        bboxes = results['img_info'].pop('bbox')
        bboxes_masks = results['img_info'].pop('bbox_masks')
        results['bbox'] = bboxes
        results['bbox_masks'] = bboxes_masks
        return results




