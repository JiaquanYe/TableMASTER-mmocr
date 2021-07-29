import mmcv
import numpy as np

from mmdet.core import BitmapMasks, PolygonMasks
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.loading import LoadAnnotations, LoadImageFromFile

import six
import lmdb
import torch
from PIL import Image

@PIPELINES.register_module()
class LoadTextAnnotations(LoadAnnotations):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True):
        super().__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            poly2mask=poly2mask)

    def process_polygons(self, polygons):
        """Convert polygons to list of ndarray and filter invalid polygons.

        Args:
            polygons (list[list]): Polygons of one instance.

        Returns:
            list[numpy.ndarray]: Processed polygons.
        """

        polygons = [np.array(p).astype(np.float32) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons

    def _load_masks(self, results):
        ann_info = results['ann_info']
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = ann_info['masks']
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h,
                w)
        gt_masks_ignore = ann_info.get('masks_ignore', None)
        if gt_masks_ignore is not None:
            if self.poly2mask:
                gt_masks_ignore = BitmapMasks(
                    [self._poly2mask(mask, h, w) for mask in gt_masks_ignore],
                    h, w)
            else:
                gt_masks_ignore = PolygonMasks([
                    self.process_polygons(polygons)
                    for polygons in gt_masks_ignore
                ], h, w)
            results['gt_masks_ignore'] = gt_masks_ignore
            results['mask_fields'].append('gt_masks_ignore')

        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results


@PIPELINES.register_module()
class LoadImageFromNdarray(LoadImageFromFile):
    """Load an image from np.ndarray.

    Similar with :obj:`LoadImageFromFile`, but the image read from
    ``results['img']``, which is np.ndarray.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        assert results['img'].dtype == 'uint8'

        img = results['img']
        if self.color_type == 'grayscale' and img.shape[2] == 3:
            img = mmcv.bgr2gray(img, keepdim=True)
        if self.color_type == 'color' and img.shape[2] == 1:
            img = mmcv.gray2bgr(img)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results


@PIPELINES.register_module()
class LoadImageFromLMDB(object):
    """Load an image from lmdb file.

    Similar with :obj:'LoadImageFromFile', but the image read from
    "results['img_info']['filename']", which is a data index of lmdb file.
    """
    def __init__(self, color_type='color'):
        self.color_type = color_type
        self.env = None
        self.txn = None

        orig_func = torch.utils.data._utils.worker._worker_loop

        def wl(*args, **kwargs):
            print('Running modified workers in dataloader.')
            ret = orig_func(*args, **kwargs)
            if self.env is not None:
                self.env.close()
                self.env = None
                print('Lmdb Loader closed.')

        torch.utils.data._utils.worker._worker_loop = wl

    def __call__(self, results):
        lmdb_index = results['img_info']['filename']
        data_root = results['img_info']['ann_file']
        img_key = b'image-%09d' % int(lmdb_index)

        if self.env is None:
            env = lmdb.open(data_root, readonly=True)
            self.env = env
        else:
            env = self.env
        if self.txn is None:
            txn = env.begin(write=False)
            self.txn = txn
        else:
            txn = self.txn

        # read image.
        imgbuf = txn.get(img_key)
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        try:
            if self.color_type == 'gray':
                img = Image.open(buf).convert('L')
            else:
                img = Image.open(buf).convert('RGB')
            img = np.asarray(img)
        except IOError:
            raise IOError('Corrupted image for {}'.format())

        results['filename'] = lmdb_index
        results['ori_filename'] = lmdb_index
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results

    def __repr__(self):
        return '{} (color_type={})'.format(self.__class__.__name__, self.color_type)

    def __del__(self):
        print('DEL!!')
        if self.env is not None:
            self.env.close()