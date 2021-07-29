import json
import os
from mmocr.datasets.builder import PARSERS
import numpy as np

@PARSERS.register_module()
class LineStrParser:
    """Parse string of one line in annotation file to dict format.

    Args:
        keys (list[str]): Keys in result dict.
        keys_idx (list[int]): Value index in sub-string list
            for each key above.
        separator (str): Separator to separate string to list of sub-string.
    """

    def __init__(self,
                 keys=['filename', 'text'],
                 keys_idx=[0, 1],
                 separator=' '):
        assert isinstance(keys, list)
        assert isinstance(keys_idx, list)
        assert isinstance(separator, str)
        assert len(keys) > 0
        assert len(keys) == len(keys_idx)
        self.keys = keys
        self.keys_idx = keys_idx
        self.separator = separator

    def get_item(self, data_ret, index):
        map_index = index % len(data_ret)
        line_str = data_ret[map_index]
        for split_key in self.separator:
            if split_key != ' ':
                line_str = line_str.replace(split_key, ' ')
        line_str = line_str.split()
        if len(line_str) <= max(self.keys_idx):
            raise Exception(
                f'key index: {max(self.keys_idx)} out of range: {line_str}')

        line_info = {}
        for i, key in enumerate(self.keys):
            line_info[key] = line_str[self.keys_idx[i]]
        return line_info


@PARSERS.register_module()
class TableTextLineStrParser:
    """Parse string of one line in annotation file to dict format.

    Args:
        keys (list[str]): Keys in result dict.
        keys_idx (list[int]): Value index in sub-string list
            for each key above.
        separator (str): Separator to separate string to list of sub-string.
    """

    def __init__(self,
                 keys=['filename', 'text'],
                 keys_idx=[0, 1],
                 separator=' '):
        assert isinstance(keys, list)
        assert isinstance(keys_idx, list)
        assert isinstance(separator, str)
        assert len(keys) > 0
        assert len(keys) == len(keys_idx)
        self.keys = keys
        self.keys_idx = keys_idx
        self.separator = separator

    def get_item(self, data_ret, index):
        map_index = index % len(data_ret)
        line_str = data_ret[map_index]
        line_str_part = []
        line_str = line_str.split(self.separator)
        line_str_part.append(line_str[0])  # file_path
        # line_str_part.append(''.join(line_str[1:]))  # merge text_list
        # remove the space char at begin of text by strip.
        line_str_part.append(''.join(line_str[1:]).strip())

        if len(line_str_part) <= max(self.keys_idx):
            raise Exception(
                f'key index: {max(self.keys_idx)} out of range: {line_str}')

        line_info = {}
        for i, key in enumerate(self.keys):
            line_info[key] = line_str_part[self.keys_idx[i]]
        return line_info


@PARSERS.register_module()
class LineJsonParser:
    """Parse json-string of one line in annotation file to dict format.

    Args:
        keys (list[str]): Keys in both json-string and result dict.
    """

    def __init__(self, keys=[], **kwargs):
        assert isinstance(keys, list)
        assert len(keys) > 0
        self.keys = keys

    def get_item(self, data_ret, index):
        map_index = index % len(data_ret)
        line_json_obj = json.loads(data_ret[map_index])
        line_info = {}
        for key in self.keys:
            if key not in line_json_obj:
                raise Exception(f'key {key} not in line json {line_json_obj}')
            line_info[key] = line_json_obj[key]

        return line_info


# some functions for table structure label parse.
def build_empty_bbox_mask(bboxes):
    """
    Generate a mask, 0 means empty bbox, 1 means non-empty bbox.
    :param bboxes: list[list] bboxes list
    :return: flag matrix.
    """
    flag = [1 for _ in range(len(bboxes))]
    for i, bbox in enumerate(bboxes):
        # empty bbox coord in label files
        if bbox == [0,0,0,0]:
            flag[i] = 0
    return flag

def get_bbox_nums_by_text(text):
    text = text.split(',')
    pattern = ['<td></td>', '<td', '<eb></eb>',
               '<eb1></eb1>', '<eb2></eb2>', '<eb3></eb3>',
               '<eb4></eb4>', '<eb5></eb5>', '<eb6></eb6>',
               '<eb7></eb7>', '<eb8></eb8>', '<eb9></eb9>',
               '<eb10></eb10>']
    count = 0
    for t in text:
        if t in pattern:
            count += 1
    return count

def align_bbox_mask(bboxes, empty_bbox_mask, label):
    """
    This function is used to in insert [0,0,0,0] in the location, which corresponding
    structure label is non-bbox label(not <td> style structure token, eg. <thead>, <tr>)
    in raw label file. This function will not insert [0,0,0,0] in the empty bbox location,
    which is done in label-preprocess.

    :param bboxes: list[list] bboxes list
    :param empty_bboxes_mask: the empty bbox mask
    :param label: table structure label
    :return: aligned bbox structure label
    """
    pattern = ['<td></td>', '<td', '<eb></eb>',
               '<eb1></eb1>', '<eb2></eb2>', '<eb3></eb3>',
               '<eb4></eb4>', '<eb5></eb5>', '<eb6></eb6>',
               '<eb7></eb7>', '<eb8></eb8>', '<eb9></eb9>',
               '<eb10></eb10>']
    assert len(bboxes) == get_bbox_nums_by_text(label) == len(empty_bbox_mask)
    bbox_count = 0
    structure_token_nums = len(label.split(','))
    # init with [0,0,0,0], and change the real bbox to corresponding value
    aligned_bbox = [[0., 0., 0., 0.] for _ in range(structure_token_nums)]
    aligned_empty_bbox_mask = [1 for _ in range(structure_token_nums)]
    for idx, l in enumerate(label.split(',')):
        if l in pattern:
            aligned_bbox[idx] = bboxes[bbox_count]
            aligned_empty_bbox_mask[idx] = empty_bbox_mask[bbox_count]
            bbox_count += 1
    return aligned_bbox, aligned_empty_bbox_mask

def build_bbox_mask(label):
    #TODO : need to debug to keep <eb></eb> or not.
    structure_token_nums = len(label.split(','))
    pattern = ['<td></td>', '<td', '<eb></eb>']
    mask = [0 for _ in range(structure_token_nums)]
    for idx, l in enumerate(label.split(',')):
        if l in pattern:
           mask[idx] = 1
    return np.array(mask)

@PARSERS.register_module()
class TableStrParser:
    """Parse a dict which include 'file_path', 'bbox', 'label' to training dict format.
    The advance parse will do here.

    Args:
        keys (list[str]): Keys in result dict.
        keys_idx (list[int]): Value index in sub-string list
            for each key above.
        separator (str): Separator to separate string to list of sub-string.
    """

    def __init__(self,
                 keys=['filename', 'text'],
                 keys_idx=[0, 1],
                 separator=','):
        assert isinstance(keys, list)
        assert isinstance(keys_idx, list)
        assert isinstance(separator, str)
        assert len(keys) > 0
        assert len(keys) == len(keys_idx)
        self.keys = keys
        self.keys_idx = keys_idx
        self.separator = separator

    def get_item(self, data_ret, index):
        map_index = index % len(data_ret)
        line_dict = data_ret[map_index]
        file_name = os.path.basename(line_dict['file_path'])
        text = line_dict['label']
        bboxes = line_dict['bbox']

        # advance parse bbox
        empty_bbox_mask = build_empty_bbox_mask(bboxes)
        bboxes, empty_bbox_mask = align_bbox_mask(bboxes, empty_bbox_mask, text)
        bboxes = np.array(bboxes)
        empty_bbox_mask = np.array(empty_bbox_mask)

        bbox_masks = build_bbox_mask(text)
        bbox_masks = bbox_masks * empty_bbox_mask

        line_info = {}
        line_info['filename'] = file_name
        line_info['text'] = text
        line_info['bbox'] = bboxes
        line_info['bbox_masks'] = bbox_masks

        return line_info
