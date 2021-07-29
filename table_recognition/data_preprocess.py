"""
This script for pubtabnet data process.
"""

import os
import cv2
import time
import glob
import json_lines
from tqdm import tqdm
import multiprocessing

def read_in_chunks(file_path, chunk_size=1024*1024):
    fid = open(file_path, 'r')
    while True:
        chunk_data = fid.read(chunk_size)
        if not chunk_data:
            break
        yield chunk_data


class PubtabnetParser(object):
    def __init__(self, jsonl_path, is_toy=True, split='val', is_pse_preLabel=False, chunks_nums=16):
        self.split = split
        self.raw_img_root = '/data_0/yejiaquan/data/TableRecognization/pubtabnet/'
        self.save_root = '/data_0/yejiaquan/data/mmocr_pubtabnet_recognition_0726/'
        self.detection_txt_folder = self.save_root + 'TxtPreLabel_{}/'.format(split)
        self.recognition_folder = self.save_root + 'recognition_{}_img/'.format(split)
        self.recognition_txt_folder = self.save_root + 'recognition_{}_txt'.format(split)
        self.jsonl_path = jsonl_path
        self.structure_txt_folder = self.save_root + 'StructureLabelAddEmptyBbox_{}/'.format(split)
        self.is_toy = is_toy
        self.is_pse_preLabel = is_pse_preLabel
        self.dataset_size = 4096 if is_toy else float("inf")
        self.chunks_nums = chunks_nums

        # alphabet path
        self.alphabet_path_1 = self.save_root + 'structure_alphabet.txt'
        self.alphabet_path_2 = self.save_root + 'textline_recognition_alphabet.txt'

        # make save folder
        self.make_folders()

        # empty box token dict, encoding for the token which is showed in image is blank.
        self.empty_bbox_token_dict = {
            "[]": '<eb></eb>',
            "[' ']": '<eb1></eb1>',
            "['<b>', ' ', '</b>']": '<eb2></eb2>',
            "['\\u2028', '\\u2028']": '<eb3></eb3>',
            "['<sup>', ' ', '</sup>']": '<eb4></eb4>',
            "['<b>', '</b>']": '<eb5></eb5>',
            "['<i>', ' ', '</i>']": '<eb6></eb6>',
            "['<b>', '<i>', '</i>', '</b>']": '<eb7></eb7>',
            "['<b>', '<i>', ' ', '</i>', '</b>']": '<eb8></eb8>',
            "['<i>', '</i>']": '<eb9></eb9>',
            "['<b>', ' ', '\\u2028', ' ', '\\u2028', ' ', '</b>']": '<eb10></eb10>',
        }
        self.empty_bbox_token_reverse_dict = {v: k for k, v in self.empty_bbox_token_dict.items()}


    @property
    def data_generator(self):
        return json_lines.reader(open(self.jsonl_path, 'rb'))


    def make_folders(self):
        if not os.path.exists(self.structure_txt_folder):
            os.makedirs(self.structure_txt_folder)
        if not os.path.exists(self.recognition_folder):
            os.makedirs(self.recognition_folder)
        if not os.path.exists(self.detection_txt_folder):
            os.makedirs(self.detection_txt_folder)
        if not os.path.exists(self.recognition_txt_folder):
            os.makedirs(self.recognition_txt_folder)
        for i in range(self.chunks_nums):
            recog_img_folder = os.path.join(self.recognition_folder, str(i))
            if not os.path.exists(recog_img_folder):
                os.makedirs(recog_img_folder)


    def divide_img(self, filenames):
        """
        This function is used to divide all files to nums chunks.
        nums is equal to process nums.
        :param filenames:
        :param nums:
        :return:
        """
        counts = len(filenames)
        nums_per_chunk = counts // self.chunks_nums
        img_chunks = []
        for n in range(self.chunks_nums):
            if n == self.chunks_nums - 1:
                s = n * nums_per_chunk
                img_chunks.append(filenames[s:])
            else:
                s = n * nums_per_chunk
                e = (n + 1) * nums_per_chunk
                img_chunks.append(filenames[s:e])
        return img_chunks


    def get_filenames(self, split='train'):
        filenames = []
        count = 0
        print("get {} filenames, it will take a moment.".format(split))
        for item in tqdm(self.data_generator):
            """
                item's keys : ['filename', 'split', 'imgid', 'html']
                item['html']'s keys : ['cells', 'structure']
                item['html']['cell'] : list of dict
                    eg. [
                        {"tokens": ["<b>", "V", "a", "r", "i", "a", "b", "l", "e", "</b>"], "bbox": [1, 4, 27, 13]},
                        {"tokens": ["<b>", "H", "a", "z", "a", "r", "d", " ", "r", "a", "t", "i", "o", "</b>"], "bbox": [219, 4, 260, 13]},
                    ]
                item['html']['structure']'s ['tokens']
                    eg. "structure": {"tokens": ["<thead>", "<tr>", "<td>", "</td>", ... ,"</tbody>"}
            """
            if count < self.dataset_size:
                if item['split'] == split:
                    filenames.append(item['filename'])
                    count += 1
                else:
                    continue
            else:
                break
        return filenames, count

    def merge_token(self, token_list):
        """
        This function used to merge the common tokens of raw tokens, and reduce the max length.
        eg. merge '<td>' and '</td>' to '<td></td>' which are always appear together.
        :param token_list: [list]. the raw tokens from the json line file.
        :return: merged tokens.
        """
        pointer = 0
        merge_token_list = []
        # </tbody> is the last token str.
        while token_list[pointer] != '</tbody>':
            if token_list[pointer] == '<td>':
                tmp = token_list[pointer] + token_list[pointer+1]
                merge_token_list.append(tmp)
                pointer += 2
            else:
                merge_token_list.append(token_list[pointer])
                pointer += 1
        merge_token_list.append('</tbody>')
        return merge_token_list

    def insert_empty_bbox_token(self, token_list, cells):
        """
        This function used to insert the empty bbox token(from empty_bbox_token_dict) to token_list.
        check every '<td></td>' and '<td'(table cell token), if 'bbox' not in cell dict, is a empty bbox.
        :param token_list: [list]. merged tokens.
        :param cells: [list]. list of table cell dict, each dict include cell's content and coord.
        :return: tokens add empty bbox str.
        """
        bbox_idx = 0
        add_empty_bbox_token_list = []
        for token in token_list:
            if token == '<td></td>' or token == '<td':
                if 'bbox' not in cells[bbox_idx].keys():
                    content = str(cells[bbox_idx]['tokens'])
                    empty_bbox_token = self.empty_bbox_token_dict[content]
                    add_empty_bbox_token_list.append(empty_bbox_token)
                else:
                    add_empty_bbox_token_list.append(token)
                bbox_idx += 1
            else:
                add_empty_bbox_token_list.append(token)
        return add_empty_bbox_token_list

    def count_merge_token_nums(self, token_list):
        """
        This function used to get the number of cells by token_list
        :param token_list: token_list after encoded (merged and insert empty bbox token str).
        :return: cells nums.
        """
        count = 0
        for token in token_list:
            if token == '<td':
                count += 1
            elif token == '<td></td>':
                count += 1
            elif token in self.empty_bbox_token_reverse_dict.keys():
                count += 1
            else:
                pass
        return count

    def convert_coord(self, coord):
        x_min, y_min, x_max, y_max = coord[0], coord[1], coord[2], coord[3]
        return [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]

    def crop(self, img, coord):
        x_min, y_min, x_max, y_max = coord[0], coord[1], coord[2], coord[3]
        return img[y_min:y_max, x_min:x_max]

    def get_thead_item_idx(self, token_list):
        """
        This function will return the index (start from 0) of cell, which is belong to table head.
        :param token_list: [list]. the raw tokens from the json line file.
        :return: list of index.
        """
        count = 0
        while token_list[count] != '</thead>':
            count += 1
        thead_tokens = token_list[:count+1]
        cell_nums_in_thead = thead_tokens.count('</td>')
        return [i for i in range(cell_nums_in_thead)]

    def remove_Bb(self, content):
        """
        This function will remove the '<b>' and '</b>' of the content.
        :param content: [list]. text content of each cell.
        :return: text content without '<b>' and '</b>'.
        """
        if '<b>' in content:
            content.remove('<b>')
        if '</b>' in content:
            content.remove('</b>')
        return content

    def get_structure_alphabet(self):
        """
        This function will return the alphabet which is used to Table Structure MASTER training.
        :return:
        """
        start_time = time.time()
        print("get structure alphabet ...")
        alphabet = []
        with open(self.alphabet_path_1, 'w') as f:
            for item in tqdm(self.data_generator):
                # record structure token
                cells = item['html']['cells']
                token_list = item['html']['structure']['tokens']
                merged_token = self.merge_token(token_list)
                encoded_token = self.insert_empty_bbox_token(merged_token, cells)
                for et in encoded_token:
                    if et not in alphabet:
                        alphabet.append(et)
                        f.write(et + '\n')
        print("get structure alphabet cost time {} s.".format(time.time()-start_time))

    def get_recognition_alphabet(self):
        """
        This function will return the alphabet which is used to Table text-line recognition training.
        :return:
        """
        start_time = time.time()
        print("get text-line recognition alphabet ...")
        alphabet = []
        fid = open(self.alphabet_path_2, 'w')
        search_folder = os.path.join(self.recognition_txt_folder, '*.txt')
        files = glob.glob(search_folder)
        for file in tqdm(files):
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    texts = line.strip().split('\t')[1:]
                    for text in texts:
                        if text not in alphabet:
                            alphabet.append(text)
                            fid.write(text + '\n')
        print("get recognition alphabet cost time {} s.".format(time.time() - start_time))

    def parse_single_chunk(self, this_chunk, chunks_idx):
        """
        This function will parse single chunk's image info.
        It will get :
            1. a label file for Table Structure Master training.
            2. some cropped images and text in table image's cell, for Table OCR Master training.
            3. get cell-level coord, for pre-Labeling (Training text line detection model).
        :param this_chunk: a list of image file names, only process the file in it.
        :param chunks_idx: [int]. chunk's id, is used to name folder.
        :return:
        """

        # init text-line recognition fid
        text_line_fid = open(os.path.join(self.recognition_txt_folder, '{}.txt'.format(chunks_idx)), 'w')

        progress_bar = tqdm(self.data_generator)
        for item in tqdm(self.data_generator):

            # progress_bar.set_description("PID : {} [{} / {}]".format(chunks_idx, sample_idx, len(self.data_generator)))

            filename = item['filename']
            if filename not in this_chunk:
                continue

            # parse info for Table Structure Master.
            """
            Structure txt include 3 lines:
                1. table image's path
                2. encoded structure token
                3. cell coord from json line file 
            """
            txt_filename = filename.replace('.png', '.txt')
            txt_filepath = os.path.join(self.structure_txt_folder, txt_filename)
            structure_fid = open(txt_filepath, 'w')

            # record image path
            image_path = os.path.join(self.raw_img_root, self.split, filename)
            structure_fid.write(image_path + '\n')

            # record structure token
            cells = item['html']['cells']
            cell_nums = len(cells)
            token_list = item['html']['structure']['tokens']
            merged_token = self.merge_token(token_list)
            encoded_token = self.insert_empty_bbox_token(merged_token, cells)
            encoded_token_str = ','.join(encoded_token)
            structure_fid.write(encoded_token_str + '\n')

            # record bbox coord
            cell_count = self.count_merge_token_nums(encoded_token)
            assert cell_nums == cell_count
            for cell in cells:
                if 'bbox' not in cell.keys():
                    bbox_line = ','.join([str(0) for _ in range(4)]) + '\n'
                else:
                    bbox_line = ','.join([str(b) for b in cell['bbox']]) + '\n'
                structure_fid.write(bbox_line)

            # if need pse preLabel, this part will get cell coord txt files.
            if self.is_pse_preLabel:
                pre_label_filename = 'gt_' + filename.replace('.png', '.txt')
                pre_label_filepath = os.path.join(self.detection_txt_folder, pre_label_filename)
                pre_label_fid = open(pre_label_filepath, 'w')

            # parse info for Table Recognition and text line detection pre-label(optional)
            img = cv2.imread(image_path)
            thead_item_idxs = self.get_thead_item_idx(token_list)
            for idx, cell in enumerate(cells):
                if 'bbox' not in cell.keys():
                    continue

                if self.is_pse_preLabel:
                    # text line detection pre-label, write to txt.
                    coord_list = self.convert_coord(cell['bbox'])
                    coord_str = '\t'.join([str(c) for c in coord_list])
                    coord_line = coord_str + '\n'
                    pre_label_fid.write(coord_line)

                # extract master recognition information and return, not write here.
                recognition_filename = filename.split('.')[0] + '_' + str(idx) + '.png'
                recognition_folder = os.path.join(self.recognition_folder, str(chunks_idx))
                if not os.path.exists(recognition_folder): os.makedirs(recognition_folder)
                recognition_filepath = os.path.join(recognition_folder, recognition_filename)

                content = cell['tokens']
                # remove '<b>' and '</b>' in thead's content.
                content = self.remove_Bb(content) if idx in thead_item_idxs else content
                text = '\t'.join([t for t in content])
                single_line = recognition_filepath + '\t' + text + '\n'
                text_line_fid.write(single_line)

                # crop text image for master recognition training.
                cropped = self.crop(img, cell['bbox'])
                text_img_path = os.path.join(self.recognition_folder, str(chunks_idx), recognition_filename)
                cv2.imwrite(text_img_path, cropped)


    def parse_images(self, img_chunks):
        """
        single process to parse raw data.
        It will take day to finish 500777 train files parsing.
        :param img_chunks:
        :return:
        """
        for i, img_chunk in enumerate(img_chunks):
            self.parse_single_chunk(img_chunk, i)


    def parse_images_mp(self, img_chunks, nproc):
        """
        multiprocessing to parse raw data.
        It will take about 7 hours to finish 500777 train files parsing.
        One process to do one chunk parsing.
        :param img_chunks:
        :param nproc:
        :return:
        """
        p = multiprocessing.Pool(nproc)
        for i in range(nproc):
            this_chunk = img_chunks[i]
            p.apply_async(self.parse_single_chunk, (this_chunk,i,))
        p.close()
        p.join()


if __name__ == '__main__':
    """
    README
    
    For ICDAR 2021 pubtabnet tasks, our method will train three models: 
        1. text-line detection model(PSENet)
        2. text-line recognition model(MASTER)
        3. table structure recognition model(Table MASTER)
    
    This python script will pre-process pubtabnet raw files for above 3 models training.
    
    For the first time you run this file, we recommend set 'is_toy = True' in PubtabnetParser Class for debug.
        
    Ps: 
        1.To train a PSENet, we only need about 2500 table images and get a perfect text-line detection. So it is 
        unnecessary to parse all train files for training. You can set 'is_toy=True' and set dataset_size a suitable number.
        
        2.The output pse Pre-Label files is cell's coord, not text-line coord, so you should go a step further to refine 
        these bboxes to text-line bboxes by Labeling software(eg. Labelme).
        
    """

    # parse train
    nproc = 16
    jsonl_path = r'/data_0/yejiaquan/data/TableRecognization/pubtabnet/PubTabNet_2.0.0.jsonl'
    parser = PubtabnetParser(jsonl_path, is_toy=False, split='train', is_pse_preLabel=False, chunks_nums=nproc)

    # multiprocessing
    start_time = time.time()
    filenames, count = parser.get_filenames(split='train')
    img_chunks = parser.divide_img(filenames)
    parser.parse_images_mp(img_chunks, nproc)
    print("parse images cost {} seconds.".format(time.time()-start_time))

    # single process
    # start_time = time.time()
    # filenames, count = parser.get_filenames(split='train')
    # img_chunks = parser.divide_img(filenames)
    # parser.parse_images(img_chunks)
    # print("parse images cost {} seconds.".format(time.time()-start_time))

    # get structure recognition alphabet.
    parser.get_structure_alphabet()

    # get text line recognition alphabet.
    parser.get_recognition_alphabet()












