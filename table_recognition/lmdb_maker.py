import os
import cv2
import glob
import lmdb
import time
import pickle
import random
import logging
import argparse
import numpy as np


def line_operation(line):
    """
    Do something of line.
    :param line:
    :return:
    """
    return line

def get_img(item):
    """
    Get img data from item.
    :param item:
    :return:
    """
    return item

def get_label(item):
    """
    Get label from item.
    :param item:
    :return:
    """
    return item

def encode_img_by_jpeg(img, quality=95):
    """
    Encode raw img by jpeg compress with quality.
    Do this function to compress lmdb file size.
    :param img:
    :param quality:
    :return:
    """
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ret, buf = cv2.imencode('.jpg', img, encode_params)
    assert ret, 'failed to encode image by jpeg'
    data_encode = np.array(buf)
    # str_encode = data_encode.tostring()
    str_encode = data_encode.tobytes()
    return str_encode


def structure_label_operation(txt):
    """
    Do something to a structure label txt file, read info and return.
    1) modify absolute img path to relative path.
    :param txt: structure label txt file path.
    :return:
    """
    f = open(txt, 'r')
    # modify path
    absolute_path = f.readline()
    tmp = absolute_path.split('/')[-2:]
    relative_path = '/'.join(tmp)

    # add relative path and return
    lines = f.read()
    new_lines = relative_path + lines
    f.close()

    return new_lines, relative_path.strip()


def master_label_operation(line, separator='\t'):
    """
    Do something to a master text-line recognition label txt file, read info and return.
    1) extract image path.
    2) get text-line image labels.
    :param line: one line of master recognition label txt file.
    :return:
    """
    # text-line image path
    image_path = line.strip().split(separator)[0]

    # extract text img's label (merged)
    text = ''.join(line.strip().split(separator)[1:])

    # remove space at begin, which will effect text-line results
    if text.startswith(' '):
        text = text[1:]

    return image_path, text


def parse_tablemaster_args():
    """
    Setting TableMASTER lmdb maker parameter.
    :return:
    """
    parser = argparse.ArgumentParser(description='Lmdb marker')
    parser.add_argument("--is-shuffle", action='store_true', help='shuffle or not.')
    parser.add_argument('--lmdb-root', type=str, default='/data_0/dataset/processed_data/lmdb/',
                        help='lmdb output path.')
    parser.add_argument('--split', type=str, default='train', help='train or val phase.')
    parser.add_argument('--prefix', type=str, default='StructureLabel_', help='train or val phase.')
    parser.add_argument('--map-size', type=int, default=1099511627776, help='map size of lmdb.')
    parser.add_argument('--txt-folder', type=str, default='/data_0/dataset/processed_data/StructureLabelAddEmptyBbox_train',
                        help='TableMASTER txt folder.')
    parser.add_argument('--img-root', type=str, default='/data_0/dataset/pubtabnet',
                        help='pubtabnet dataset imgs root.')
    args = parser.parse_args()
    return args


def parse_master_args():
    """
    Setting MASTER lmdb maker parameter.
    :return:
    """
    parser = argparse.ArgumentParser(description='Lmdb marker')
    parser.add_argument("--is-shuffle", action='store_true', help='shuffle or not.')
    parser.add_argument('--lmdb-root', type=str, default='/data_0/dataset/processed_data/lmdb/',
                        help='lmdb output path.')
    parser.add_argument('--split', type=str, default='train', help='train or val phase.')
    parser.add_argument('--prefix', type=str, default='MasterRecLabel_', help='train or val phase.')
    parser.add_argument('--map-size', type=int, default=1099511627776, help='map size of lmdb.')
    parser.add_argument('--txt-root', type=str, default='/data_0/dataset/processed_data_0927',
                        help='MASTER txts root.')
    parser.add_argument('--img-root', type=str, default='/data_0/dataset/processed_data_0927',
                        help='text-line cropped imgs root.')
    args = parser.parse_args()
    return args


class LmdbMaker:
    def __init__(self, args):
        self.args = args
        self.init_db()
        self.begin_txn()
        assert args.is_shuffle is False # to comfirm index in list is right

    def init_db(self):
        lmdb_path = os.path.join(args.lmdb_root, args.prefix+args.split)
        self.db = lmdb.open(lmdb_path, map_size=self.args.map_size, readonly=False)

    def begin_txn(self):
        # begin or reset txn.
        self.txn = self.db.begin(write=True)

    def read_list(self):
        raise NotImplementedError

    def dumps_data(self,obj):
        """
        Serialize an object.
        :return:
        """
        return pickle.dumps(obj)

    def creat_lmdb(self):
        raise NotImplementedError


class TableMASTER_LmdbMaker(LmdbMaker):
    def __init__(self, args):
        """
        This part use to convert table structure recognition dataset to lmdb files.
        :param args:
        """
        super(TableMASTER_LmdbMaker, self).__init__(args)
        self.args = args

    def read_list(self):
        """
        Read a txt file and return all lines.
        :return:
        """
        folder = os.path.join(args.txt_folder, '*.txt')
        txt_lst = glob.glob(folder)

        if args.is_shuffle:
            random.shuffle(txt_lst)

        for txt in txt_lst:
            try:
                # do something, extract info from a txt file and pack to item.
                item = structure_label_operation(txt)
            except Exception as e:
                print("Parsing txt file met error for %s, detail: %s" % (txt, e))
                continue
            yield item

    def creat_lmdb(self):
        cnt = 0
        write_cnt = 0
        pre_time = time.time()
        file_list = self.read_list()
        for i, item in enumerate(file_list):
            # get lines and img path from item.
            info_lines, relative_path = item
            img_name = os.path.basename(relative_path)

            # read img
            img_path = os.path.join(self.args.img_root, relative_path)
            img = cv2.imread(img_path)

            # compress to reduce lmdb file size
            img = encode_img_by_jpeg(img)

            # construct one data item in lmdb file
            if img is not None:
                data = (img_name, img, info_lines)
                self.txn.put(u'{}'.format(i).encode(), self.dumps_data(data))
                write_cnt += 1
            else:
                raise ValueError('{} read fail in construct lmdb file.'.format(img_name))

            # flash
            if cnt % 100 == 0:
                # 100 this value should be small to flash.
                cur_time = time.time()
                print('time:', cur_time - pre_time, ' count:', cnt)
                self.txn.commit()
                self.begin_txn()  # reset
            cnt += 1

        # finish iterating through dataset
        keys = [u'{}'.format(k).encode() for k in range(write_cnt)]
        # __keys__是给dataloader索引用的，并不一定对应原来lst文件的行号。
        self.txn.put(b'__keys__', self.dumps_data(keys))
        self.txn.put(b'__len__', self.dumps_data(write_cnt))
        self.txn.commit()

        print("Flushing database ...")
        self.db.close()
        print("Done.")


class MASTER_LmdbMaker(LmdbMaker):
    def __init__(self, args):
        """
        This part use to convert text-line recognition dataset to lmdb files.
        :param args:
        """
        super(MASTER_LmdbMaker, self).__init__(args)
        self.args = args

    def read_list(self):
        """
        Read a txt file and return all lines.
        :return:
        """
        folder = os.path.join(args.txt_root, 'recognition_{}_txt'.format(args.split), '*.txt')
        txt_lst = glob.glob(folder)

        if args.is_shuffle:
            random.shuffle(txt_lst)

        for txt in txt_lst:
            print('parsing txt file : {}'.format(txt))
            with open(txt, 'r', encoding='utf8') as f:
                for line in f.readlines():
                    try:
                        # do something, extract info from a txt file and pack to item.
                        item = master_label_operation(line)
                    except Exception as e:
                        print("Parsing txt file met error for %s, detail: %s" % (txt, e))
                        continue
                    yield item

    def creat_lmdb(self):
        cnt = 0
        write_cnt = 0
        pre_time = time.time()
        file_list = self.read_list()
        for i, item in enumerate(file_list):
            # get text_img path and text from item.
            img_path, text = item
            img_name = os.path.basename(img_path)

            # read img
            img = cv2.imread(img_path)

            # compress to reduce lmdb file size
            img = encode_img_by_jpeg(img)

            # construct one data item in lmdb file
            if img is not None:
                data = (img, text)
                self.txn.put(u'{}'.format(i).encode(), self.dumps_data(data))
                write_cnt += 1
            else:
                raise ValueError('{} read fail in construct lmdb file.'.format(img_name))

            # flash
            if cnt % 1000 == 0:
                # 100 this value should be small to flash.
                cur_time = time.time()
                print('time:', cur_time - pre_time, ' count:', cnt)
                self.txn.commit()
                self.begin_txn()  # reset
            cnt += 1

        # finish iterating through dataset
        keys = [u'{}'.format(k).encode() for k in range(write_cnt)]
        # __keys__是给dataloader索引用的，并不一定对应原来lst文件的行号。
        self.txn.put(b'__keys__', self.dumps_data(keys))
        self.txn.put(b'__len__', self.dumps_data(write_cnt))
        self.txn.commit()

        print("Flushing database ...")
        self.db.close()
        print("Done.")



if __name__ == '__main__':
    # # TableMASTER lmdb create
    # args = parse_tablemaster_args()
    # logging.info(args)
    # lmdb_maker = TableMASTER_LmdbMaker(args)
    # lmdb_maker.creat_lmdb()

    # MASTER lmdb create
    # args = parse_master_args()
    # logging.info(args)
    # lmdb_maker = MASTER_LmdbMaker(args)
    # lmdb_maker.creat_lmdb()

    # # TableMASTER lmdb test
    # lmdb_path = '/data_0/dataset/processed_data/lmdb/StructureLabel_train/'
    # coding = 'utf8'
    # env = lmdb.open(
    #     lmdb_path,
    #     max_readers=1,
    #     readonly=True,
    #     lock=False,
    #     readahead=False,
    #     meminit=False,
    # )
    # with env.begin(write=False) as txn:
    #     # get lmdb's length
    #     total_number = int(pickle.loads(txn.get(b"__len__")))
    #     print('The length of TableMASTER lmdb is {}'.format(total_number))
    #     # get images
    #     data = pickle.loads(txn.get(b'0'))
    #     # img_name, img, info_lines
    #     img_name = data[0]
    #     bytes = data[1]
    #     buf = np.frombuffer(bytes, dtype=np.uint8)
    #     img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    #     info_lines = data[2]
    #     import pdb;pdb.set_trace()

    # MASTER lmdb test
    lmdb_path = '/data_0/dataset/processed_data/lmdb/MasterRecLabel_train/'
    coding = 'utf8'
    env = lmdb.open(
        lmdb_path,
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    with env.begin(write=False) as txn:
        # get lmdb's length
        total_number = int(pickle.loads(txn.get(b"__len__")))
        print('The length of MASTER lmdb is {}'.format(total_number))
        # get first image to check
        data = pickle.loads(txn.get(b'0'))
        # img, label
        bytes = data[0]
        label = data[1]
        buf = np.frombuffer(bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        # get loop to check
        for i in range(total_number):
            data = pickle.loads(txn.get('{}'.format(i).encode()))
            # img, label
            bytes = data[0]
            label = data[1]
            if label.startswith(' '):
                buf = np.frombuffer(bytes, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                import pdb;pdb.set_trace()


