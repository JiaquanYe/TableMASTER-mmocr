import os
import glob
import mmcv


def list_to_file(filename, lines):
    """Write a list of strings to a text file.

    Args:
        filename (str): The output filename. It will be created/overwritten.
        lines (list(str)): Data to be written.
    """
    mmcv.mkdir_or_exist(os.path.dirname(filename))
    with open(filename, 'w', encoding='utf-8') as fw:
        for line in lines:
            fw.write(f'{line}\n')


def convert_bbox(bbox_str_list):
    bbox_list = []
    for bbox_str in bbox_str_list:
        bbox_list.append(int(bbox_str))
    return bbox_list


def list_from_file(filename, encoding='utf-8'):
    """Load a text file and parse the content as a list of strings. The
    trailing "\\r" and "\\n" of each line will be removed.

    Note:
        This will be replaced by mmcv's version after it supports encoding.

    Args:
        filename (str): Filename.
        encoding (str): Encoding used to open the file. Default utf-8.

    Returns:
        list[str]: A list of strings.
    """
    item_list = []
    with open(filename, 'r', encoding=encoding) as f:
        for line in f:
            item_list.append(line.rstrip('\n\r'))
    return item_list


def list_from_folder_table(folder, max_seq_len):
    """Load table structure label files and parse the content as a list of dict. The
    advance parse will do in parser object.

    Args:
        folder (str): label files folder.
        max_seq_len (int): max length of sequence.

    Returns:
        list[str]: A list of dict.
    """
    item_list = []
    bbox_split = ','
    folder = os.path.join(folder, '*.txt')
    files = glob.glob(folder)
    count = 0
    print("Loading table data ...")
    for file in files:
        item_dict = dict()
        with open(file, 'r') as f:
            # get file_path
            file_path = f.readline().strip()
            label = f.readline().strip()

            # filter the samples, which length is greater than max_seq_len-2.
            # max_seq_len-2 because of include the <SOS> and <EOS>.
            if len(label.split(',')) > max_seq_len-2:
                continue

            # get bbox's label
            lines = f.readlines()
            bboxes = [convert_bbox(line.strip().split(bbox_split)) for line in lines]
            item_dict['file_path'] = file_path
            item_dict['label'] = label
            item_dict['bbox'] = bboxes
        item_list.append(item_dict)

        # process display
        count += 1
        if count % 10000 == 0:
            print("Loading table data, process : {}".format(count))

    # display samples number of dataset
    print("Load {} samples from folder : {}".format(len(item_list), folder))

    return item_list



