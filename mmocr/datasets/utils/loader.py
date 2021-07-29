import os.path as osp

from mmocr.datasets.builder import LOADERS, build_parser
from mmocr.utils import list_from_file, list_from_folder_table


@LOADERS.register_module()
class Loader:
    """Load annotation from annotation file, and parse instance information to
    dict format with parser.

    Args:
        ann_file (str): Annotation file path.
        parser (dict): Dictionary to construct parser
            to parse original annotation infos.
        repeat (int): Repeated times of annotations.
    """

    def __init__(self, ann_file, parser, repeat=1, max_seq_len=40):
        assert isinstance(ann_file, str)
        assert isinstance(repeat, int)
        assert isinstance(parser, dict)
        assert repeat > 0
        assert osp.exists(ann_file), f'{ann_file} is not exist'

        self.max_seq_len = max_seq_len
        self.ori_data_infos = self._load(ann_file)
        self.parser = build_parser(parser)
        self.repeat = repeat

    def __len__(self):
        return len(self.ori_data_infos) * self.repeat

    def _load(self, ann_file):
        """Load annotation file."""
        raise NotImplementedError

    def __getitem__(self, index):
        """Retrieve anno info of one instance with dict format."""
        return self.parser.get_item(self.ori_data_infos, index)

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        if self._n < len(self):
            data = self[self._n]
            self._n += 1
            return data
        raise StopIteration


@LOADERS.register_module()
class HardDiskLoader(Loader):
    """Load annotation file from hard disk to RAM.

    Args:
        ann_file (str): Annotation file path.
    """

    def _load(self, ann_file):
        return list_from_file(ann_file)


@LOADERS.register_module()
class TableHardDiskLoader(Loader):
    """Load table structure recognition annotation file from hard disk to RAM.

    Args:
        ann_files_folder (str): Annotation file folder.
    """

    def _load(self, ann_files_folder):
        return list_from_folder_table(ann_files_folder, self.max_seq_len)


@LOADERS.register_module()
class LmdbLoader(Loader):
    """Load annotation file with lmdb storage backend."""

    def _load(self, ann_file):
        lmdb_anno_obj = LmdbAnnFileBackend(ann_file)

        return lmdb_anno_obj


class LmdbAnnFileBackend:
    """Lmdb storage backend for annotation file.

    Args:
        lmdb_path (str): Lmdb file path.
    """

    def __init__(self, lmdb_path, coding='utf8'):
        self.lmdb_path = lmdb_path
        self.coding = coding
        env = self._get_env()
        with env.begin(write=False) as txn:
            self.total_number = int(
                txn.get('total_number'.encode(self.coding)).decode(
                    self.coding))

    def __getitem__(self, index):
        """Retrieval one line from lmdb file by index."""
        # only attach env to self when __getitem__ is called
        # because env object cannot be pickle
        if not hasattr(self, 'env'):
            self.env = self._get_env()
        with self.env.begin(write=False) as txn:
            line = txn.get(str(index).encode(self.coding)).decode(self.coding)
        return line

    def __len__(self):
        return self.total_number

    def _get_env(self):
        import lmdb
        return lmdb.open(
            self.lmdb_path,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )


@LOADERS.register_module()
class MJSTLmdbLoader(Loader):
    """Load annotation file with lmdb storage backend with FastOCR lmdb style."""

    def _load(self, ann_file):
        lmdb_anno_obj = MJSTLmdbAnnFileBackend(ann_file)

        return lmdb_anno_obj


class MJSTLmdbAnnFileBackend:
    """Lmdb storage backend for annotation file FastOCR lmdb style.

    Args:
        lmdb_path (str): Lmdb file path.
    """

    def __init__(self, lmdb_path, coding='utf8'):
        self.lmdb_path = lmdb_path
        self.coding = coding
        env = self._get_env()
        with env.begin(write=False) as txn:
            self.total_number = int(txn.get(b"num-samples"))

    def __getitem__(self, index):
        """Retrieval one line from lmdb file by index."""
        # only attach env to self when __getitem__ is called
        # because env object cannot be pickle
        if not hasattr(self, 'env'):
            self.env = self._get_env()

        # MJST lmdb file is start as index 1, and offset 1 ...
        index = index + 1
        # return index as filename for lmdb image reading.
        label_key, filename = b'label-%09d' % index, r'%s' % index
        separator = ' '
        with self.env.begin(write=False) as txn:
            line = filename + separator + txn.get(label_key).decode()
        return line

    def __len__(self):
        return self.total_number

    def _get_env(self):
        import lmdb
        return lmdb.open(
            self.lmdb_path,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
