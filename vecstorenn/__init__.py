import warnings
import lmdb
from numpy import dtype, float32
from struct import Struct
import mmap
import numpy
import os
from os.path import join as pjoin, dirname
import logging


# TODO: FAISS
NEG_INF = float("-inf")
TYPE = float32
IND_HEADER_STRUCT = Struct("<I")  # offset
HEADER_STRUCT = Struct("<II")  # offset, length
DEFAULT_MAP_SIZE = 2 * 1024 ** 3

logger = logging.getLogger(__name__)


class VecStorage:
    """
    Vector storage utility class. Constructor takes a path, a vector width, a mode
    flag and a value size in bytes.
    Mode flags are:
        r: read
        w: write
        d: create a directory rather than using path as a prefix
        i: mapping addresses individual vectors rather than groups of vectors
    """

    def __init__(self, path, vec_width, mode="r", value_bytes=0, map_size=DEFAULT_MAP_SIZE):
        self.mode = mode
        self.readonly = "r" in self.mode
        self.grouped = "i" not in self.mode
        if not self.grouped:
            assert value_bytes == 0
        self.path = path
        if "d" in self.mode:
            index_path = pjoin(path, "idx")
            data_path = pjoin(path, "dat")
            if not self.readonly:
                os.makedirs(path, exist_ok=True)
        else:
            index_path = path + ".idx"
            data_path = path
            if not self.readonly:
                dir = dirname(path)
                if dir:
                    os.makedirs(dir, exist_ok=True)
        self.index = lmdb.open(index_path, readonly=self.readonly, subdir=False, map_size=map_size)
        self.data = open(data_path, ("r" if self.readonly else "w") + "b")
        self.vec_width = vec_width
        self.value_bytes = value_bytes
        dtype_bytes = dtype(TYPE).itemsize
        value_dtypes = (value_bytes + dtype_bytes - 1) // dtype_bytes
        self.value_bytes_padded = value_dtypes * dtype_bytes
        if self.readonly:
            self.data.seek(0, 2)
            size_bytes = self.data.tell()
            self.vec_bytes = self.vec_width * dtype_bytes
            self.stride_bytes = self.vec_bytes + self.value_bytes_padded
            self.stride_dtypes = self.vec_width + value_dtypes
            self.total_rows, rem = divmod(size_bytes, self.stride_bytes)
            assert rem == 0

            self.mmap = mmap.mmap(self.data.fileno(), length=0, access=mmap.ACCESS_READ)
            self.mmap_vec = numpy.ndarray.__new__(
                numpy.ndarray,
                shape=(self.total_rows, self.stride_dtypes),
                dtype=TYPE,
                buffer=self.mmap,
                offset=0,
                order="C",
            )
        else:
            assert "w" in self.mode
            self.total_rows = 0
        logger.info("Created {}".format(repr(self)))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.index.close()
        self.data.close()

    def __repr__(self):
        if self.readonly:
            rest = (
                f" stride_bytes={self.stride_bytes} stride_dtypes={self.stride_dtypes}"
            )
        else:
            rest = ""
        return f"<VecStorage path={self.path} mode={self.mode} rows={self.total_rows} vec_width={self.vec_width} value_bytes={self.value_bytes} value_bytes_padded={self.value_bytes_padded}{rest}>"

    def add_vec(self, key, vec):
        self.add_vec_bytes(key.encode("utf-8"), vec)

    def add_vec_bytes(self, key, vec):
        assert not self.grouped
        self._write_vector(vec)
        with self.index.begin(write=True) as txn:
            txn.put(
                key,
                IND_HEADER_STRUCT.pack(self.total_rows),
            )
        self.total_rows += 1

    def add_group(self, group_key, iter):
        assert self.grouped
        group_len = 0
        for payload in iter:
            self._write_vector(payload)
            group_len += 1
        if group_len:
            with self.index.begin(write=True) as txn:
                txn.put(
                    group_key.encode("utf-8"),
                    HEADER_STRUCT.pack(self.total_rows, group_len),
                )
            self.total_rows += group_len

    def _write_vector(self, payload):
        if self.value_bytes > 0:
            k, v = payload
            if k.dtype != TYPE:
                warnings.warn(
                    f"Cannot use zero-copy when adding vector to VecStorage due to wrong data type: {k.dtype!r}"
                )
                k = TYPE(k)
            assert k.shape == (self.vec_width,)
            self.data.write(memoryview(k))
            self.data.write(v)
            actual_bytes = len(v)
            padding = self.value_bytes_padded - actual_bytes
            assert padding >= 0
            self.data.write(bytes(padding))
        else:
            self.data.write(memoryview(payload))

    def get_all(self):
        return self.mmap_vec[:, : self.vec_width]

    def get_vec(self, key):
        assert not self.grouped
        with self.index.begin(write=False, buffers=True) as txn:
            result = txn.get(key.encode("utf-8"))
        if result is None:
            return None
        offset, = IND_HEADER_STRUCT.unpack(result)
        return self.mmap_vec[offset, : self.vec_width]

    def get_group(self, group_key):
        assert self.grouped
        logger.debug(f"get_group({group_key})")
        with self.index.begin(write=False, buffers=True) as txn:
            result = txn.get(group_key.encode("utf-8"))
        if result is None:
            if self.value_bytes > 0:
                return None, []
            else:
                return None
        offset, group_len = HEADER_STRUCT.unpack(result)
        logger.debug(f"offset, group_len: {offset}, {group_len}")

        mat = self.mmap_vec[offset : offset + group_len, : self.vec_width]
        if self.value_bytes > 0:
            byte_offset = offset * self.stride_bytes

            def val_it():
                cur_off = byte_offset + self.vec_bytes
                for i in range(group_len):
                    val_start = cur_off
                    yield self.mmap[val_start : val_start + self.value_bytes]
                    cur_off += self.stride_bytes

            return mat, val_it()
        else:
            return mat


def nearest_from_mats(mats, query_vec):
    if query_vec is None:
        return None
    best_sim = NEG_INF
    best_idx = None
    for mat_idx, mat in enumerate(mats):
        if mat is None:
            continue
        sim = numpy.amax(mat.dot(query_vec))
        if sim > best_sim:
            best_sim = sim
            best_idx = mat_idx
    return best_idx


def value_from_mat(mat, vals, query_vec):
    if (mat is None) or (query_vec is None):
        return None
    best_idx = numpy.argmax(mat.dot(query_vec))
    return vals[best_idx]
