"""msgpack with NumPy array support — wire codec of the openpi policy protocol.

Adapted from openpi (Apache-2.0):
https://github.com/Physical-Intelligence/openpi/blob/main/packages/openpi-client/src/openpi_client/msgpack_numpy.py
which is itself adapted from https://github.com/lebedov/msgpack-numpy.

This copy exists so that servers for models whose venvs do not include the
`openpi-client` package (GR00T, OpenVLA) can still speak the exact same wire
format. Keep byte-compatible with the upstream file: same b"__ndarray__" /
b"__npgeneric__" envelope, same dtype/shape fields.
"""

import functools

import msgpack
import numpy as np


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
