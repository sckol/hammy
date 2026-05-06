"""numpy/cupy backend switcher.

Call `set_backend(numpy)` (CPU) or `set_backend(cupy)` (GPU) once at worker startup.
All code using this module is backend-agnostic.
"""
import numpy as _np

_backend = _np


def set_backend(backend) -> None:
    global _backend
    _backend = backend


def get_backend():
    return _backend


def array(data, **kwargs):
    return _backend.array(data, **kwargs)


def zeros(shape, **kwargs):
    return _backend.zeros(shape, **kwargs)


def ones(shape, **kwargs):
    return _backend.ones(shape, **kwargs)


def matmul(A, B):
    return _backend.matmul(A, B)


def to_numpy(x):
    """Convert array to numpy regardless of backend."""
    if _backend is _np:
        return _np.asarray(x)
    return _backend.asnumpy(x)


def arange(*args, **kwargs):
    return _backend.arange(*args, **kwargs)


def sum(x, **kwargs):
    return _backend.sum(x, **kwargs)
