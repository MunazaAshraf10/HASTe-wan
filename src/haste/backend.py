from importlib.util import find_spec

if find_spec('triton') is not None:
    from haste import kernels as cuda
else:
    cuda = None
