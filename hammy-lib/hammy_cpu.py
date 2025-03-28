from cffi import FFI
from pathlib import Path
import numpy as np
import os
from glob import glob
def run(loops, out, seed):
  ffi = FFI()
  lib = ffi.dlopen(str(list(Path.cwd().glob("hammy_cpu_kernel.*.so"))[0]))
  ffi.cdef("""void run(unsigned long long loops, unsigned long long* out, long long seed);""")
  res = out.values
  dd = ffi.cast(f"unsigned long long*", ffi.from_buffer(res))
  lib.run(loops, dd, seed)
  ffi.dlclose(lib)
return res
