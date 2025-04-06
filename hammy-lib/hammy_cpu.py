from cffi import FFI
from pathlib import Path
import numpy as np
from glob import glob
def run_single_simulation(xxx, loops, out, seed):
  ffi = FFI()
  lib = ffi.dlopen(str(list((Path.cwd() / 'build-cffi').glob("hammy_cpu_kernel_lib.*"))[0]))
  ffi.cdef("""void run_simulation(unsigned long long loops, const unsigned long long seed,  unsigned long long* out);""")
  res = out.values
  dd = ffi.cast(f"unsigned long long*", ffi.from_buffer(res))
  lib.run_simulation(loops, seed, dd)
  ffi.dlclose(lib)
  return res
