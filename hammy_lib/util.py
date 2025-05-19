import abc
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import xarray as xr
from typing import Dict
from time import time
from .machine_configuration import MachineConfiguration
from typing import Optional
from .hashes import to_int_hash, hash_to_digest

def generate_random_seed() -> int:
    return int(time() * 1000)


