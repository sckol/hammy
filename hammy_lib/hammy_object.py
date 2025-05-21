import re
import json
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from collections import OrderedDict

class HammyObject(ABC):
  RESULTS_DIR = Path('results')

  @staticmethod
  def generate_digest(s: str) -> str:    
    return hex(abs(int(hashlib.sha256(s.encode()).hexdigest(), 16)))[2:].zfill(6)[:6]

  def __init__(self, digest: str | None=None):    
    self.resolved = False
    self.metadata = OrderedDict()    
    self.digest: str | None = digest

  def fill_metadata(self) -> None:    
    for attr_name, attr_value in self.get_all_variables().items():
      if attr_name in ['metadata', 'digest', 'resolved', 'RESULTS_DIR']:
        continue
      if isinstance(attr_value, HammyObject):
        for k, v in  attr_value.metadata.items():
          if k in self.metadata and self.metadata[k] != v:
            raise ValueError(f"Metadata conflict for {k}: {self.metadata[k]} != {v}")
          self.metadata[k] = v        
      else:                        
        self.metadata[attr_name] = self.value_to_clear_string(attr_value)
    new_digest = self.generate_digest(json.dumps({k: v for k, v in self.metadata.items() if k != 'digest'}, sort_keys=True))  
    if self.digest is None:
      self.digest = new_digest
    elif self.digest != new_digest:
      raise ValueError(f"Digest mismatch: {self.digest} != {new_digest}")    
    self.metadata['digest'] = self.digest
    self.metadata.move_to_end('digest', last=False)

  @staticmethod
  def value_to_clear_string(object: object) -> str:
    str_value = re.sub(r'[\r\n]+', ' ', str(object)).strip()    
    str_value = re.sub(r'\s+', ' ', str_value)
    if len(str_value) > 255:
      str_value = HammyObject.generate_digest(str_value)
    return str_value

  def get_all_variables(self) -> dict:
    result = {}  
    for cls in reversed(type(self).mro()):        
      for name, value in vars(cls).items():          
        if not(name.startswith('_') or callable(value)):
          result[name] = value
    instance_attrs = vars(self).items()    
    for name, value in instance_attrs:
      if not(name.startswith('_') or callable(value)):
        result[name] = value  
    return result

  def resolve(self, no_load=False) -> None:
    if self.resolved:
      return
    # Iterate over all attributes of the class and resolve all HammyObjects    
    for _, attr_value in vars(self).items():
      if isinstance(attr_value, HammyObject):        
          attr_value.resolve(no_load=no_load)        
    if no_load or not self.load():
      self.calculate()
    self.fill_metadata()
    self.resolved = True

  def get_filename(self) -> str:
    folder_name = self.get_experiment_string(error_if_not_found=False) or ""
    return self.RESULTS_DIR / folder_name / f"{self.get_id()}.{self.get_file_extension()}"
  
  def get_experiment_string(self, error_if_not_found: bool = True) -> str | None:    
    found = all(f"experiment_{attr}" in self.metadata for attr in ['name', 'version', 'number'])
    if not found and error_if_not_found:
      raise ValueError("Experiment metadata not found.")
    if not found:
      return None
    return f"{self.metadata['experiment_number']}_{self.metadata['experiment_name']}_{self.metadata['experiment_version']}"    

  def dump(self) -> None:
    self.resolve()
    if not self.get_filename().parent.exists():
      self.get_filename().parent.mkdir(parents=True, exist_ok=True)
    self.dump_to_filename(self.get_filename())

  def load(self) -> bool:
    if not self.get_filename().exists():
      return False
    self.load_from_filename(self.get_filename())
    print(f"Loaded cached object {self.get_id()} from file")  
    return True    

  @abstractmethod
  def dump_to_filename(self, filename: str) -> None:
    pass

  @abstractmethod  
  def load_from_filename(self, filename: str) -> None:
    pass

  @abstractmethod
  def calculate(self) -> None:
    pass

  @abstractmethod
  def get_id(self) -> str:
    pass

  @staticmethod
  @abstractmethod
  def get_file_extension() -> str:
    pass
    
class DictHammyObject(HammyObject):
  def dump_to_filename(self, filename: str) -> None:
    with open(filename, 'w') as f:
      json.dump(self.metadata, f, indent=2)

  def load_from_filename(self, filename: str) -> None:    
    with open(filename, 'r') as f:      
      self.metadata = json.load(f, object_pairs_hook=OrderedDict)

  @staticmethod
  def get_file_extension() -> str:
    return 'json'
  