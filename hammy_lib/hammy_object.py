from abc import ABC, abstractmethod
import dataclasses
from pathlib import Path
import json


class HammyObject(ABC):
  RESULTS_DIR = Path('results')

  def __init__(self):    
    self.resolved = False    

  def resolve(self, no_load=False) -> None:
    if self.resolved:
      return
    # Iterate over all attributes of the class and resolve all HammyObjects    
    for _, attr_value in vars(self).items():
      if isinstance(attr_value, HammyObject):        
          attr_value.resolve(no_load=no_load)        
    if no_load or not self.load():
      self.calculate()
    self.resolved = True

  def dump(self) -> None:
    self.resolve()
    if not self.get_filename().parent.exists():
      self.get_filename().parent.mkdir(parents=True, exist_ok=True)
    self.dump_to_filename(self.get_filename())

  @abstractmethod
  def dump_to_filename(self, filename: str) -> None:
    pass

  @abstractmethod
  def validate_loaded_object(self) -> None:
    pass

  def load(self) -> bool:
    if not self.get_filename().exists():
      return False
    self.load_from_filename(self.get_filename())
    self.validate_loaded_object()
    print(f"Loaded cached object {self.get_id()} from file")  
    return True
  
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

  @abstractmethod
  def get_foldername(self) -> str:
    pass

  def get_filename(self) -> str:
    return self.RESULTS_DIR / str(self.get_foldername()) / f"{self.get_id()}.{self.get_file_extension()}" 
    
class DictHammyObject(HammyObject):
  def __init__(self):
    super().__init__() 
    self.data = {}    

  def dump_to_filename(self, filename: str) -> None:
    with open(filename, 'w') as f:
      json.dump(self.data, f, indent=2)

  def load_from_filename(self, filename: str) -> None:    
    with open(filename, 'r') as f:
      self.data = json.load(f)

  def validate_loaded_object(self) -> None:    
    loaded_experiment_dict = {k[len("experiment_"):]: v for k, v in self.data.items() if k.startswith("experiment_")}
    if loaded_experiment_dict != dataclasses.asdict(self.experiment):
      raise ValueError(f"Loaded experiment does not match the current experiment. Loaded: {loaded_experiment_dict}, Current: {dataclasses.asdict(self.experiment)}")

  @staticmethod
  def get_file_extension() -> str:
    return 'json'
  