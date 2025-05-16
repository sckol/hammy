from abc import ABC, abstractmethod
from .util import Experiment
from collections import OrderedDict
import dataclasses

class HammyObject(ABC):
  RESULTS_DIR = Path('results')

  def __init__(self, experiment: Experiment):
    self.experiment = experiment

  def dump():
    if not self.get_filename().parent.exists():
      self.get_filename().parent.mkdir(parents=True, exist_ok=True)
    self.dump_to_filename(self.get_filename())

  @abstractmethod
  def dump_to_filename(self, filename: str) -> None:
    pass

  def load(self) -> bool:
    if not self.get_filename().exists():
      return False
    self.load_from_filename(self.get_filename())
    print(f"Loaded cached object {self.get_id()} from file")  
    return True
  
  @abstractmethod  
  def load_from_filename(self, filename: str) -> None:
    pass

  @abstractmethod
  def calculate(self):
    pass

  @abstractmethod
  def get_id(self) -> str:
    pass

  @abstractmethod
  @staticmethod
  def get_file_extension() -> str:
    pass

  def get_filename(self) -> str:
    return self.RESULTS_DIR / str(self.experiment.to_folder_name()) / f"{self.get_id()}.{self.get_file_extension()}" 
    
class DictHammyObject(HammyObject):
  def __init__(self, experiment: Experiment):
    super().__init__(experiment) 
    self.data = OrderedDict()
    self.data.update({f"experiment_{k}": v for k, v in dataclasses.asdict(experiment).items()})

  def dump_to_filename(self, filename: str) -> None:
    with open(filename, 'w') as f:
      json.dump(self.data, f, indent=2)

  def load_from_filename(self, filename: str) -> None:    
    with open(filename, 'r') as f:
      self.data = json.load(f)
    loaded_experiment_dict = {k[len("experiment_"):]: v for k, v in self.data.items() if k.startswith("experiment_")}
    if loaded_experiment_dict != dataclasses.asdict(self.experiment):
      raise ValueError(f"Loaded experiment does not match the current experiment. Loaded: {loaded_experiment_dict}, Current: {dataclasses.asdict(self.experiment)}")

  @staticmethod
  def get_file_extension() -> str:
    return 'json'