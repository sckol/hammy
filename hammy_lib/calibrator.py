from .hammy_object import DictHammyObject
from .util import Experiment, SimulatorConstants
class BaseCalibrator(DictHammyObject):
  def __init__(self, experiment: Experiment, simulation_constants: SimulatorConstants):
    super().__init__(experiment)
    self.simulation_constants = simulation_constants
    self.data.update()