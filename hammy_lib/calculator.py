import abc
import xarray as xr
import numpy as np
from typing import List
import scipy
import itertools
from tqdm import tqdm

class Calculator(abc.ABC):
  def __init__(self, input: xr.DataArray, independent_dimensions: List[str] = [], simple_type_return=False):
      self.input = input
      self.independent_dimensions = independent_dimensions
      self.simple_type_return = simple_type_return

  @staticmethod
  def extend_simulation_results(simulation_results: xr.DataArray) -> xr.DataArray:
    # Check if already extended
    if "TOTAL" in simulation_results.platform:
        raise ValueError("simulation_results is already extended (contains 'TOTAL' platform)")
    # Verify required dimensions exist
    if "level" not in simulation_results.dims or "platform" not in simulation_results.dims:
        raise ValueError("simulation_results must have both 'level' and 'platform' dimensions")
    # Calculate cumulative sum along level dimension
    cumsum_results = simulation_results.cumsum(dim="level")
    # Calculate total across platforms and maintain all other dimensions
    platform_total = cumsum_results.sum(dim="platform")
    # Create a DataArray for the total with the correct platform coordinate
    platform_total = platform_total.expand_dims(platform=["TOTAL"])
    # Add total as a new platform coordinate
    extended_results = xr.concat([cumsum_results, platform_total], dim="platform")
    return extended_results
  
  @abc.abstractmethod
  def calculate_unit(self, input: xr.DataArray) -> xr.DataArray | int | float | bool:
      pass
        

  def calculate(self) -> xr.DataArray:
    # Check if simulation_results is extended (contains "TOTAL" on "platform" dimension)
    if "platform" not in self.input.dims or "TOTAL" not in self.input.coords["platform"].values:
      raise ValueError("Input must be extended with 'TOTAL' coordinate on 'platform' dimension")
    required_dims = ["level", "platform"] + self.independent_dimensions
    results = []
    # Create all combinations of the required dimensions
    all_coords = [self.input.coords[dim].values for dim in required_dims]
    all_combinations = list(itertools.product(*all_coords))
    
    # Iterate over all combinations with a progress bar
    for indices in tqdm(all_combinations, desc="Processing coordinates", unit="combination"):
      # Create a dictionary of coordinates for selection
      coords_dict = {dim: indices[i] for i, dim in enumerate(required_dims)}
      # Select the data for these coordinates
      selected_data = self.input.sel(coords_dict)
      # Apply calculate_unit method
      result = self.calculate_unit(selected_data)
      # Store the result with its coordinates
      if self.simple_type_return:
        # Create array shape based on number of dimensions (all singleton)
        shape = tuple(1 for _ in required_dims)
        # If result is a simple type, create a DataArray with the coordinates
        result_array = xr.DataArray(
            np.full(shape, result),  # Create array of correct shape filled with result
            coords={dim: [coords_dict[dim]] for dim in required_dims},
            dims=required_dims
        )
        results.append(result_array)
      else:
        # If result is already an xarray, ensure it has the right coordinates
        for dim in required_dims:
          if dim not in result.coords:
            result = result.assign_coords({dim: coords_dict[dim]})
        results.append(result)
    
    # Combine all results
    return xr.combine_by_coords(results)


  
class MeanCalculator(Calculator):
  def __init__(self, simulation_results: xr.DataArray, parallelizable_dimensions: List[str] = []):
    # simple_type_return must be true if input contains exactly one dimension
    # excep platform, level and parallilaable_dimensions. If it contains 0 such dimensions, throw error 
    super(simulation_results, parallelizable_dimensions, simple_type_return)

  def calculate_unit(input):
    """Given the input is an N-dimensional xarray with  that contains unnormalized
    distribution of N-dimensional variable return an 1-dimensional xarray (domension='coordinate')
    of length N with marginal mean by each dimension. If N=1 return a single value"""

class ArgMaxCalculator(Calculator):
  def __init__(self, simulation_results: xr.DataArray, independent_dimensions: List[str] = []):
    extra_dims = [dim for dim in simulation_results.dims 
           if dim not in ['platform', 'level'] + independent_dimensions]
    if len(extra_dims) == 0:
      raise ValueError("Input must have at least one dimension besides platform, level and independent_dimensions")
    simple_type_return = len(extra_dims) == 1
    super().__init__(simulation_results, independent_dimensions, simple_type_return)

  def calculate_unit(self, input_array: xr.DataArray):
    """Given the input is an N-dimensional xarray that contains unnormalized 
    distribution of N-dimensional variable, return a 1-dimensional xarray (dimension='coordinate') 
    of length N with the most popular combination of coordinates. If N=1 return a single value with the most popular value.
    """
    # Check if input is 1-dimensional
    if len(input_array.dims) == 1:
      # For 1D array, return the coordinate with the maximum value as a simple type
      max_idx = input_array.argmax().item()
      return input_array.coords[input_array.dims[0]].values[max_idx]
    
    # For N-dimensional array, find the coordinate combination with the maximum value
    flat_idx = input_array.argmax().item()
    
    # Convert flat index to multi-dimensional indices
    multi_idx = np.unravel_index(flat_idx, input_array.shape)
    
    # Get the coordinate values and names for each dimension
    coord_values = []
    coord_names = []
    
    for dim_idx, dim_name in enumerate(input_array.dims):
      coord_value = input_array.coords[dim_name].values[multi_idx[dim_idx]]
      coord_values.append(coord_value)
      coord_names.append(dim_name)
    
    # Create a 1D array with the most popular combination
    result = xr.DataArray(
      coord_values,
      coords={'coordinate': coord_names},
      dims=['coordinate']
    )
    return result



class RandomnessTestCalculator(Calculator):  
  def __init__(self, simulation_results: xr.DataArray, theoretical_values: xr.DataArray):
     super(simulation_results, simple_type_return=True)
     self.theoretical_values = theoretical_values

  def calculate_unit():
     # for each dimension except platform, level and parallelizable diensions calculate
     # number of samples and variance and then chi-square test value. Does multivalue 
     # chi square test ever exist?
     return super().calculate()
  
class SCSCalculator(Calculator):
  def __init__(self, simulation_results: xr.DataArray, parallelizable_dimensions: [], walk_matrix: scipy.sparse,
              dimensionality: int):
    # if dimensionality not >=1 throw an error
    super(simulation_results, parallelizable_dimensions, dimensionality==1)
    self.walk_matrix = walk_matrix
    self.dimensionality = dimensionality

  class SCS_CONSTANTS:
    pass
    # n = N.BINS_LEN
    # A = sparse.csc_matrix(np.vstack((np.ones((1, n)), -np.eye(n))))
    # b = np.zeros(n + 1)
    # b[0] = 1
    # cone = dict(z=1, l=n)

  def get_approximation(width_in_steps, data):
    M = N.BINS_LEN
    y = data / np.sum(y)
    inv_y = data.astype(np.float64)
    inv_y[inv_y == 0] = .5
    inv_y = 1 / inv_y
    W = sparse.dia_matrix(((1. / y).reshape(1, -1), np.array((0,))), shape=(M, M))
    X = get_walk_matrix(1 - width_in_steps / N.T)
    n = X.shape[1]
    P = X.T @ W @ X
    c = -(X.T @ W @ y + y.T @ W @ X) /2
    data = dict(P=P, A=SCS_CONSTANTS.A, b=SCS_CONSTANTS.b, c=c)
    solver = scs.SCS(data, SCS_CONSTANTS.cone, eps_abs=1e-9, eps_rel=1e-9, verbose=False)
    return solver.solve()

  def get_approximation_error(width_in_steps, data):
    approximation = get_approximation(width_in_steps, data)
    return approximation['info']['pobj'] - np.linalg.norm(approximation['x'], 2)


  def calculate_unit():
    optimization_results = optimize.minimize_scalar(get_approximation_error, method='bounded', bounds=(10, N.T), args=(data,))
    width_in_steps = optimization_results['x']
    approximation_results = get_approximation(width_in_steps, data)
    x = approximation_results['x']
    x[np.where(np.abs(x) < 1e-4)] = 0
    mmin = np.min(np.nonzero(x))
    mmax = np.max(np.nonzero(x))
    ssum = np.sum(x)
    x = x / ssum
    assert get_trajectory_stats()
    return TrajectoryStat(optimization_results['success'] \
                          and approximation_results['info']['status'] == 'solved' \
                          and mmax - mmin < 2 \
                          and np.abs(ssum - 1) < 1e-4,
                          mmin * x[mmin] + mmax * x[mmax] if mmax - mmin == 1 else mmax,
                          width_in_steps)



