import numpy as np
import xarray as xr
from ..calculation import FlexDimensionCalculation


class ArgMaxCalculation(FlexDimensionCalculation):

    @property
    def simple_type_return(self) -> bool:
        return len(self.dimensions) == 1
    
    def calculate_unit(self, input_array: xr.DataArray):
        """Given the input is an N-dimensional xarray that contains unnormalized
        distribution of N-dimensional variable, return a 1-dimensional xarray (dimension='coordinate')
        of length N with the most popular combination of coordinates. If N=1 return a single value with the most popular value.
        """
        # Check if input is 1-dimensional
        if self.simple_type_return:
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
            coord_values, coords={"coordinate": coord_names}, dims=["coordinate"]
        )
        return result
