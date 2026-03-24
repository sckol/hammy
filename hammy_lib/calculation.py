import itertools
import numpy as np
import xarray as xr
from tqdm import tqdm
from abc import ABC, abstractmethod
from .hammy_object import ArrayHammyObject
from .simulation import Simulation


class Calculation(ArrayHammyObject, ABC):
    """Base class for post-processing computations on simulation results.

    A Calculation takes an ArrayHammyObject (typically a Simulation) and applies
    a function to slices of its data. The key concept is the split between
    "independent dimensions" and "data dimensions":

    - independent_dimensions: iterated over (all coordinate combinations are
      enumerated). Always includes level and platform.
    - The remaining dimensions form the data slice passed to calculate_unit().

    For example, if input has dims (level, platform, target, x) and
    independent_dimensions = [level, platform, target], then calculate_unit()
    receives a 1D array along x for each (level, platform, target) combination.

    When the input is a Simulation, extend_simulation_results() is called first
    to add cumulative sums across levels and a TOTAL platform.

    Subclasses must define:
    - independent_dimensions: which dims to iterate over
    - simple_type_return: True if calculate_unit returns a scalar, False for DataArray
    - calculate_unit(): the actual computation on each slice
    """

    _not_checked_fields = ["main_input"]

    def __init__(
        self,
        main_input: ArrayHammyObject,
        id: str | None = None,
    ):
        super().__init__(id=id)
        self.main_input = main_input

    @property
    @abstractmethod
    def independent_dimensions(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def simple_type_return(self) -> bool:
        pass

    def extend_simulation_results(self) -> xr.DataArray:
        # Check if already extended
        if "TOTAL" in self.main_input.results.platform:
            return self.main_input.results
        # Verify required dimensions exist
        if (
            "level" not in self.main_input.results.dims
            or "platform" not in self.main_input.results.dims
        ):
            raise ValueError(
                "simulation_results must have both 'level' and 'platform' dimensions"
            )
        # Calculate cumulative sum along level dimension
        cumsum_results = self.main_input.results.cumsum(dim="level")
        # Calculate total across platforms and maintain all other dimensions
        platform_total = cumsum_results.sum(dim="platform")
        # Create a DataArray for the total with the correct platform coordinate
        platform_total = platform_total.expand_dims(platform=["TOTAL"])
        # Add total as a new platform coordinate
        extended_results = xr.concat([cumsum_results, platform_total], dim="platform")
        return extended_results

    @abstractmethod
    def calculate_unit(self, input_array: xr.DataArray, coords: dict) -> xr.DataArray | int | float | bool:
        """Calculate a single unit given a selected DataArray slice and the
        current coordinates dictionary.

        Args:
            input: xarray DataArray selected for the current coordinates
            coords: dict mapping dimension names to their current coordinate values

        Returns:
            xr.DataArray or a simple type (int/float/bool)
        """
        pass

    def calculate(self) -> None:
        if isinstance(self.main_input, Simulation):
            main_input_results = self.extend_simulation_results()
        else:
            main_input_results = self.main_input.results
        required_dims = ["level", "platform"] + [
            x
            for x in self.independent_dimensions
            if x not in ["level", "platform"]
        ]
        results = []
        # Create all combinations of the required dimensions
        all_coords = [main_input_results.coords[dim].values for dim in required_dims]
        all_combinations = list(itertools.product(*all_coords))

        # Iterate over all combinations with a progress bar
        for indices in tqdm(
            all_combinations, desc="Processing coordinates", unit="combination"
        ):
            # Create a dictionary of coordinates for selection
            coords_dict = {dim: indices[i] for i, dim in enumerate(required_dims)}
            # Select the data for these coordinates
            selected_data = main_input_results.sel(coords_dict)
            # Apply calculate_unit method (pass current coordinates too)
            result = self.calculate_unit(selected_data, coords_dict)
            # Store the result with its coordinates
            if self.simple_type_return:
                # Create array shape based on number of dimensions (all singleton)
                shape = tuple(1 for _ in required_dims)
                # If result is a simple type, create a DataArray with the coordinates
                result_array = xr.DataArray(
                    np.full(
                        shape, result
                    ),  # Create array of correct shape filled with result
                    coords={dim: [coords_dict[dim]] for dim in required_dims},
                    dims=required_dims,
                )
                results.append(result_array)
            else:
                # If result is already an xarray, ensure it has the right coordinates
                for dim in required_dims:
                    if dim not in result.coords:
                        result = result.expand_dims({dim: [coords_dict[dim]]})
                results.append(result)
        # Combine all results
        self._results = xr.combine_by_coords(results)

    def generate_id(self) -> str:
        return f"{self.main_input.id}_{self.simple_name.lower()}"

    @property
    def simple_name(self) -> str:
        class_name = self.__class__.__name__
        if class_name.endswith("Calculation"):
            class_name = class_name[: -len("Calculation")]
            return class_name        


class FlexDimensionCalculation(Calculation, ABC):
    """Calculation where independent_dimensions are derived automatically.

    Instead of listing what to iterate over, you list the "data dimensions" --
    the ones that calculate_unit() operates on. Everything else becomes
    independent and gets iterated.
    """

    def __init__(
        self,
        main_input: ArrayHammyObject,
        dimensions: list[str],
        id: str | None = None,
    ):
        super().__init__(main_input=main_input, id=id)
        self.dimensions = dimensions

    @property
    def independent_dimensions(self) -> list[str]:
        # Get all dimensions of the main input except the specified dimensions
        all_dims = set(self.main_input.results.dims) - set(self.dimensions)
        return [str(d) for d in all_dims]
