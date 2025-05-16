import xarray as xd
# # Some ideas about vizualize function. It must generate 2d array linechart,
# so we have to define x dimension and y dimension. Also specify by which dimension to plot a chart (line).
# The method first apply the filters. Filter dict's keys are dimensions and values are what should be left.
# If filter dict's value is array-like, it just filters, if it is a single value, it reduces the number of dimensions.
# After filtering we should have an xarray with only dimensions corresponding to x, y and line.
# Also comparison is possible. On the linechart it is depicted with gray line. In case of comparison, we 
# specify a dict that updates the filters dict to make a new datacraf for comparison. 
# Also user can set reference xarray with dimensions like in x, y and line,
# the reference line is depicted with black
# In graph label we specify value for x, then for y
# Example: vizualize(results, 'checkpoint', 'target', 'x', {'target': [100, 500, 800, 1000]})
def vizualize(calculation_results: xd.DataArray, x: str, y: str, line: str, filters: dict = {}, 
              comparison: dict = None, reference: xd.DataArray = None):
    pass
