import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from typing import Dict, Callable, Optional

def visualize(
    results: xr.DataArray,
    x: str,
    y: Optional[str] = None,
    axis: str = None,
    filter: Dict = {},
    groupby: Optional[str] = None,
    comparison: Optional[Dict] = None,
    reference: Optional[Callable] = None
):
    def _apply_filters(data: xr.DataArray, filters: Dict) -> xr.DataArray:
        filtered = data.copy()
        for dim, values in filters.items():
            if dim in filtered.dims:
                if not isinstance(values, (list, tuple, np.ndarray)):
                    values = [values]
                filtered = filtered.sel({dim: values})
        return filtered

    def _get_blue_palette(n_colors: int) -> list:
        if n_colors == 1:
            return ['#1976d2']
        blues = ['#90caf9', '#42a5f5', '#1976d2', '#1565c0', '#0d47a1']
        if n_colors <= len(blues):
            return blues[-n_colors:]
        from matplotlib.cm import Blues
        cmap = plt.get_cmap('Blues')
        return [cmap(0.4 + 0.5 * (i / (n_colors - 1))) for i in range(n_colors)]

    def _plot_grouped_lines(data: xr.DataArray, axis: str, groupby: Optional[str], 
                           ax: plt.Axes, is_comparison: bool = False, show_legend: bool = False):
        dims_to_sum = [dim for dim in data.dims if dim not in [axis, groupby] and dim is not None]
        plot_data = data.sum(dims_to_sum) if dims_to_sum else data
        linestyle = '--' if is_comparison else '-'
        alpha = 0.7 if is_comparison else 0.95
        if groupby is None or groupby not in plot_data.dims:
            x_vals = plot_data.coords[axis].values
            y_vals = plot_data.values.flatten()
            colors = _get_blue_palette(1)
            ax.plot(x_vals, y_vals, color=colors[0], linewidth=2.5, linestyle=linestyle, alpha=alpha)
        else:
            groupby_values = plot_data.coords[groupby].values
            colors = _get_blue_palette(len(groupby_values))
            for i, group_val in enumerate(groupby_values):
                line_data = plot_data.sel({groupby: group_val})
                x_vals = line_data.coords[axis].values
                y_vals = line_data.values.flatten()
                label = f'{groupby}={group_val}' if show_legend else None
                ax.plot(x_vals, y_vals, color=colors[i], linewidth=2.5, linestyle=linestyle, alpha=alpha, label=label)
            if show_legend and len(groupby_values) > 1:
                ax.legend(frameon=False, fontsize=9, loc='upper right')

    def _plot_reference_line(data: xr.DataArray, reference_func: Callable,
                            filters: Dict, x_val, y_val, x: str, y: Optional[str], axis: str, 
                            ax: plt.Axes, is_comparison: bool = False):
        filtered_data = _apply_filters(data, filters)
        subplot_data = filtered_data.sel({x: x_val, y: y_val}) if y is not None else filtered_data.sel({x: x_val})
        ref_result = reference_func(subplot_data)
        y_vals = ref_result.values if hasattr(ref_result, 'values') else np.array(ref_result)
        x_vals = ref_result.coords[axis].values if hasattr(ref_result, 'coords') and axis in ref_result.coords else subplot_data.coords[axis].values
        y_vals = np.ravel(y_vals)
        x_vals = np.ravel(x_vals)
        linestyle = '--' if is_comparison else '-'
        ax.plot(x_vals, y_vals, color='black', linewidth=1.2, linestyle=linestyle, alpha=0.8)

    filtered_data = _apply_filters(results, filter)
    x_values = filtered_data.coords[x].values
    y_values = filtered_data.coords[y].values if y is not None else [None]
    nrows = len(y_values)
    ncols = len(x_values)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 2.5*nrows),
                             squeeze=False, facecolor='white', sharex=True, sharey=True)

    # Main title
    arg_str = f"{x} vs {y if y else '(no y)'} | axis: {axis}"
    if groupby: arg_str += f" | groupby: {groupby}"
    if comparison: arg_str += f" | comparison: {comparison}"
    fig.suptitle(f"Line Chart Grid: {arg_str}", fontsize=14, fontweight='bold', y=0.995)

    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            ax = axes[i, j]
            show_legend = (i == 0 and j == 0)
            subplot_data = filtered_data.sel({x: x_val, y: y_val}) if y is not None else filtered_data.sel({x: x_val})

            _plot_grouped_lines(subplot_data, axis, groupby, ax, is_comparison=False, show_legend=show_legend)
            if comparison is not None:
                comp_filters = {**filter, **comparison}
                comp_data = _apply_filters(results, comp_filters).sel({x: x_val, y: y_val}) if y is not None else _apply_filters(results, comp_filters).sel({x: x_val})
                _plot_grouped_lines(comp_data, axis, groupby, ax, is_comparison=True, show_legend=False)
            if reference is not None:
                _plot_reference_line(filtered_data, reference, filter, x_val, y_val, x, y, axis, ax, is_comparison=False)
                if comparison is not None:
                    comp_filters = {**filter, **comparison}
                    _plot_reference_line(results, reference, comp_filters, x_val, y_val, x, y, axis, ax, is_comparison=True)

            # Grid and spines
            ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_title("")  # No subplot title
            
            ax.tick_params(axis='x', labelsize=9)
            ax.tick_params(axis='y', labelsize=9)


    # Column labels (top row) with dark gray color and smaller font
    label_color = '#444444'
    for j, x_val in enumerate(x_values):
        ax = axes[0, j]
        ax.annotate(f"{x}: {x_val}", xy=(0.5, 1.09), xycoords="axes fraction",
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=label_color, annotation_clip=False)

    # Row labels (first column, left, vertical) with dark gray color and smaller font
    if y is not None:
        for i, y_val in enumerate(y_values):
            ax = axes[i, 0]
            ax.annotate(f"{y}: {y_val}", xy=(-0.18, 0.5), xycoords="axes fraction",
                        ha='center', va='center', fontsize=9, fontweight='bold', color=label_color, rotation=90, annotation_clip=False)

    plt.subplots_adjust(left=0.13, right=0.97, top=0.92, bottom=0.08, wspace=0.18, hspace=0.22)
    plt.show()
    return fig, axes
