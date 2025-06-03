import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from typing import Dict, Callable, Optional
from .hammy_object import HammyObject
from .hammy_object import ArrayHammyObject
from collections import OrderedDict


class Vizualization(HammyObject):
    def __init__(
        self,
        results_object: ArrayHammyObject,
        x: str,
        axis: str,
        y: Optional[str] = None,
        filter: Dict = {},
        groupby: Optional[str] = None,
        comparison: Optional[Dict] = None,
        reference: Optional[Callable] = None,
        y_axis_label: Optional[str] = None,
        id: str = None,
    ):
        super().__init__(id=id)
        self.results_object = results_object
        self.x = x
        self.axis = axis
        self.y = y
        self.filter = filter
        self.groupby = groupby
        self.comparison = comparison
        self.reference = reference
        self.y_axis_label = y_axis_label
        self._figure = None
        self._axes = None

    @property
    def file_extension(self) -> str:
        return "png"

    @property
    def simple_name(self) -> str:
        return "Vizualization"

    def generate_id(self):
        return f"{self.results_object.id}_viz"

    def _apply_filters(self, data: xr.DataArray, filters: Dict) -> xr.DataArray:
        filtered = data.copy()
        for dim, values in filters.items():
            if dim in filtered.dims:
                if not isinstance(values, (list, tuple, np.ndarray)):
                    values = [values]
                filtered = filtered.sel({dim: values})
        return filtered

    def _get_blue_palette(self, n_colors: int) -> list:
        if n_colors == 1:
            return ["#1976d2"]
        blues = ["#90caf9", "#42a5f5", "#1976d2", "#1565c0", "#0d47a1"]
        if n_colors <= len(blues):
            return blues[-n_colors:]
        cmap = plt.get_cmap("Blues")
        return [cmap(0.4 + 0.5 * (i / (n_colors - 1))) for i in range(n_colors)]

    def _to_gray(self, color):
        import matplotlib.colors as mcolors

        rgb = mcolors.to_rgb(color)
        brightness = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return (brightness, brightness, brightness)

    def _plot_grouped_lines(
        self,
        data: xr.DataArray,
        axis: str,
        groupby: Optional[str],
        ax: plt.Axes,
        is_comparison: bool = False,
        show_legend: bool = False,
    ):
        dims_to_sum = [
            dim for dim in data.dims if dim not in [axis, groupby] and dim is not None
        ]
        plot_data = data.sum(dims_to_sum) if dims_to_sum else data
        linestyle = "--" if is_comparison else "-"
        alpha = 0.95
        marker = "o"
        markersize = 1.5 if is_comparison else 1.9
        if groupby is None or groupby not in plot_data.dims:
            x_vals = plot_data.coords[axis].values
            y_vals = plot_data.values.flatten()
            colors = self._get_blue_palette(1)
            plot_color = self._to_gray(colors[0]) if is_comparison else colors[0]
            ax.plot(
                x_vals,
                y_vals,
                color=plot_color,
                linewidth=1.5,
                linestyle=linestyle,
                alpha=alpha,
                marker=marker,
                markersize=markersize,
            )
        else:
            groupby_values = plot_data.coords[groupby].values
            colors = self._get_blue_palette(len(groupby_values))
            for i, group_val in enumerate(groupby_values):
                line_data = plot_data.sel({groupby: group_val})
                x_vals = line_data.coords[axis].values
                y_vals = line_data.values.flatten()
                label = f"{groupby}={group_val}" if show_legend else None
                plot_color = self._to_gray(colors[i]) if is_comparison else colors[i]
                ax.plot(
                    x_vals,
                    y_vals,
                    color=plot_color,
                    linewidth=1.5,
                    linestyle=linestyle,
                    alpha=alpha,
                    label=label,
                    marker=marker,
                    markersize=markersize,
                )
            if show_legend and len(groupby_values) > 1:
                ax.legend(frameon=False, fontsize=9, loc="upper right")

    def _plot_reference_line(
        self,
        data: xr.DataArray,
        reference_func: Callable,
        filters: Dict,
        x_val,
        y_val,
        x: str,
        y: Optional[str],
        axis: str,
        ax: plt.Axes,
        is_comparison: bool = False,
    ):
        filtered_data = self._apply_filters(data, filters)
        subplot_data = (
            filtered_data.sel({x: x_val, y: y_val})
            if y is not None
            else filtered_data.sel({x: x_val})
        )
        ref_result = reference_func(subplot_data)
        y_vals = (
            ref_result.values if hasattr(ref_result, "values") else np.array(ref_result)
        )
        x_vals = (
            ref_result.coords[axis].values
            if hasattr(ref_result, "coords") and axis in ref_result.coords
            else subplot_data.coords[axis].values
        )
        y_vals = np.ravel(y_vals)
        x_vals = np.ravel(x_vals)
        linestyle = "--" if is_comparison else "-"
        ax.plot(
            x_vals, y_vals, color="black", linewidth=0.8, linestyle=linestyle, alpha=0.9
        )

    def calculate(self) -> None:
        results = self.results_object.results
        filtered_data = self._apply_filters(results, self.filter)
        x_values = filtered_data.coords[self.x].values
        y_values = filtered_data.coords[self.y].values if self.y is not None else [None]
        nrows = len(y_values)
        ncols = len(x_values)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4 * ncols, 2.5 * nrows),
            squeeze=False,
            facecolor="white",
            sharex=True,
            sharey=True,
        )
        self._figure = fig
        self._axes = axes
        filter_str = (
            ", ".join(f"{k}={v}" for k, v in self.filter.items())
            if self.filter
            else "No filters"
        )
        if self.comparison:
            comp_str = ", ".join(f"{k}={v}" for k, v in self.comparison.items())
            arg_str = f"[{filter_str}] vs [{comp_str}]"
        else:
            arg_str = f"{filter_str}"
        if self.groupby:
            arg_str += f" | groupby: {self.groupby}"
        y_axis_label = self.y_axis_label or self.results_object.simple_name
        fig.suptitle(
            f"{y_axis_label} by {self.axis} | {arg_str}",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )
        for i, y_val in enumerate(y_values):
            for j, x_val in enumerate(x_values):
                ax = axes[i, j]
                show_legend = i == 0 and j == 0
                subplot_data = (
                    filtered_data.sel({self.x: x_val, self.y: y_val})
                    if self.y is not None
                    else filtered_data.sel({self.x: x_val})
                )
                self._plot_grouped_lines(
                    subplot_data,
                    self.axis,
                    self.groupby,
                    ax,
                    is_comparison=False,
                    show_legend=show_legend,
                )
                if self.comparison is not None:
                    comp_filters = {**self.filter, **self.comparison}
                    comp_data = (
                        self._apply_filters(results, comp_filters).sel(
                            {self.x: x_val, self.y: y_val}
                        )
                        if self.y is not None
                        else self._apply_filters(results, comp_filters).sel(
                            {self.x: x_val}
                        )
                    )
                    self._plot_grouped_lines(
                        comp_data,
                        self.axis,
                        self.groupby,
                        ax,
                        is_comparison=True,
                        show_legend=False,
                    )
                if self.reference is not None:
                    self._plot_reference_line(
                        filtered_data,
                        self.reference,
                        self.filter,
                        x_val,
                        y_val,
                        self.x,
                        self.y,
                        self.axis,
                        ax,
                        is_comparison=False,
                    )
                    if self.comparison is not None:
                        comp_filters = {**self.filter, **self.comparison}
                        self._plot_reference_line(
                            results,
                            self.reference,
                            comp_filters,
                            x_val,
                            y_val,
                            self.x,
                            self.y,
                            self.axis,
                            ax,
                            is_comparison=True,
                        )
                ax.grid(True, alpha=0.15, linestyle="-", linewidth=0.8)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_title("")
                ax.tick_params(axis="x", labelsize=9)
                ax.tick_params(axis="y", labelsize=9)
        label_color = "#444444"
        for j, x_val in enumerate(x_values):
            ax = axes[0, j]
            ax.annotate(
                f"{self.x}: {x_val}",
                xy=(0.5, 1.09),
                xycoords="axes fraction",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=label_color,
                annotation_clip=False,
            )
        if self.y is not None:
            for i, y_val in enumerate(y_values):
                ax = axes[i, 0]
                ax.annotate(
                    f"{self.y}: {y_val}",
                    xy=(-0.18, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=label_color,
                    rotation=90,
                    annotation_clip=False,
                )
        plt.subplots_adjust(
            left=0.05, right=0.97, top=0.92, bottom=0.08, wspace=0.18, hspace=0.22
        )

    def dump_to_filename(self, filename: str) -> None:
        self._figure.savefig(
            filename,
            dpi=300,
            bbox_inches="tight",
            metadata={"hammy": json.dumps(self.metadata)}
        )

    def load_from_filename(self, filename: str) -> None:
        # Load the PNG and extract metadata, set figure to loaded image
        img = mpimg.imread(filename)
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis("off")
        self._figure = fig
        self._axes = ax
        # Load metadata from PNG using Pillow
        pil_img = Image.open(filename)
        pnginfo = pil_img.info
        self.metadata = json.loads(pnginfo["hammy"], object_pairs_hook=OrderedDict)
        print(self.metadata)
