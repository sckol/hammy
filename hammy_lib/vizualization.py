import json
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from typing import Callable
from .hammy_object import HammyObject
from .hammy_object import ArrayHammyObject
from collections import OrderedDict
from dataclasses import dataclass


def apply_filters(data: xr.DataArray, filters: dict) -> xr.DataArray:
    filtered = data.copy()
    for dim, values in filters.items():
        if dim in filtered.dims:
            if not isinstance(values, (list, tuple, np.ndarray)):
                values = [values]
            filtered = filtered.sel({dim: values})
    return filtered


@dataclass
class CellContext:
    """Context passed to cell renderers."""
    x_val: object
    y_val: object | None
    x_name: str
    y_name: str | None
    row: int
    col: int
    is_first: bool
    full_data: xr.DataArray   # filtered data (before x/y selection)
    raw_data: xr.DataArray    # completely unfiltered data
    filter: dict              # the filter that was applied


class Vizualization(HammyObject):
    def __init__(
        self,
        results_object: ArrayHammyObject,
        x: str,
        cell_renderer: Callable[[matplotlib.axes.Axes, xr.DataArray, CellContext], None],
        y: str | None = None,
        filter: dict | None = None,
        title: str | None = None,
        figsize_per_cell: tuple[float, float] = (4, 2.5),
        sharex: bool = True,
        sharey: bool = True,
        id: str | None = None,
    ):
        super().__init__(id=id)
        self.results_object = results_object
        self.x = x
        self.y = y
        self.filter = filter if filter is not None else {}
        self.cell_renderer = cell_renderer
        self.title = title
        self.figsize_per_cell = figsize_per_cell
        self.sharex = sharex
        self.sharey = sharey
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

    def calculate(self) -> None:
        results = self.results_object.results
        filtered_data = apply_filters(results, self.filter)
        x_values = filtered_data.coords[self.x].values
        y_values = filtered_data.coords[self.y].values if self.y is not None else [None]
        nrows = len(y_values)
        ncols = len(x_values)

        w, h = self.figsize_per_cell
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(w * ncols, h * nrows),
            squeeze=False,
            facecolor="white",
            sharex=self.sharex,
            sharey=self.sharey,
        )
        self._figure = fig
        self._axes = axes

        if self.title:
            fig.suptitle(self.title, fontsize=14, fontweight="bold", y=0.995)

        for i, y_val in enumerate(y_values):
            for j, x_val in enumerate(x_values):
                ax = axes[i, j]
                subplot_data = (
                    filtered_data.sel({self.x: x_val, self.y: y_val})
                    if self.y is not None
                    else filtered_data.sel({self.x: x_val})
                )
                ctx = CellContext(
                    x_val=x_val,
                    y_val=y_val,
                    x_name=self.x,
                    y_name=self.y,
                    row=i,
                    col=j,
                    is_first=(i == 0 and j == 0),
                    full_data=filtered_data,
                    raw_data=results,
                    filter=self.filter,
                )
                self.cell_renderer(ax, subplot_data, ctx)

                ax.grid(True, alpha=0.15, linestyle="-", linewidth=0.8)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(axis="x", labelsize=9)
                ax.tick_params(axis="y", labelsize=9)

        label_color = "#444444"
        for j, x_val in enumerate(x_values):
            axes[0, j].annotate(
                f"{self.x}: {x_val}",
                xy=(0.5, 1.09), xycoords="axes fraction",
                ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=label_color, annotation_clip=False,
            )
        if self.y is not None:
            for i, y_val in enumerate(y_values):
                axes[i, 0].annotate(
                    f"{self.y}: {y_val}",
                    xy=(-0.18, 0.5), xycoords="axes fraction",
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", color=label_color, rotation=90,
                    annotation_clip=False,
                )

        plt.subplots_adjust(
            left=0.05, right=0.97, top=0.92, bottom=0.08, wspace=0.18, hspace=0.22
        )

    def dump_to_filename(self, filename: str) -> None:
        self._figure.savefig(
            filename, dpi=300, bbox_inches="tight",
            metadata={"hammy": json.dumps(self.metadata, default=str)},
        )

    def load_from_filename(self, filename: str) -> None:
        img = mpimg.imread(filename)
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis("off")
        self._figure = fig
        self._axes = ax
        pil_img = Image.open(filename)
        pnginfo = pil_img.info
        self.metadata = json.loads(pnginfo["hammy"], object_pairs_hook=OrderedDict)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_blue_palette(n_colors: int) -> list:
    if n_colors == 1:
        return ["#1976d2"]
    blues = ["#90caf9", "#42a5f5", "#1976d2", "#1565c0", "#0d47a1"]
    if n_colors <= len(blues):
        return blues[-n_colors:]
    cmap = plt.get_cmap("Blues")
    return [cmap(0.4 + 0.5 * (i / (n_colors - 1))) for i in range(n_colors)]


def _to_gray(color):
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)
    brightness = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return (brightness, brightness, brightness)


def _plot_lines(data, axis, groupby, ax, allow_aggregation=True,
                is_comparison=False, show_legend=False):
    dims_to_sum = [
        dim for dim in data.dims if dim not in [axis, groupby] and dim is not None
    ]
    if allow_aggregation:
        plot_data = data.sum(dims_to_sum) if dims_to_sum else data
    else:
        for dim in dims_to_sum:
            if data.sizes[dim] != 1:
                raise ValueError(
                    f"Dimension '{dim}' has size {data.sizes[dim]}, "
                    f"but allow_aggregation is False."
                )
        plot_data = data

    linestyle = "--" if is_comparison else "-"
    marker = "o"
    markersize = 1.5 if is_comparison else 1.9

    if groupby is None or groupby not in plot_data.dims:
        x_vals = plot_data.coords[axis].values
        y_vals = plot_data.values.flatten()
        colors = _get_blue_palette(1)
        color = _to_gray(colors[0]) if is_comparison else colors[0]
        ax.plot(x_vals, y_vals, color=color, linewidth=1.5, linestyle=linestyle,
                alpha=0.95, marker=marker, markersize=markersize)
    else:
        groupby_values = plot_data.coords[groupby].values
        colors = _get_blue_palette(len(groupby_values))
        for i, gv in enumerate(groupby_values):
            line_data = plot_data.sel({groupby: gv})
            x_vals = line_data.coords[axis].values
            y_vals = line_data.values.flatten()
            label = f"{groupby}={gv}" if show_legend else None
            color = _to_gray(colors[i]) if is_comparison else colors[i]
            ax.plot(x_vals, y_vals, color=color, linewidth=1.5, linestyle=linestyle,
                    alpha=0.95, label=label, marker=marker, markersize=markersize)
        if show_legend and len(groupby_values) > 1:
            ax.legend(frameon=False, fontsize=9, loc="upper right")


def _plot_reference(data, reference_func, axis, ax, is_comparison=False):
    ref_result = reference_func(data)
    y_vals = ref_result.values if hasattr(ref_result, "values") else np.array(ref_result)
    x_vals = (
        ref_result.coords[axis].values
        if hasattr(ref_result, "coords") and axis in ref_result.coords
        else data.coords[axis].values
    )
    linestyle = "--" if is_comparison else "-"
    ax.plot(np.ravel(x_vals), np.ravel(y_vals),
            color="black", linewidth=0.8, linestyle=linestyle, alpha=0.9)


# ---------------------------------------------------------------------------
# Built-in renderer factories
# ---------------------------------------------------------------------------

def line_renderer(
    axis: str,
    groupby: str | None = None,
    comparison: dict | None = None,
    reference: Callable | None = None,
    allow_aggregation: bool = True,
) -> Callable:
    """Factory: returns a cell renderer that draws grouped line plots.

    Reproduces the original Vizualization behavior: main lines, optional gray
    comparison overlay, optional black reference curve.
    """
    def render(ax, data, ctx: CellContext):
        _plot_lines(data, axis, groupby, ax,
                    allow_aggregation=allow_aggregation,
                    show_legend=ctx.is_first)

        comp_cell = None
        if comparison is not None:
            comp_filters = {**ctx.filter, **comparison}
            comp_data = apply_filters(ctx.raw_data, comp_filters)
            comp_cell = (
                comp_data.sel({ctx.x_name: ctx.x_val, ctx.y_name: ctx.y_val})
                if ctx.y_name
                else comp_data.sel({ctx.x_name: ctx.x_val})
            )
            _plot_lines(comp_cell, axis, groupby, ax,
                        allow_aggregation=allow_aggregation, is_comparison=True)

        if reference is not None:
            _plot_reference(data, reference, axis, ax)
            if comp_cell is not None:
                _plot_reference(comp_cell, reference, axis, ax, is_comparison=True)

    return render


def line_with_errors_renderer(
    axis: str,
    error: float | Callable | None = None,
    reference: Callable | None = None,
) -> Callable:
    """Factory: returns a cell renderer that draws a position line with error band.

    Args:
        axis: dimension plotted on the x-axis (e.g. "checkpoint").
        error: fixed float for symmetric error band, or a callable
               (data: DataArray) -> array of per-point errors, or None (no band).
        reference: optional callable (data) -> reference y-values.
    """
    def render(ax, data, ctx: CellContext):
        dims_to_squeeze = [d for d in data.dims if d != axis]
        plot_data = data
        for dim in dims_to_squeeze:
            if data.sizes[dim] == 1:
                plot_data = plot_data.squeeze(dim, drop=True)
            else:
                raise ValueError(
                    f"Dimension '{dim}' has size {data.sizes[dim]}; "
                    f"filter it to a single value before plotting."
                )

        x_vals = plot_data.coords[axis].values
        y_vals = plot_data.values.flatten().astype(float)
        color = "#1976d2"

        ax.plot(x_vals, y_vals, color=color, linewidth=1.5,
                marker="o", markersize=1.9, alpha=0.95)

        if error is not None:
            if callable(error):
                err_vals = np.array(error(data)).flatten()
            else:
                err_vals = np.full_like(y_vals, float(error))
            ax.fill_between(x_vals, y_vals - err_vals, y_vals + err_vals,
                            color=color, alpha=0.15)

        if reference is not None:
            _plot_reference(plot_data, reference, axis, ax)

    return render


def bar_chart_renderer(
    axis: str,
    reference: Callable | None = None,
    allow_aggregation: bool = True,
) -> Callable:
    """Factory: returns a cell renderer that draws a bar chart.

    Args:
        axis: dimension plotted on the x-axis (e.g. "x" for bin positions).
        reference: optional callable (data) -> reference y-values.
        allow_aggregation: if True, sum over extra dimensions; if False, require singletons.
    """
    def render(ax, data, ctx: CellContext):
        dims_to_sum = [d for d in data.dims if d != axis]
        if allow_aggregation:
            plot_data = data.sum(dims_to_sum) if dims_to_sum else data
        else:
            for dim in dims_to_sum:
                if data.sizes[dim] != 1:
                    raise ValueError(
                        f"Dimension '{dim}' has size {data.sizes[dim]}, "
                        f"but allow_aggregation is False."
                    )
            plot_data = data.squeeze(dims_to_sum, drop=True) if dims_to_sum else data

        x_vals = plot_data.coords[axis].values
        y_vals = plot_data.values.flatten().astype(float)
        color = "#1976d2"

        width = np.min(np.diff(x_vals)) * 0.8 if len(x_vals) > 1 else 0.8
        ax.bar(x_vals, y_vals, width=width, color=color, alpha=0.75, edgecolor="none")

        if reference is not None:
            _plot_reference(plot_data, reference, axis, ax)

    return render
