# Prevent OpenBLAS deadlock after multiprocessing.Pool fork.
import os as _os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
del _v

import os
import sys
import argparse
import glob
import ctypes.util
import xarray as xr
import numpy as np
from pathlib import Path
from hammy_lib.hammy_object import HammyObject
from hammy_lib.machine_configuration import MachineConfiguration
from hammy_lib.experiment import Experiment
from hammy_lib.ccode import CCode
from hammy_lib.experiment_configuration import ExperimentConfiguration
from hammy_lib.sequential_calibration import SequentialCalibration
from hammy_lib.parallel_calibration import ParallelCalibration
from hammy_lib.simulation import Simulation
from hammy_lib.graph import (
    LatticeGraph2D, TriangularGraph2D, HexagonalGraph2D, BrickGraph2D,
)
from hammy_lib.calculations.position import CellPositionCalculation
from hammy_lib.calculations.bootstrap_position import BootstrapCellPositionCalculation
from hammy_lib.vizualization import Vizualization, line_renderer, line_with_errors_renderer


def _try_enable_mkl():
    if "libmkl_rt" in os.environ.get("LD_PRELOAD", ""):
        return
    candidates = set()
    for prefix in {sys.prefix, os.path.join(os.path.dirname(sys.executable), "..")}:
        candidates.update(glob.glob(os.path.join(prefix, "lib", "libmkl_rt*")))
    for sp in sys.path:
        if "site-packages" in sp or "dist-packages" in sp:
            candidates.update(glob.glob(os.path.join(os.path.dirname(sp), "libmkl_rt*")))
            candidates.update(glob.glob(os.path.join(sp, "*", "libmkl_rt*")))
    found = ctypes.util.find_library("mkl_rt")
    if found and os.path.isabs(found):
        candidates.add(found)
    for path in candidates:
        if os.path.isfile(path):
            print(f"[MKL] Using {os.path.basename(path)} from {os.path.dirname(path)}", flush=True)
            os.environ["LD_PRELOAD"] = path
            os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:])


try:
    HammyObject.RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
except NameError:
    HammyObject.RESULTS_DIR = Path("results")


# Shared parameters across all lattice types
T = 1000
CHECKPOINTS = [100, 200, 300, 400, 500, 600, 700, 800, 900, T]
CHECKPOINTS_LEN = len(CHECKPOINTS)
TARGETS_X = [0, 1, 0, 1, 3]
TARGETS_Y = [0, 0, 1, 1, 2]
TARGETS_LEN = len(TARGETS_X)
X_BINS_TUPLE = (-7, 8)  # 15 bins per dimension
Y_BINS_TUPLE = (-7, 8)
X_BINS_LEN = X_BINS_TUPLE[1] - X_BINS_TUPLE[0]
Y_BINS_LEN = Y_BINS_TUPLE[1] - Y_BINS_TUPLE[0]
BINS_FLAT_LEN = X_BINS_LEN * Y_BINS_LEN
NUMPY_WIDTH = 100

C_DEFINITIONS = f"""
  #define T {T}
  int TARGETS_X[] = {{{",".join(str(x) for x in TARGETS_X)}}};
  int TARGETS_Y[] = {{{",".join(str(y) for y in TARGETS_Y)}}};
  #define TARGETS_LEN { TARGETS_LEN }
  int CHECKPOINTS[] = {{{",".join(str(x) for x in CHECKPOINTS)}}};
  #define CHECKPOINTS_LEN { CHECKPOINTS_LEN }
  #define X_BINS_MIN { X_BINS_TUPLE[0] }
  #define X_BINS_MAX { X_BINS_TUPLE[1] - 1 }
  #define X_BINS_LEN { X_BINS_LEN }
  #define Y_BINS_MIN { Y_BINS_TUPLE[0] }
  #define Y_BINS_MAX { Y_BINS_TUPLE[1] - 1 }
  #define Y_BINS_LEN { Y_BINS_LEN }
  #define BINS_FLAT_LEN { BINS_FLAT_LEN }
"""

# Map of graph type name → Graph class
GRAPH_CLASSES = {
    "square": LatticeGraph2D,
    "triangular": TriangularGraph2D,
    "hexagonal": HexagonalGraph2D,
    "brick": BrickGraph2D,
}


def create_empty_results():
    """Shared result shape for all lattice types."""
    dims = ["target", "checkpoint", "x", "y"]
    coords = {
        "target": list(range(TARGETS_LEN)),
        "checkpoint": CHECKPOINTS,
        "x": np.arange(*X_BINS_TUPLE),
        "y": np.arange(*Y_BINS_TUPLE),
    }
    return xr.DataArray(
        np.zeros(tuple(len(coords[i]) for i in dims), dtype=np.int64),
        coords=coords,
        dims=dims,
    )


def run_single_lattice(graph_type, level=4, dry_run=False, no_calculations=False,
                       no_viz=False, no_upload=True, seed=1748065639484):
    """Run simulation + position calculation for one lattice type."""
    from importlib import import_module

    # Dynamic import of the lattice experiment module
    mod = import_module(f".{graph_type}", package="experiments.03_lattice_types")
    ExperimentClass = mod.EXPERIMENT_CLASS

    experiment = ExperimentClass()
    experiment.dump()
    experiment.compile()

    conf = MachineConfiguration()
    conf.dump()

    ec = ExperimentConfiguration(experiment, conf, seed=seed)
    ec.dump()

    sc = SequentialCalibration(ec, dry_run=dry_run)
    sc.dump()

    calibration_tolerance = 100 if ec.cores == 1 else 25
    pc = ParallelCalibration(sc, calibration_tolerance=calibration_tolerance)
    pc.dump()

    simulation = Simulation(pc, simulation_level=level)
    simulation.dump()

    if no_calculations:
        return {"simulation": simulation, "graph_type": graph_type}

    graph = GRAPH_CLASSES[graph_type](X_BINS_LEN, Y_BINS_LEN)
    graph.dump()

    position = CellPositionCalculation(
        simulation, graph=graph, spatial_dims=("x", "y"),
    )
    position.dump()

    bootstrap = BootstrapCellPositionCalculation(
        simulation, graph=graph, spatial_dims=("x", "y"), n_bootstrap=20,
    )
    bootstrap.dump()

    return {
        "simulation": simulation,
        "graph_type": graph_type,
        "graph": graph,
        "position": position,
        "bootstrap": bootstrap,
    }


def run(lattice_types=None, level=4, dry_run=False, no_calculations=False,
        no_viz=False, no_upload=True):
    """Run all lattice types and produce comparison visualizations."""
    if lattice_types is None:
        lattice_types = ["square", "triangular", "hexagonal", "brick"]

    results = {}
    for lt in lattice_types:
        print(f"\n{'='*60}", flush=True)
        print(f"  LATTICE TYPE: {lt.upper()}", flush=True)
        print(f"{'='*60}", flush=True)
        results[lt] = run_single_lattice(
            lt, level=level, dry_run=dry_run,
            no_calculations=no_calculations, no_viz=no_viz, no_upload=no_upload,
        )

    if no_calculations or no_viz:
        return results

    # Position visualizations per lattice type
    for lt, res in results.items():
        if "position" not in res:
            continue
        position = res["position"]
        simulation = res["simulation"]
        last_level = int(simulation.results.level.values[-1])

        def make_row_ref(lt_name):
            def row_ref(data):
                ti = int(data.coords["target"].item())
                tx = TARGETS_X[ti]
                cps = data.coords["checkpoint"].values
                return tx * cps / T - X_BINS_TUPLE[0]
            return row_ref

        def make_col_ref(lt_name):
            def col_ref(data):
                ti = int(data.coords["target"].item())
                ty = TARGETS_Y[ti]
                cps = data.coords["checkpoint"].values
                return ty * cps / T - Y_BINS_TUPLE[0]
            return col_ref

        # Position row
        Vizualization(
            results_object=position, x="platform", y="target",
            cell_renderer=line_renderer(axis="checkpoint", reference=make_row_ref(lt)),
            filter={"level": last_level, "position_data": "position_row"},
            title=f"{lt}: Position row by checkpoint",
            id=f"viz_{lt}_position_row",
        ).dump()

        # Position col
        Vizualization(
            results_object=position, x="platform", y="target",
            cell_renderer=line_renderer(axis="checkpoint", reference=make_col_ref(lt)),
            filter={"level": last_level, "position_data": "position_col"},
            title=f"{lt}: Position col by checkpoint",
            id=f"viz_{lt}_position_col",
        ).dump()

        # Fit quality
        Vizualization(
            results_object=position, x="platform", y="target",
            cell_renderer=line_renderer(
                axis="checkpoint",
                reference=lambda data: np.ones(len(data.coords["checkpoint"].values)),
            ),
            filter={"level": last_level, "position_data": "fit_quality"},
            title=f"{lt}: Cell fit quality by checkpoint",
            id=f"viz_{lt}_fit_quality",
        ).dump()

        # Cell dim
        Vizualization(
            results_object=position, x="platform", y="target",
            cell_renderer=line_renderer(
                axis="checkpoint",
                reference=lambda data: np.full(len(data.coords["checkpoint"].values), 2.0),
            ),
            filter={"level": last_level, "position_data": "cell_dim"},
            title=f"{lt}: Cell dimension by checkpoint",
            id=f"viz_{lt}_cell_dim",
        ).dump()

    return results
