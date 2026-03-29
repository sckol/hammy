import argparse
from .lattice import run

parser = argparse.ArgumentParser(description="Lattice experiment (2D)")
parser.add_argument("--level", type=int, default=4, help="Simulation level (0-N)")
parser.add_argument("--no-viz", action="store_true", help="Skip visualizations")
parser.add_argument("--no-upload", action="store_true", help="Skip S3 upload")
parser.add_argument("--no-calculations", action="store_true", help="Skip calculations and viz")
parser.add_argument("--dry-run", action="store_true", help="Fast calibration (10%% loops)")
args = parser.parse_args()

run(
    level=args.level,
    dry_run=args.dry_run,
    no_calculations=args.no_calculations,
    no_viz=args.no_viz,
    no_upload=args.no_upload,
)
