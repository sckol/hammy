import argparse
from .common import run

parser = argparse.ArgumentParser(description="Multi-lattice experiment")
parser.add_argument("--level", type=int, default=4, help="Simulation level (0-N)")
parser.add_argument("--no-viz", action="store_true", help="Skip visualizations")
parser.add_argument("--no-upload", action="store_true", help="Skip S3 upload")
parser.add_argument("--no-calculations", action="store_true", help="Skip calculations and viz")
parser.add_argument("--dry-run", action="store_true", help="Fast calibration")
parser.add_argument("--lattice", type=str, default=None,
                    help="Run only this lattice type (square/triangular/hexagonal/brick)")
args = parser.parse_args()

lattice_types = [args.lattice] if args.lattice else None

run(
    lattice_types=lattice_types,
    level=args.level,
    dry_run=args.dry_run,
    no_calculations=args.no_calculations,
    no_viz=args.no_viz,
    no_upload=args.no_upload,
)
