import argparse
import os
import sys
from typing import Optional


def parse_num_workers(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid value for num_workers")


def main():
    parser = argparse.ArgumentParser(
        description="Grow crops using xyzpy-gen-cropping."
    )
    parser.add_argument(
        "crop_name", type=str, help="The name of the crop to grow."
    )
    parser.add_argument(
        "--parent-dir",
        type=str,
        default=".",
        help="The parent directory of the crop.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debugging mode."
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="The number of threads per worker.",
    )
    parser.add_argument(
        "--num-workers",
        type=parse_num_workers,
        default=None,
        help="The number of worker processes to use.",
    )
    parser.add_argument(
        "--verbosity", type=int, default=1, help="The verbosity level."
    )
    args = parser.parse_args()

    # common thread control environment variables
    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(args.num_threads)
    os.environ["NUMBA_NUM_THREADS"] = str(args.num_threads)

    sys.path.append(args.parent_dir)
    import xyzpy

    crop = xyzpy.Crop(name=args.crop_name, parent_dir=args.parent_dir)
    print("Growing:")
    print(crop)
    crop.grow_missing(
        num_workers=args.num_workers,
        verbosity=args.verbosity,
    )
    print("Done!")


if __name__ == "__main__":
    main()
