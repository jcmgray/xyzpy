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


def parse_int_or_float(string):
    if string is None:
        return None
    try:
        return int(string)
    except ValueError:
        return float(string)


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
        help="The number of threads per worker (OMP_NUM_THREADS etc.)",
    )
    parser.add_argument(
        "--num-workers",
        type=parse_num_workers,
        default=None,
        help="The number of worker processes to use.",
    )
    parser.add_argument(
        "--ray",
        action="store_true",
        help=(
            "Use a ray executor, either connecting to an existing cluster, "
            "or starting a new one with num_workers"
        ),
    )
    parser.add_argument(
        "--gpus-per-task",
        type=parse_int_or_float,
        default=None,
        help=(
            "The number of gpus to request per task, if using a ray executor. "
            "The overall GPUs available is set by CUDA_VISIBLE_DEVICES, which "
            "ray follows."
        ),
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

    grow_kwargs = {
        "num_workers": args.num_workers,
        "verbosity": args.verbosity,
    }

    if args.ray:
        if args.gpus_per_task is None:
            grow_kwargs["executor"] = xyzpy.RayExecutor(
                num_cpus=args.num_workers,
            )
        else:
            grow_kwargs["executor"] = xyzpy.RayGPUExecutor(
                num_cpus=args.num_workers,
                gpus_per_task=args.gpus_per_task,
            )

    crop = xyzpy.Crop(name=args.crop_name, parent_dir=args.parent_dir)

    if not crop.is_prepared():
        raise xyzpy.utils.XYZPYError(f"The crop {crop} has not been sown yet.")

    print("Growing:")
    print(crop)
    crop.grow_missing(**grow_kwargs)
    print("Done!")


if __name__ == "__main__":
    main()
