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


def parse_bool_flag(value: Optional[str] = None) -> bool:
    if value is None:
        return False
    if value.lower() in ("false", "0", "no"):
        return False
    elif value.lower() in ("true", "1", "yes"):
        return True
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value {value}.")


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
        "--batch-ids",
        type=str,
        default="missing",
        help=(
            "Comma separated list of which batches to grow, "
            "by default all missing results."
        ),
    )
    parser.add_argument(
        "--raise-errors",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool_flag,
        help="Raise batch errors.",
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
        "--subprocess",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool_flag,
        help=(
            "Run each batch in its own fresh subprocess. "
            "This is most robust in terms of memory, at the cost of the "
            "process startup overhead. Optional value: true/false."
        ),
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
        "--affinities",
        type=str,
        default=None,
        help=(
            "If subprocess is enabled, this is an optional comma separated "
            "list of affinities to use, one for each process. This ensures a "
            "single cpu core is used for each batch, regardless of other "
            "environment variables."
        ),
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        help="The verbosity level.",
    )
    parser.add_argument(
        "--verbosity-grow",
        type=int,
        default=1,
        help="The verbosity level.",
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
        "subprocess": args.subprocess,
        "raise_errors": args.raise_errors,
        "verbosity": args.verbosity,
        "verbosity_grow": args.verbosity_grow,
    }

    if args.subprocess:
        # this calls `xyzpy-grow` itself and we want to match env vars above
        grow_kwargs["num_threads"] = args.num_threads
        grow_kwargs["affinities"] = args.affinities

    if args.ray:
        if args.subprocess:
            raise xyzpy.utils.XYZError(
                "Cannot use subprocess mode with ray executor."
            )

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
        raise xyzpy.utils.XYZError(f"The crop {crop} has not been sown yet.")

    if args.batch_ids == "missing":
        batch_ids = crop.missing_results()
    else:
        batch_ids = tuple(
            map(int, filter(None, args.batch_ids.replace(" ", "").split(",")))
        )

    print("Growing:")
    print(crop)
    crop.grow(batch_ids=batch_ids, **grow_kwargs)
    print("Done!")


if __name__ == "__main__":
    main()
