"""Compare full shuffling and on-demand bucket sampling without loading audio.

Run with an editable Lhotse installation, for example::

    python tools/benchmark_dynamic_bucketing.py random --bucket-size 10000 --draws 48
    python tools/benchmark_dynamic_bucketing.py random --bucket-size 10000 --draws 10000
    python tools/benchmark_dynamic_bucketing.py sampler cuts.jsonl.gz --buffer-size 300000

The random benchmark includes the locked snapshot and index initialization.
The sampler benchmark reads the supplied manifest without recipe filtering or
augmentation. It runs each variant in a fresh process and reports initialization
separately. Use --batches 0 --warmup 0 --no-drop-last to measure an entire
manifest, including initialization and the depleted tail. Neither measurement
represents DataLoader or training throughput.
"""

import argparse
import gc
import hashlib
import itertools
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from queue import Queue

import torch

import lhotse
import lhotse.dataset.sampling.dynamic_bucketing as dynamic_bucketing


def full_shuffle(bucket, rng, out_indexes_used):
    """The implementation preceding on-demand sampling."""
    with bucket.mutex:
        items = list(bucket.queue)
    indexes = list(range(len(items)))
    rng.shuffle(indexes)
    for index in indexes:
        out_indexes_used.append(index)
        yield items[index]


def summarize(values):
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def random_benchmark(args):
    if args.draws > args.bucket_size:
        raise ValueError("draws cannot exceed bucket size")
    bucket = Queue()
    for index in range(args.bucket_size):
        bucket.put(index)
    variants = {
        "shuffle": full_shuffle,
        "on_demand": dynamic_bucketing.pick_at_random,
    }
    results = {name: [] for name in variants}
    for repeat in range(args.repeats):
        names = list(variants)
        if repeat % 2:
            names.reverse()
        for name in names:
            rng = random.Random(args.seed + repeat)
            gc.collect()
            started = time.perf_counter()
            for _ in range(args.iterations):
                indexes = []
                sampled = list(
                    itertools.islice(variants[name](bucket, rng, indexes), args.draws)
                )
            elapsed = time.perf_counter() - started
            assert sampled == indexes
            assert len(set(indexes)) == len(indexes) == args.draws
            results[name].append(1e6 * elapsed / args.iterations)
    return {
        "bucket_size": args.bucket_size,
        "draws": args.draws,
        "iterations_per_repeat": args.iterations,
        "microseconds_per_prefix": results,
        "summary_microseconds_per_prefix": {
            name: summarize(values) for name, values in results.items()
        },
    }


def sampler_once(args):
    original = dynamic_bucketing.pick_at_random
    pick = full_shuffle if args.variant == "shuffle" else original
    sampled_prefixes = []

    def measured_pick(bucket, rng, out_indexes_used):
        sampled_prefixes.append((bucket.qsize(), out_indexes_used))
        return pick(bucket, rng, out_indexes_used)

    dynamic_bucketing.pick_at_random = measured_pick
    try:
        started = time.perf_counter()
        sampler = dynamic_bucketing.DynamicBucketingSampler(
            lhotse.CutSet.from_file(args.manifest),
            max_duration=args.max_duration,
            num_buckets=args.num_buckets,
            buffer_size=args.buffer_size,
            shuffle=True,
            drop_last=args.drop_last,
            seed=args.seed,
            world_size=1,
            rank=0,
            concurrent=False,
        )
        constructor_seconds = time.perf_counter() - started
        iterator = iter(sampler)
        iterator_seconds = time.perf_counter() - started - constructor_seconds
        started = time.perf_counter()
        warmup_batches = 0
        first_batch_seconds = None
        for _ in range(args.warmup):
            batch_started = time.perf_counter()
            batch = next(iterator, None)
            if first_batch_seconds is None:
                first_batch_seconds = time.perf_counter() - batch_started
            if batch is None:
                break
            warmup_batches += 1
        warmup_seconds = time.perf_counter() - started
        sampled_prefixes.clear()
        durations = []
        cut_count = 0
        audio_seconds = 0.0
        batch_hash = hashlib.sha256()
        cut_id_checksum = 0
        exhausted = False
        exhaustion_seconds = 0.0
        measured_started = time.perf_counter()
        steps = itertools.count() if args.batches == 0 else range(args.batches)
        for _ in steps:
            started = time.perf_counter()
            batch = next(iterator, None)
            elapsed = time.perf_counter() - started
            if first_batch_seconds is None:
                first_batch_seconds = elapsed
            if batch is None:
                exhausted = True
                exhaustion_seconds = elapsed
                break
            durations.append(elapsed)
            cut_count += len(batch)
            audio_seconds += sum(cut.duration for cut in batch)
            ids = list(batch.ids)
            batch_hash.update(json.dumps(ids).encode())
            for cut_id in ids:
                cut_id_checksum += int.from_bytes(
                    hashlib.sha256(cut_id.encode()).digest()
                )
            cut_id_checksum %= 2**256
        measured_wall_seconds = time.perf_counter() - measured_started
        prefixes = [(size, len(indexes)) for size, indexes in sampled_prefixes]
        return {
            "variant": args.variant,
            "constructor_seconds": constructor_seconds,
            "first_batch_seconds": first_batch_seconds,
            "iterator_seconds": iterator_seconds,
            "warmup_seconds": warmup_seconds,
            "exhaustion_seconds": exhaustion_seconds,
            "measured_wall_seconds": measured_wall_seconds,
            "total_sampler_seconds": (
                constructor_seconds
                + iterator_seconds
                + warmup_seconds
                + sum(durations)
                + exhaustion_seconds
            ),
            "warmup_batches": warmup_batches,
            "batches": len(durations),
            "exhausted": exhausted,
            "cuts": cut_count,
            "audio_seconds": audio_seconds,
            "batch_ids_sha256": batch_hash.hexdigest(),
            "unordered_cut_ids_checksum": f"{cut_id_checksum:064x}",
            "sampler_seconds": sum(durations),
            "batch_seconds": durations,
            "milliseconds_per_batch": (
                1000 * sum(durations) / len(durations) if durations else None
            ),
            "bucket_sizes_and_draws": prefixes,
            "consumed_fraction": (
                summarize([draws / size for size, draws in prefixes])
                if prefixes
                else None
            ),
        }
    finally:
        dynamic_bucketing.pick_at_random = original


def sampler_benchmark(args):
    results = []
    for repeat in range(args.repeats):
        variants = ["shuffle", "on_demand"]
        if repeat % 2:
            variants.reverse()
        for variant in variants:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "sampler",
                str(args.manifest.resolve()),
                "--variant",
                variant,
            ]
            for name in (
                "buffer_size",
                "num_buckets",
                "max_duration",
                "batches",
                "warmup",
                "seed",
            ):
                command.extend(
                    ["--" + name.replace("_", "-"), str(getattr(args, name))]
                )
            command.append("--drop-last" if args.drop_last else "--no-drop-last")
            completed = subprocess.run(
                command, check=True, capture_output=True, text=True
            )
            result = json.loads(completed.stdout)
            result["repeat"] = repeat
            results.append(result)
    return {
        "manifest": str(args.manifest),
        "buffer_size": args.buffer_size,
        "num_buckets": args.num_buckets,
        "max_duration": args.max_duration,
        "drop_last": args.drop_last,
        "runs": results,
    }


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    direct = modes.add_parser(
        "random", help="Time snapshot and random prefix selection"
    )
    direct.add_argument("--bucket-size", type=positive_int, default=10000)
    direct.add_argument("--draws", type=positive_int, default=48)
    direct.add_argument("--iterations", type=positive_int, default=100)
    direct.add_argument("--repeats", type=positive_int, default=5)
    direct.add_argument("--seed", type=int, default=0)
    sampler = modes.add_parser("sampler", help="Time metadata-only complete sampling")
    sampler.add_argument("manifest", type=Path)
    sampler.add_argument("--buffer-size", type=positive_int, default=20000)
    sampler.add_argument("--num-buckets", type=positive_int, default=10)
    sampler.add_argument("--max-duration", type=float, default=200.0)
    sampler.add_argument(
        "--drop-last", action=argparse.BooleanOptionalAction, default=True
    )
    sampler.add_argument(
        "--batches",
        type=nonnegative_int,
        default=300,
        help="Measured batches; 0 consumes the whole input",
    )
    sampler.add_argument("--warmup", type=nonnegative_int, default=20)
    sampler.add_argument("--repeats", type=positive_int, default=3)
    sampler.add_argument("--seed", type=int, default=0)
    sampler.add_argument(
        "--variant", choices=["shuffle", "on_demand"], help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.mode == "sampler" and args.max_duration <= 0:
        parser.error("max-duration must be positive")
    if args.mode == "sampler" and args.variant:
        result = sampler_once(args)
    else:
        result = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "lhotse": lhotse.__version__,
            "torch": torch.__version__,
            "seed": args.seed,
            "measurement": args.mode,
            "results": (
                random_benchmark(args)
                if args.mode == "random"
                else sampler_benchmark(args)
            ),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
