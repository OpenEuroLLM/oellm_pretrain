#!/usr/bin/env python3

"""
Utility to launch Megatron-LM training sweeps via SLURM job arrays.

Use case:
- Parse a YAML config with sbatch parameters, megatron args, and sweep keys.
- Expand sweeps into individual configurations, generate unique job names,
  and save them as sweep.json for the sbatch template to consume.
- Write and submit a ready-to-run sbatch script.

Usage:
    python oellm_pretrain/main.py config/example.yaml
"""

import json
import os
import subprocess
import sys
import re
from itertools import product
from string import Template
from typing import List

import yaml


def divide_rounding_up(a: int, b: int) -> int:
    return (a + b - 1) // b

# Check whether an argument is a valid command-line argument for Megatron-LM
def validate_megatron_args(cfg):
    regex = re.compile(r"\s+group\.add_argument\('--([^']+)'")
    megatron_argfile = os.path.join(os.environ['MEGATRON_PATH'], "megatron/training/arguments.py")
    megatron_args = []
    with open(megatron_argfile, 'r') as f:
        megatron_args = f.readlines()
        for line in megatron_args:
            match = regex.match(line)
            if match:
                arg = match.groups()[0]
                arg = arg.replace("-", "_")
                megatron_args.append(arg)
    
    unknown_keys = set(cfg.keys()).difference(megatron_args)
    assert len(unknown_keys)==0, f"Non-matching Megatron-arguments in the .yaml-file: {unknown_keys}"
    


def maybe_add_derived_configs(cfg):
    # Some args are mutually exclusive
    assert sum(k in cfg for k in ("train_iters", "train_samples", "train_tokens")) == 1
    assert (
        sum(
            k in cfg
            for k in ("lr_decay_iters", "lr_decay_samples", "lr_decay_fraction")
        )
        == 1
    )

    # This is not a Megatron-LM arg. We need to derive the correspondent one.
    if "train_tokens" in cfg:
        train_tokens = cfg.pop("train_tokens")
        train_tokens = int(str(train_tokens).replace("_", ""))
        tokens_per_iter = int(cfg["seq_length"]) * int(cfg["global_batch_size"])
        cfg["train_iters"] = divide_rounding_up(train_tokens, tokens_per_iter)

    # This is not a Megatron-LM arg. We derive the correspondent one here.
    if "lr_decay_fraction" in cfg:
        lr_decay_fraction = cfg.pop("lr_decay_fraction")
        if "train_samples" in cfg:
            cfg["lr_decay_samples"] = int(cfg["train_samples"] * lr_decay_fraction)
        elif "train_iters" in cfg:
            cfg["lr_decay_iters"] = int(cfg["train_iters"] * lr_decay_fraction)

    # Set WSD args. TODO: check if WSD needs lr_wsd_decay_iters or lr_decay_iters and how to set them
    if "lr_decay_iters" in cfg:
        cfg["lr_wsd_decay_iters"] = cfg["lr_decay_iters"]
    elif "lr_decay_samples" in cfg:
        cfg["lr_wsd_decay_samples"] = cfg["lr_decay_samples"]

    return cfg


def dict_to_flags(cfg: dict) -> List[str]:
    """Convert config dict into CLI flags (list of strings)."""
    flags = []
    for k, v in cfg.items():
        if v is None:
            raise ValueError(f"Missing value for {k}.")
        flag = "--" + k.lower().replace("_", "-")
        if isinstance(v, bool):
            if v:
                flags.append(flag)
        elif isinstance(v, (list, tuple)):
            if not v:
                continue
            flags.append(" ".join([flag] + [str(x) for x in v]))
        else:
            flags.append(f"{flag} {v}")
    return flags


def cartesian_product(cfg: dict, sweep_args: List[str]) -> List[dict]:
    """
    Expand cfg into multiple configs over `sweep_args`.
    Example:
        >>> cfg = {'a': 1, 'b': [1, 2], 'c': [3, 4, 5]}
        >>> sweep_args = ['c']
        >>> pools
        [[1], [[1, 2]], [3, 4, 5]]
        >>> cartesian_product(cfg, sweep_args)
        [{'a': 1, 'b': [1, 2], 'c': 3},
         {'a': 1, 'b': [1, 2], 'c': 4},
         {'a': 1, 'b': [1, 2], 'c': 5}]
    """
    # No sweep args, return config as a single-element list
    if not sweep_args:
        return [cfg]
    sweep_args = set(sweep_args)
    pools = [
        v if k in sweep_args and isinstance(v, (list, tuple)) else [v]
        for k, v in cfg.items()
    ]
    return [dict(zip(cfg.keys(), combo)) for combo in product(*pools)]


def make_job_name(base: str, args: dict, sweep_keys: List[str]) -> str:
    """
    Construct a job name by combining a base string with key-value pairs
    from the given args restricted to sweep_keys.
    Example: base='run', args={'lr': 0.01, 'bsz': 32}, sweep_keys=['lr','bsz'] -> 'run_lr0_01_bsz32'.
    """
    parts = [base]
    for k in sweep_keys:
        if k in args:
            val = str(args[k]).replace(".", "_").replace("-", "_")
            parts.append(f"{k}{val}")
    return "_".join(parts)


def main():
    # Parse CLI args
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} config.yaml", file=sys.stderr)
        sys.exit(1)

    # Load YAML config
    yaml_path = sys.argv[1]
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    sbatch_args = cfg["sbatch_args"]
    megatron_args = cfg["megatron_args"]
    sweep_args = cfg.pop("sweep_args", [])  # by default no sweeps (empty list)

    # Dirs
    out_dir = sbatch_args["out_dir"]
    slurm_logs_dir = os.path.join(out_dir, "slurm_logs")
    slurm_scripts_dir = os.path.join(out_dir, "slurm_scripts")
    sweep_path = os.path.join(out_dir, "sweep.json")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(slurm_logs_dir, exist_ok=True)
    os.makedirs(slurm_scripts_dir, exist_ok=True)

    # Add derived configs
    megatron_args = maybe_add_derived_configs(megatron_args)
    validate_megatron_args(megatron_args)
    # Generate all configs
    args_list = cartesian_product(megatron_args, sweep_args)

    # Build sweep list, dump to sweep.json
    sweep = []
    for args in args_list:
        job_name = make_job_name(sbatch_args["job_name"], args, sweep_args)
        sweep.append({"job_name": job_name, "flags": dict_to_flags(args)})
    print(f"Saving sweep to: {sweep_path}.")
    with open(sweep_path, "w") as f:
        json.dump(sweep, f, indent=2)

    # Render sbatch script from template, and write
    pkg_dir = os.path.dirname(__file__)
    #TODO: Make these cli arguments
    template_path = os.path.join(pkg_dir, "template_lumi.sbatch")
    cluster_setup_path = os.path.join(pkg_dir, "setup_lumi.sh")
    with open(template_path) as f:
        sbatch_script = Template(f.read())
    sbatch_script = sbatch_script.safe_substitute(
        out_dir=out_dir,
        load = megatron_args.get("load"),
        save = megatron_args.get("save"), # separate save directory from 
        sweep_path=sweep_path,
        slurm_logs_dir=slurm_logs_dir,
        cluster_setup_script=cluster_setup_path,
        job_name=sbatch_args["job_name"],
        slurm_time=sbatch_args['time'],
        slurm_account=sbatch_args['account'],
        slurm_nodes=sbatch_args['nodes'],
        slurm_partition=sbatch_args['partition']
    )
    sbatch_script_path = os.path.join(
        slurm_scripts_dir, f"{sbatch_args['job_name']}.sbatch"
    )
    print(f"Writing sbatch script to: {sbatch_script_path}.")
    with open(sbatch_script_path, "w") as f:
        f.write(sbatch_script)

    # Submit job array
    cmd = ["sbatch", f"--array=0-{len(sweep) - 1}", sbatch_script_path]
    print("Submitting:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
