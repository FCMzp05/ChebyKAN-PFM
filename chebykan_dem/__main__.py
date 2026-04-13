"""Command-line entry point for the public ChebyKAN-DEM repository."""

from __future__ import annotations

import argparse
import sys

from chebykan_dem.cases import CASE_HELP, DEFAULT_OUTPUT_BY_CASE, require_mesh_or_exit, resolve_output_dir
from chebykan_dem.config import conf, set_output_dir
from chebykan_dem.solver import run_pipeline
from chebykan_dem.utils import log


def main() -> None:
    choices = list(DEFAULT_OUTPUT_BY_CASE.keys())
    parser = argparse.ArgumentParser(
        description=(
            "ChebyKAN-DEM phase-field fracture solver. "
            "Homogeneous and heterogeneous benchmark entry points follow the same "
            "case-directory and output conventions."
        )
    )
    parser.add_argument(
        "--case",
        type=str,
        default="sent_homogeneous",
        choices=choices,
        help="Benchmark id. Output folder names follow the manuscript / revisecode naming convention.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Override the output directory. If omitted, the case default is used.",
    )
    args = parser.parse_args()

    if args.case == "heterogeneous_user":
        log("Case `heterogeneous_user` is a template utility for custom material inputs.")
        log("Run: python examples/run_heterogeneous_user.py")
        log("This creates: <output_dir>/material_template.json")
        sys.exit(0)

    if args.output:
        set_output_dir(args.output)
    else:
        set_output_dir(resolve_output_dir(args.case))

    log("=" * 60)
    log("ChebyKAN-DEM (deep energy method)")
    log(f"Case: {args.case} — {CASE_HELP.get(args.case, '')}")
    log(f"Device: {conf.device}")
    log(f"Output: {conf.output_dir}")
    log(f"Steps: {conf.n_steps}, u_max: {conf.u_max * 1000:.1f} μm")
    log("=" * 60)

    if args.case != "sent_homogeneous":
        require_mesh_or_exit(conf.output_dir, args.case)

    run_pipeline(
        conf.output_dir,
        case_id=args.case,
        generate_mesh_if_missing=(args.case == "sent_homogeneous"),
    )


if __name__ == "__main__":
    main()
