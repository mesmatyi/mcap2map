"""mcap2map command line.

  python -m mcap2map run BAG [--out DIR] [--config CFG.json] [--from STAGE] [--to STAGE] [--only STAGE] [--force S ...]
  python -m mcap2map batch DIR [--out-root ROOT] [--config CFG.json]   every *.mcap in DIR, failures do not stop it
  python -m mcap2map inspect BAG
  python -m mcap2map status DIR
  python -m mcap2map stages
"""
import argparse
import glob
import json
import os
import sys
import time

from . import inspect_bag
from .pipeline import STAGES, Run, StageError


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mcap2map", description="MCAP bag -> point cloud map, RGB point cloud map, "
                                 "Lanelet2 map (Lexus3-style NovAtel INS + lidar + cameras), no manual steps.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run (or resume) the whole pipeline")
    r.add_argument("bag")
    r.add_argument("--out", help="run directory (default: <bag stem>_maps next to the bag)")
    r.add_argument("--config", help="JSON overrides: frame {name, ox, oy}, topics, params, cameras_exclude, profile=false")
    r.add_argument("--from", dest="start", choices=STAGES, help="redo this stage and everything after it")
    r.add_argument("--to", dest="stop", choices=STAGES, help="stop after this stage")
    r.add_argument("--only", choices=STAGES, help="run just this stage")
    r.add_argument("--force", nargs="*", default=[], choices=STAGES, help="redo these stages even if up to date")
    r.add_argument("--python", help="interpreter for the tools (default: this one)")
    b = sub.add_parser("batch", help="map every *.mcap in a folder (resumes finished ones, continues past failures)")
    b.add_argument("dir")
    b.add_argument("--out-root", help="parent of the run directories (default: next to each bag)")
    b.add_argument("--config", help="JSON overrides applied to every bag")
    b.add_argument("--python", help="interpreter for the tools (default: this one)")
    i = sub.add_parser("inspect", help="show what would be detected in a bag")
    i.add_argument("bag")
    s = sub.add_parser("status", help="stage status of a run directory")
    s.add_argument("out")
    sub.add_parser("stages", help="list the stages")
    a = ap.parse_args(argv)

    if a.cmd == "stages":
        print("\n".join(STAGES))
        return 0
    if a.cmd == "inspect":
        print(inspect_bag.describe(inspect_bag.inspect(a.bag)))
        return 0
    if a.cmd == "status":
        for st in STAGES:
            p = os.path.join(a.out, "stages", f"{st}.json")
            d = json.load(open(p)) if os.path.exists(p) else None
            print(f"{st:13s} {'done ' + d['finished'] + f' ({d['seconds']:.0f} s)' if d else '-'}")
        return 0
    if a.cmd == "batch":
        bags = sorted(glob.glob(os.path.join(a.dir, "*.mcap")))
        summary = []
        for bag in bags:
            stem = os.path.splitext(os.path.basename(bag))[0]
            out = os.path.join(a.out_root or os.path.dirname(os.path.abspath(bag)), stem + "_maps")
            args = ["run", bag, "--out", out] + (["--config", a.config] if a.config else []) + \
                (["--python", a.python] if a.python else [])
            print(f"=== {bag} -> {out}", flush=True)
            try:
                rc = main(args)
            except Exception as e:  # a broken bag must not stop the batch
                print(f"ERROR {bag}: {e}", flush=True)
                rc = 3
            summary.append((bag, out, rc))
        print("\nbatch summary:")
        for bag, out, rc in summary:
            print(f"  {'ok    ' if rc == 0 else f'FAILED({rc})'} {os.path.basename(bag)} -> {out}")
        return 0 if all(rc == 0 for _, _, rc in summary) else 1
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.bag)),
                                os.path.splitext(os.path.basename(a.bag))[0] + "_maps")
    user = json.load(open(a.config)) if a.config else {}
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "pipeline.log"), "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    run = Run(a.bag, out, user, a.python, log)
    try:
        run.run(start=a.start, stop=a.stop, force=a.force, only=a.only)
    except StageError as e:
        log(f"ERROR: {e}")
        return 2
    except Exception as e:  # unexpected: keep the traceback in pipeline.log
        import traceback
        log(f"ERROR in stage {run.stage}: {e}\n{traceback.format_exc()}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
