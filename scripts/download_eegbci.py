#!/usr/bin/env python
"""Pre-telechargement des EDF EEGBCI depuis le miroir AWS de PhysioNet."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from msvae.data import download_eegbci  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--first", type=int, default=1)
    p.add_argument("--last", type=int, default=60)
    p.add_argument("--runs", type=int, nargs="+", default=[1, 2])
    p.add_argument("--dest", default="cache/eegbci_raw")
    a = p.parse_args()

    ok, fail = 0, []
    for s in range(a.first, a.last + 1):
        for r in a.runs:
            try:
                path = download_eegbci(s, r, a.dest)
                ok += 1
                print(f"S{s:03d}R{r:02d} {path.stat().st_size / 1e6:.1f} MB", flush=True)
            except Exception as exc:
                fail.append((s, r, str(exc)))
                print(f"S{s:03d}R{r:02d} ECHEC {exc}", flush=True)
    print(f"\n{ok} fichiers, {len(fail)} echecs")


if __name__ == "__main__":
    main()
