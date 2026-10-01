"""Diff recomputed tables against released ones, cell by cell.

For each (new, old, keys) triple: rows are matched on the key columns (all
columns when no keys are given, in file order); every shared numeric column is
compared as floats and every other column as text. The released file may cover
more rows than the recomputed one (it was cut from a larger run set); the
report says how many rows each side has and how many matched.

Writes one line per file: max absolute difference, number of differing cells,
columns that differ, and the verdict (identical / within 1e-9 / DIFFERENT).
Exit status is 0 even when files differ: the report is the result.

Usage: python diff_tables.py --out REPORT.csv --new A.csv ... --old A0.csv ... --keys "k1,k2" ... (use - for none)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 1e-9
# Free-text columns that record where and how a file was made, not a result.
IGNORE = {"notes"}


def compare(new: Path, old: Path, keys: list[str]) -> dict:
    a, b = pd.read_csv(new), pd.read_csv(old)
    rec = dict(new=str(new), old=str(old), rows_new=len(a), rows_old=len(b))
    if new.read_bytes() == old.read_bytes():
        return dict(rec, rows_matched=len(a), max_abs_diff=0.0, n_cells_differ=0,
                    columns_differ="", verdict="identical (bytes)")
    cols = [c for c in a.columns if c in b.columns and c not in IGNORE]
    missing = sorted(set(a.columns) ^ set(b.columns))
    if keys:
        a = a.set_index(keys)
        b = b.set_index(keys)
        idx = a.index.intersection(b.index)
        a, b = a.loc[idx], b.loc[idx]
        cols = [c for c in cols if c not in keys]
    else:
        if len(a) != len(b):
            return dict(rec, rows_matched=0, max_abs_diff=np.nan, n_cells_differ=np.nan,
                        columns_differ="row count", verdict="DIFFERENT")
    worst, n_bad, bad_cols = 0.0, 0, []
    for c in cols:
        x, y = a[c], b[c]
        if (pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y)
                and not pd.api.types.is_bool_dtype(x)):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            both_nan = np.isnan(xv) & np.isnan(yv)
            d = np.where(both_nan, 0.0, np.abs(xv - yv))
            d = np.where(np.isnan(d), np.inf, d)
            k = int((d > TOL).sum())
            if len(d):
                worst = max(worst, float(d.max()))
        else:
            k = int((x.astype(str).to_numpy() != y.astype(str).to_numpy()).sum())
        if k:
            n_bad += k
            bad_cols.append(c)
    verdict = "within 1e-9" if n_bad == 0 else "DIFFERENT"
    if missing:
        verdict += f" (columns only on one side: {', '.join(missing)})"
    return dict(rec, rows_matched=len(a), max_abs_diff=worst, n_cells_differ=n_bad,
                columns_differ=",".join(bad_cols), verdict=verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--new", nargs="+", required=True)
    ap.add_argument("--old", nargs="+", required=True)
    ap.add_argument("--keys", nargs="+", required=True)
    a = ap.parse_args()
    assert len(a.new) == len(a.old) == len(a.keys)
    rows = [compare(Path(n), Path(o), [k for k in ks.split(",") if k and k != "-"])
            for n, o, ks in zip(a.new, a.old, a.keys)]
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    print(df[["new", "rows_new", "rows_old", "rows_matched", "max_abs_diff",
              "n_cells_differ", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
