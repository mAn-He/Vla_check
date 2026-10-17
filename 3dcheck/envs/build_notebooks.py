"""Build .ipynb notebooks from their jupytext percent sources and validate them.

Sources live in <part>/notebooks/src/NN_name.py; outputs go to <part>/notebooks/NN_name.ipynb.
Run in the 3dv env:  python envs/build_notebooks.py [--check] [--part outdoor]

--check  : fail if an .ipynb is missing or differs from its source (does not write).
Validation: nbformat schema, kernelspec present, first cell is markdown with the required sections,
code cells parse as Python.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import jupytext
import nbformat

ROOT = Path(__file__).resolve().parents[1]
# Required first-cell sections (CLAUDE.md "노트북 작성 규칙")
REQUIRED_HEADINGS = ["목적", "비교 모델", "논문", "커널", "VRAM", "질문"]
KERNELS = {"3dv", "3dv-point"}


def check_notebook(nb, name: str) -> list[str]:
    errs = []
    try:
        nbformat.validate(nb)
    except nbformat.ValidationError as e:
        errs.append(f"nbformat: {e.message}")
    ks = nb.metadata.get("kernelspec", {})
    if ks.get("name") not in KERNELS:
        errs.append(f"kernelspec.name must be one of {KERNELS}, got {ks.get('name')}")
    if not nb.cells or nb.cells[0].cell_type != "markdown":
        errs.append("first cell must be markdown")
    else:
        missing = [h for h in REQUIRED_HEADINGS if h not in nb.cells[0].source]
        if missing:
            errs.append(f"first cell missing sections: {missing}")
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        src = "\n".join(line for line in c.source.splitlines() if not line.lstrip().startswith(("!", "%")))
        try:
            ast.parse(src)
        except SyntaxError as e:
            errs.append(f"cell {i}: SyntaxError {e.msg} (line {e.lineno})")
    return [f"{name}: {e}" for e in errs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default=None, help="outdoor | indoor (default: both)")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    parts = [args.part] if args.part else ["outdoor", "indoor"]
    errors, n = [], 0
    for part in parts:
        for src in sorted((ROOT / part / "notebooks" / "src").glob("[0-9][0-9]_*.py")):
            nb = jupytext.read(src)
            nb.metadata.pop("jupytext", None)
            for i, cell in enumerate(nb.cells):  # jupytext assigns random ids; make builds reproducible
                cell.id = f"c{i:03d}"
            out = src.parent.parent / f"{src.stem}.ipynb"
            errors += check_notebook(nb, src.name)
            new_text = nbformat.writes(nb)
            if args.check:
                if not out.exists() or out.read_text() != new_text:
                    errors.append(f"{out.name}: out of date (run envs/build_notebooks.py)")
            else:
                out.write_text(new_text)
            n += 1
    for e in errors:
        print("ERROR", e)
    print(f"{n} notebook(s) {'checked' if args.check else 'built'}, {len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
