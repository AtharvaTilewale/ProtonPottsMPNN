#!/usr/bin/env bash
# Build the Proton-PottsMPNN environment with uv.
#   ./install.sh            # create .venv and install everything
#   source .venv/bin/activate
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/2] creating uv venv (Python 3.12) -> .venv   (--clear rebuilds a fresh one if it exists)"
uv venv --clear --python 3.12 .venv

echo "[2/2] installing the foundry package (mpnn + foundry core) + extras, in one resolution"
# one command so the resolver keeps BOTH sets — installing them separately can prune the extras.
uv pip install --python .venv/bin/python -e ./foundry -r requirements-extra.txt

echo "[verify] import mpnn + foundry from the packaged tree"
.venv/bin/python -P -c "import mpnn, foundry, inspect; assert 'ProtonPottsMPNN' in inspect.getfile(mpnn), inspect.getfile(mpnn); print('  ok:', inspect.getfile(mpnn))"

echo "[kernel] register the venv as a Jupyter kernel (for the notebook)"
.venv/bin/python -m ipykernel install --user --name protonpottsmpnn --display-name "ProtonPottsMPNN (.venv)" \
  >/dev/null 2>&1 && echo "  ok: select the 'ProtonPottsMPNN (.venv)' kernel" \
  || echo "  (skipped — just run:  .venv/bin/jupyter lab inference/design_ph.ipynb)"

echo
echo "done.  activate with:  source .venv/bin/activate"
echo "notebook:  jupyter lab inference/design_ph.ipynb   (pick the 'ProtonPottsMPNN (.venv)' kernel)"
echo "script:    python inference/design_ph.py"
