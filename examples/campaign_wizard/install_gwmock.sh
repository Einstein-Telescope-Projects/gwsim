#!/usr/bin/env bash
# install_gwmock.sh -- one-shot bootstrap for gwmock v0.8.2 + gwmock-wizard.
#
# Author: Gianluca Inguglia <gianluca.inguglia@oeaw.ac.at>
#
# Creates an isolated Python env, installs gwmock, then runs the readiness checks
# the wizard relies on:
#   (1) `import gwmock`                 -> the package imports cleanly
#   (2) `gwmock simulate --help`        -> the generated launchers can call it
#   (3) version >= 0.8                  -> the new globals + orchestration schema
#
# gwmock lives at https://github.com/Leuven-Gravity-Institute/gwmock and is
# published on PyPI, so `uv pip install gwmock` is the default path.
#
# Usage:
#   ./install_gwmock.sh                 # install gwmock from PyPI (DEFAULT)
#   ./install_gwmock.sh --pypi          # same as default
#   ./install_gwmock.sh --source        # clone GitHub + install the clone into the env
#   GWMOCK_ENV=/path/to/env ./install_gwmock.sh
#
# NOTE: the wizard's `interview` and `generate` steps are stdlib-only and run
# without gwmock installed -- gwmock is only needed at run time, when the
# launchers call `gwmock simulate`.
set -euo pipefail

ENV_DIR="${GWMOCK_ENV:-$(pwd)/gwmock-env}"
SOURCE_REPO="https://github.com/Leuven-Gravity-Institute/gwmock.git"
FROM_SOURCE=0
case "${1:-}" in
  --source)        FROM_SOURCE=1 ;;
  --pypi|"")       FROM_SOURCE=0 ;;
  *) echo "unknown arg '${1}' (use --pypi [default] or --source)"; exit 2 ;;
esac

say() { printf '\n[install_gwmock] %s\n' "$*"; }

# --- 0. python >=3.12 ------------------------------------------------------- #
PY="${PYTHON:-python3}"
"$PY" - <<'PY' || { echo "Need Python >= 3.12"; exit 1; }
import sys; sys.exit(0 if sys.version_info[:2] >= (3,12) else 1)
PY

# --- 1. make the env (prefer uv, fall back to venv+pip) --------------------- #
if command -v uv >/dev/null 2>&1; then
  BACKEND=uv
  say "uv found -> creating venv at $ENV_DIR (Python 3.13)"
  uv venv --python 3.13 "$ENV_DIR"
else
  BACKEND=pip
  # The stdlib venv inherits the interpreter it is built from, so refuse to
  # build a <3.12 env: pick a >=3.12 interpreter or fail loudly.
  VENV_PY=""
  for cand in python3.13 python3.12; do
    if command -v "$cand" >/dev/null 2>&1; then VENV_PY="$cand"; break; fi
  done
  if [[ -z "$VENV_PY" ]]; then
    if "$PY" - <<'PY'
import sys; sys.exit(0 if sys.version_info[:2] >= (3,12) else 1)
PY
    then
      VENV_PY="$PY"
    else
      FOUND_VER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
      echo "Need Python >= 3.12 for gwmock; found ${FOUND_VER}"
      exit 1
    fi
  fi
  say "uv not found -> using stdlib venv + pip at $ENV_DIR (via $VENV_PY)"
  "$VENV_PY" -m venv "$ENV_DIR"
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

pipi() { if [[ "$BACKEND" == uv ]]; then uv pip install "$@"; else python -m pip install --upgrade pip >/dev/null && python -m pip install "$@"; fi; }

# --- 2. install gwmock ------------------------------------------------------ #
if [[ "$FROM_SOURCE" == 1 ]]; then
  say "installing gwmock from source ($SOURCE_REPO)"
  CLONE="${ENV_DIR%/}-src"
  [[ -d "$CLONE/.git" ]] || git clone "$SOURCE_REPO" "$CLONE"
  # Install the clone INTO $ENV_DIR so `source $ENV_DIR/bin/activate` exposes
  # gwmock. (Do NOT `uv sync` inside the clone -- that builds the clone's own
  # .venv, which the activation path below never touches.)
  pipi "$CLONE"
else
  say "installing gwmock from PyPI"
  pipi gwmock
fi

# --- 3. verify the install -------------------------------------------------- #
say "verifying installation"
VERIFY_RC=0
python - <<'PY' || VERIFY_RC=$?
import sys
try:
    import gwmock
except Exception as e:
    print(f"  [!!] import gwmock           -> FAILED: {e}")
    sys.exit(1)
print(f"  [ok] import gwmock           -> {gwmock.__file__}")

import importlib.metadata as md
try:
    ver = md.version("gwmock")
except Exception as e:
    print(f"  [!!] gwmock version          -> could not resolve: {e}")
    sys.exit(1)

def parsed_ge_0_8(v):
    try:
        from packaging.version import Version
        return Version(v) >= Version("0.8")
    except Exception:
        parts = v.split(".")
        major = int(parts[0]) if parts and parts[0].isdigit() else 0
        minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return (major, minor) >= (0, 8)

if not parsed_ge_0_8(ver):
    print(f"  [!!] gwmock version          -> {ver} (need >= 0.8)")
    sys.exit(1)
print(f"  [ok] gwmock version          -> {ver}")
sys.exit(0)
PY

# `gwmock simulate` must be a real subcommand (the wizard's launchers call it).
if gwmock simulate --help >/dev/null 2>&1; then
  echo "  [ok] gwmock simulate subcmd  -> available"
else
  echo "  [!!] gwmock simulate subcmd  -> NOT a valid subcommand"
  VERIFY_RC=1
fi

if [[ "$VERIFY_RC" -ne 0 ]]; then
  say "VERIFICATION FAILED -- gwmock did not install cleanly."
  say "Fix: remove the env and reinstall:"
  echo "    rm -rf \"$ENV_DIR\" && ./install_gwmock.sh"
  say "or try the source install:"
  echo "    ./install_gwmock.sh --source"
  exit 1
fi

# --- 4. next steps ---------------------------------------------------------- #
WIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gwmock_wizard.py"
say "gwmock is ready. Activate this env in any new shell with:"
echo "    source $ENV_DIR/bin/activate"
say "then drive a campaign with the wizard:"
echo "    python3 $WIZ interview"
echo "    python3 $WIZ generate <name>.campaign.json"
echo "    python3 $WIZ run <name>.campaign.json --submit"
