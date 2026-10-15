#!/usr/bin/env bash
# Meant to be sourced: "source ./run-diff-analysis-code.sh"; uses return, not exit, to keep the caller's shell alive.

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || return 1

# A running VS Code would receive "code ." via IPC and ignore the pixi environment set up below.
if pgrep -u "$(id -un)" -x code >/dev/null 2>&1; then
    echo "VS Code is already running. Quit it completely, then source this script again." >&2
    unset _script_dir
    return 1
fi

export PIXI_CACHE_DIR="/tmp/$(id -un)-pixi-cache"

# /run/user/<uid> is removed when the login session ends, which breaks kernel restarts.
export JUPYTER_RUNTIME_DIR="$HOME/.local/share/jupyter/runtime"
mkdir -p -m 700 "$JUPYTER_RUNTIME_DIR" || { unset _script_dir; return 1; }

pushd "$_script_dir" >/dev/null || { unset _script_dir; return 1; }
if pixi run python -m ipykernel install --user --name diffmap --display-name "diffmap (pixi)"; then
    pixi run diff-analysis-code
    _status=$?
else
    echo "Kernel installation failed; VS Code was not launched." >&2
    _status=1
fi
popd >/dev/null
unset _script_dir
return "$_status"
