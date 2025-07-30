if ! command -v uv &> /dev/null && ! command -v python &> /dev/null; then
    if ! command -v python &> /dev/null; then
        winget install -e --id Python.Python.3.12
    fi
    if ! command -v python &> /dev/null; then
        echo "uv and python are not installed and trying to install python (assuming windows) did not work." >&2
        echo "Install python or uv on this machine and rerun this script." >&2
        return 2
    fi
fi

if ! command -v uv &> /dev/null; then
    pip install uv
fi
if ! command -v uv &> /dev/null; then
    echo "uv is not installed and pip failed to install it." >&2
    echo "Install uv on this machine and rerun this script." >&2
    return 2    
fi

uv venv --clear
uv pip install -e .
echo '' | uv run -