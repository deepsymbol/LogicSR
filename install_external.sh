#!/bin/bash
# Installs the external dependencies of LogicSR. Run from the repository root,
# inside the conda environment created for LogicSR (see README.md).
set -e

echo "=== LogicSR External Dependencies Installer ==="

if [ ! -f "test.py" ]; then
    echo "Error: please run this script from the LogicSR root directory"
    exit 1
fi

# 1. ABC
echo "1. Installing ABC..."
if [ ! -d "external/abc" ]; then
    git clone https://github.com/berkeley-abc/abc.git external/abc
    make -C external/abc ABC_USE_NO_READLINE=1
else
    echo "  external/abc already exists, skipping"
fi

# 2. Boolformer pretrained weights
echo "2. Downloading Boolformer models..."
if [ ! -f "external/boolformer/boolformer_noiseless.pt" ]; then
    gdown 1cULlE16yKBqUZMMGv7CC5fgHXXJ7OQQQ -O external/boolformer/boolformer_noiseless.pt
else
    echo "  noiseless model already exists, skipping"
fi
if [ ! -f "external/boolformer/boolformer_noisy.pt" ]; then
    gdown 1IFSc_sHfiTckMy-cwggvVMBH1YgXas8G -O external/boolformer/boolformer_noisy.pt
else
    echo "  noisy model already exists, skipping"
fi

# 3. NeuraLUT
echo "3. Installing NeuraLUT dependencies..."
conda install -y packaging pyparsing
conda install -y docrep -c conda-forge
pip install --no-cache-dir git+https://github.com/Xilinx/brevitas.git@67be9b58c1c63d3923cac430ade2552d0db67ba5
pip install pyverilator==0.7.0

# 4. DiffLogic
echo "4. Installing DiffLogic..."
(cd external/difflogic && python setup.py install)

# 5. PySR (SymbolicRegression.jl v1.11.3 through PyJulia)
echo "5. Installing Julia and SymbolicRegression.jl..."
conda install -c conda-forge julia -y
export JULIA_PROJECT="$PWD/julia_env"
julia -e 'using Pkg; Pkg.add(name="SymbolicRegression", version="1.11.3"); Pkg.add("PyCall")'
echo "  Set JULIA_PROJECT=$JULIA_PROJECT before running sr_pysr or sr_pysr_multi"

# 6. SATNet (needs CUDA and nvcc)
echo "6. Installing SATNet..."
if command -v nvcc >/dev/null 2>&1; then
    if [ ! -d "external/SATNet" ]; then
        git clone https://github.com/locuslab/SATNet.git external/SATNet
    fi
    (cd external/SATNet && pip install -e .)
else
    echo "  nvcc not found, skipping SATNet"
fi

echo ""
echo "=== Installation complete ==="
