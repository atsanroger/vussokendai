#!/bin/bash
# Build a runnable copy of the old AllClass.py WITHOUT touching the repo's files:
#   - cupy -> numpy shim (no GPU needed), '.get()' removed
#   - module-level script at the bottom (fits/plots, starts at "start = time.time()") cut off
# Usage: bash make_patched.sh <workdir> <dir that contains RBCQCD/>; then, in <workdir>:
#   python cmp_kernels.py        (needs numpy scipy pandas gvar lsqfit)
set -e
W=${1:?workdir}; D=${2:?data dir with RBCQCD}; R=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$W"; cp "$R"/{AllClass.py,args.py,my_header.py} "$W"/; cp "$(dirname "$0")/cmp_kernels.py" "$W"/
ln -sfn "$D/RBCQCD" "$W/RBCQCD"; cd "$W"
sed -i 's/^import cupy as cp.*/import numpy as cp  # shim: no GPU here/' my_header.py
sed -i 's/\.get()//g' AllClass.py
n=$(grep -n '^start = time.time()' AllClass.py | cut -d: -f1)
head -n $((n-1)) AllClass.py > AllClass_lib.py
echo "patched copy in $W (AllClass_lib.py)"
