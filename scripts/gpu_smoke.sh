#!/usr/bin/env bash
# Run this FIRST on a freshly rented machine (takes < 1 min).
# If the ncu step prints ERR_NVGPUCTRPERM, the host does not expose GPU
# performance counters: shut the machine down and pick another one.
set -uo pipefail
echo "== GPU / driver";        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "== nvcc";                nvcc --version | tail -1
echo "== ncu";                 ncu --version | tail -1
echo "== torch";               python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
echo "== ncu counter access"
OUT=$(ncu --metrics sm__cycles_elapsed.avg python -c "import torch; torch.randn(1024, 1024, device='cuda').sum(); torch.cuda.synchronize()" 2>&1)
if echo "$OUT" | grep -q ERR_NVGPUCTRPERM; then
  echo "FAIL: ERR_NVGPUCTRPERM — no performance-counter access on this host"; exit 1
elif echo "$OUT" | grep -q "sm__cycles_elapsed.avg"; then
  echo "OK: ncu can read performance counters"
else
  echo "UNKNOWN — full ncu output:"; echo "$OUT"; exit 1
fi
