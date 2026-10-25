#!/usr/bin/env bash
# Profile one implementation with Nsight Compute and export the numbers.
#
#   bash profile/ncu.sh fa2 4096 64          # impl N D [dtype]
#
# Writes results/ncu/<impl>_N<N>_D<D>_<dtype>.ncu-rep  (open in the Nsight
# Compute GUI on the Mac) plus .csv exports of the details page and raw
# metrics, so the numbers quoted in the README are reproducible from git.
set -euo pipefail
IMPL=${1:?impl}; N=${2:-4096}; D=${3:-64}; DTYPE=${4:-fp16}
OUT=results/ncu/${IMPL}_N${N}_D${D}_${DTYPE}
mkdir -p results/ncu

# --set full: all sections incl. roofline (SpeedOfLight), memory workload,
#             warp stall reasons, source counters (needs -lineinfo, set in setup.py).
# --nvtx-include profiled/: only kernels launched inside run_one.py's range.
ncu --set full --nvtx --nvtx-include "profiled/" --import-source yes \
    -f -o "$OUT" \
    python profile/run_one.py --impl "$IMPL" --N "$N" --D "$D" --dtype "$DTYPE"

ncu --import "$OUT.ncu-rep" --page details --csv > "$OUT.details.csv"
ncu --import "$OUT.ncu-rep" --page raw --csv \
    --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active \
    > "$OUT.raw.csv"
echo "wrote $OUT.{ncu-rep,details.csv,raw.csv}"
