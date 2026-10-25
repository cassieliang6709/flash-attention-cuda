# flash-attention-cuda

从零用 CUDA 写 FlashAttention **前向**：naive → FA v1 → FA v2 → 由 Nsight Compute 决定的后续优化。通过 PyTorch extension 接入，和 PyTorch SDPA 做精度对齐和性能对比。

> 规则：本 README 里的每个性能数字都要注明来自 `results/` 下的哪个文件。还没测的写「待测」，不估。

## 状态

| 步骤 | 文件 | 精度测试 | 性能数字 |
|---|---|---|---|
| naive（3 个 kernel，S 写回 HBM） | `csrc/naive.cu` | 待测 | 待测 |
| FA v1（外层循环 KV，O/l/m 放在 HBM） | `csrc/fa1.cu` | 核心循环待写 | 待测 |
| FA v2a（外层循环 Q，O 放在寄存器，按 seq 维并行） | `csrc/fa2.cu` | 核心循环待写 | 待测 |
| FA v2b（WMMA tensor core） | 未开始，先看 v2a 的 ncu 结果再定 | — | — |

范围：只做前向，non-causal，输入 `[B, H, N, D]`，`D ∈ {64, 128}`，fp32 / fp16。

## 怎么跑

```bash
# 0. 新租的机器上先跑这个（不到 1 分钟）；ncu 没权限就换机器
bash scripts/gpu_smoke.sh

# 1. 编译（8.9 = 4090，9.0 = H100）
TORCH_CUDA_ARCH_LIST="8.9" pip install -e . --no-build-isolation

# 2. 精度测试
pytest -q tests/

# 3. benchmark → results/bench_<gpu>_<time>.csv
python bench/bench.py

# 4. Nsight Compute → results/ncu/<impl>_N<N>_D<D>_<dtype>.{ncu-rep,details.csv,raw.csv}
bash profile/ncu.sh fa2 4096 64
```

`.ncu-rep` 用 `scp` 拷回 Mac，在 Nsight Compute GUI 里打开（Mac 版只能看报告，不能 profile）。

---

## 原理

符号：一个 head 有 N 个 token，head_dim 为 D。Q 的 tile 有 Br 行，K/V 的 tile 有 Bc 行。Tr = ⌈N/Br⌉，Tc = ⌈N/Bc⌉。本仓库里 Br = Bc = 32。

### 1. naive：瓶颈在 S 的 HBM 往返

```
S = scale · Q Kᵀ     写 N² 个 float
P = softmax(S)       读 N² + 写 N²（原地）
O = P V              读 N²
```

每个 head 的 HBM 流量里有大约 **4N² × 4 B** 和 N 平方成正比，而 FLOPs 是 4N²D。所以算术强度（FLOP/byte）大约是 D/4，不随 N 增长。D = 64 时约 16 FLOP/byte，远低于 GPU 的 ridge point，因此 **memory bound**。这一点要用 ncu 的 SpeedOfLight 验证，不能只靠推算。

`naive.cu` 里的矩阵乘也是 naive 的（每个线程算一个输出元素，不做 tiling），所以拿它比只能说明「最差情况」。为了把「融合带来的收益」和「好 GEMM 带来的收益」分开，benchmark 里另有一个 `torch_naive`（cuBLAS matmul + torch.softmax）：它同样把 S 写回 HBM，但 GEMM 很快。**FA 和 torch_naive 的差距，才是 fusion 本身的收益。**

### 2. online softmax（复用 CS336 的推导）

推导和 `~/cs336-a2/cs336_systems/flash_pytorch.py` 一致：对每一行维护运行中的最大值 m 和分母 l。每来一块新的分数，就把旧的状态乘以 `alpha = exp(m_old − m_new)` 做 rescale。最后 `O = o / l`，结果和一次性做 softmax 在数学上完全相同。减去 max 是为了让 `exp` 不溢出（对应 `test_large_logits`）。

### 3. FA v1：S 不出芯片，但 O 要反复进出 HBM

循环顺序（论文 Algorithm 1）：**外层循环 K/V tile j，内层循环 Q tile i**。

- 优点：S_ij 只存在于寄存器 / SMEM 里，[N, N] 矩阵不再写进 HBM。
- 代价 1：每一对 (i, j) 都要从 HBM 读出 O_i、l_i、m_i，更新完再写回去。O 的流量 ≈ 2 · Tc · N · D 个 float。v1 每一步还要把 O 归一化一次（除以 l_new）。
- 代价 2：grid 只有 B·H 个 block，每个 block 串行走完所有 (j, i)。B·H 比 SM 数少的时候，GPU 大部分是闲着的。

本仓库的选择：O 的 workspace 用 fp32，最后再 cast。理由是如果每个 K/V tile 之后都把 O 舍入成 fp16，会累积 Tc 次舍入误差。这让 v1 的 O 流量翻倍，算在 v1 的成本里。

> 值得注意：Bc = 32、O 用 fp32 时，O 的流量 2·(N/32)·N·D·4 B 和 naive 的 4N²·4 B 是同一量级（D = 64 时两者相等）。v1 能不能比 naive 快，取决于 L2 能接住多少，以及「S 不出芯片」省下的那部分。**以 ncu 的 `dram__bytes_read/write` 为准。** 论文里 v1 的 IO 复杂度是 Θ(N²D²/M)，M 是 SRAM 大小，tile 越大越好，而我们的 tile 很小。

### 4. FA v2：交换循环 + 按 sequence 维度并行

1. **外层循环 Q tile，内层循环 K/V tile。** 每个 block 只负责一个 Q tile，o、l、m 一直放在寄存器里，最后写一次 O。O 的 HBM 流量从 ~2·Tc·N·D 降到 N·D。代价是 K/V 会被每个 Q tile 各读一遍（Tr 次）。不过同一个 head 的 K/V 只有 2·N·D·2 B（N = 4096、D = 64 时是 1 MB），能放进 L2，所以 DRAM 实际读的次数会远少于 Tr 次。用 ncu 对比 `dram__bytes` 和 L2（`lts__t_bytes`）来验证。
2. **不再逐步归一化**：o 一直保持未归一化的形式，只在最后除一次 l。
3. **grid = (Tr, B·H)**：N = 4096、Br = 32 时，每个 head 有 128 个 block，SM 能被占满。

### 5. 精度测试的容差

- **fp32**：每个元素和 fp64 参考的差在 `atol = rtol = 1e-4` 以内。kernel 内部全程用 fp32 累加，误差应该在 1e-6 量级。1e-4 留了余量，但仍然能抓住 off-by-one、漏 rescale 这类 bug，因为这类 bug 的误差是 O(1)。
- **fp16**：(a) 和 fp64 的差在 `atol = rtol = 1e-2` 以内；(b) 最大绝对误差 ≤ 2 × PyTorch SDPA 在同一输入上的最大绝对误差 + 1e-5。(b) 借鉴 FlashAttention 官方测试的思路：只用一个固定容差，要么太松、要么太严；而「不比生产级 kernel 差两倍以上」是一个相对的、有意义的标准。到 v2b 改用 tensor core 后，P 要先 cast 成 fp16 再乘 V，(b) 这条就会真正起作用。
- 测试形状里有 N = 1000 和 N = 77，它们不是 32 的倍数，用来测最后一个不完整 tile 的 padding 和 mask。

### 6. 用 ncu 判断瓶颈的流程

1. **SpeedOfLight**：看 Compute (SM) % 和 Memory % 哪个接近峰值。memory 高、compute 低 → memory bound，下一步减少字节数或提高复用；compute 高 → 看是哪条 pipe（FMA 还是 tensor）。两个都低 → latency bound，看 occupancy 和 warp stall reasons。
2. **Roofline**：算术强度 = FLOPs / DRAM bytes，FLOPs 用 4N²D，bytes 用 ncu 实测值。和 ridge point（峰值算力 / 峰值带宽）比较。4090 的规格：DRAM 1008 GB/s，FP32 非 tensor 82.6 TFLOP/s，FP16 tensor（FP32 累加）165 TFLOP/s。来源是 NVIDIA Ada 白皮书，**租卡后用 ncu 的 device 属性核对**。
3. **v2a 是标量 FMA**，所以它的算力上限是 FP32 CUDA core，而不是 tensor core。SDPA 用的是 tensor core，两者有差距是预期之内的。这个差距就是 v2b 的动机。
4. 后续可能的优化（**ncu 指向哪个才做哪个**）：SMEM bank conflict（`Qs[tx*D + d]` 这种 stride 为 D 的访问）→ padding；occupancy 低（每个 block 只有 1 个 warp）→ 一个 block 放多个 warp；load 和 compute 串行 → `cp.async` 双缓冲；FMA 吞吐到顶 → WMMA。

## 结果

待测。数字填进来之前，`results/` 下必须已经有对应的 csv / ncu-rep。

| impl | dtype | N | D | ms | TFLOP/s | 来源 |
|---|---|---|---|---|---|---|

## 负面结果

（如实记录：试过但没变快的优化、比预期慢的地方，以及原因。）

## 简历描述

最后再写。每个数字都要标明来自哪个 csv 的哪一行，或哪个 ncu 报告。
