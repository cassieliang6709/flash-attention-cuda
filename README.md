# flash-attention-cuda

从零用 CUDA 写 FlashAttention **前向**：naive → FA v1 → FA v2 → 由 Nsight Compute 决定的后续优化。通过 PyTorch extension 接入，和 PyTorch SDPA 做精度对齐和性能对比。

> 规则：本 README 里的每个性能数字都要注明来自 `results/` 下的哪个文件。还没测的写「待测」，不估。

## 状态

| 步骤 | 文件 | 精度测试 | 性能数字 |
|---|---|---|---|
| naive（3 个 kernel，S 写回 HBM） | `csrc/naive.cu` | 通过 | 见「结果」 |
| FA v1（外层循环 KV，O/l/m 放在 HBM） | `csrc/fa1.cu` | 通过 | 见「结果」 |
| FA v2a（外层循环 Q，O 放在寄存器，按 seq 维并行） | `csrc/fa2.cu` | 通过 | 见「结果」 |
| FA v2b（WMMA tensor core） | 未开始，等 ncu 确认 v2a 的瓶颈再做 | — | — |

精度测试：RTX 4090 上 `pytest tests/` 结果为 27 passed（`results/pytest_4090.txt`）。
Nsight Compute：**还没有数据**。AutoDL 的容器读不到性能计数器（`ERR_NVGPUCTRPERM`），下面的瓶颈判断都是 roofline **推算**，不是 ncu 实测。

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

## 结果（RTX 4090，2026-10-26）

环境：RTX 4090 24 GB，驱动 595.71.05，CUDA 12.8，torch 2.8.0+cu128（`results/env_4090.txt`）。
代码版本：commit `4730a67`，外加 `bench.py` 新增的显存列（下一个 commit 提交）。CSV 里的 `git_rev` 一列是 `unknown`，因为远端 git 报了 dubious ownership，没能读出版本号。
设置：B = 4，H = 2048 / D（D = 64 时 32 个 head，D = 128 时 16 个），non-causal。每次计时前先 flush L2，报告 30 次的中位数。单次超过 1 s 的配置只测 3 次。

文件缩写：
- **[B]** = `results/bench_NVIDIA_GeForce_RTX_4090_20261026-005135.csv`（原始耗时和显存，144 行）
- **[R]** = `results/roofline_NVIDIA_GeForce_RTX_4090_20261026-005135.csv`（由 `bench/roofline.py` 从 [B] 推算）
- **[A]** = `results/accuracy_NVIDIA_GeForce_RTX_4090.csv`（最大误差）

### 代表性形状：fp16，N = 4096，D = 64

| impl | ms | TFLOP/s | 额外显存 (MiB) | 来源 |
|---|---|---|---|---|
| naive | 291.39 | 1.89 | 8256 | [B] `naive,fp16,4,32,4096,64` |
| torch_naive | 跳过：S 需要约 12 B/元素，超过 10 GiB 上限 | — | — | [B] |
| fa1 | 687.84 | 0.80 | 196 | [B] `fa1,fp16,4,32,4096,64` |
| fa2 (v2a) | 245.00 | 2.24 | 64 | [B] `fa2,fp16,4,32,4096,64` |
| sdpa_flash | 3.36 | 163.63 | 66 | [B] `sdpa_flash,fp16,4,32,4096,64` |

### 关键比值（都由 [B] 的 `ms_median` 算出，覆盖所有能跑的 N）

| 对比 | D = 64 | D = 128 |
|---|---|---|
| **fa1 / fa2 耗时**（v2 的收益） | fp16 2.38–2.81×，fp32 2.39–2.90× | fp16 2.63–3.93×，fp32 2.67–3.98× |
| naive / fa2 耗时 | 1.17–1.33×（fa2 更快） | **0.60–0.64×（naive 更快）** |
| torch_naive / fa2 耗时 | 0.25–0.30×（cuBLAS 更快） | 0.07–0.08× |
| fa2 / sdpa_flash 耗时（fp16） | 45–73×（SDPA 更快） | 98–131× |

v1 → v2 的加速比随 N 和 D 增大而变大：D = 128、N = 16384 时最大，fp16 是 3.93×，fp32 是 3.98×。

### 显存：FA 的真正收益

fp16，D = 64，额外显存（[B] 的 `extra_mem_MiB`，即调用期间峰值显存减去调用前的分配量，包含输出 O）：

| N | naive | torch_naive | fa1 | fa2 | sdpa_flash |
|---|---|---|---|---|---|
| 2048 | 2080 | 5120 | 98 | 32 | 33 |
| 4096 | 8256 | 跳过 | 196 | **64** | 66 |
| 8192 | 跳过 | 跳过 | 392 | 128 | 132 |

- N = 4096 时，fa2 的额外显存比 naive 少 **129×**（8256 / 64）。fa2 的 64 MiB 正好是输出 O 本身（4·32·4096·64·2 B），说明它没有用任何 O(N²) 的 workspace，和 SDPA 一样。
- N = 8192 时，naive 的 S 矩阵需要 4·32·8192²·4 B = 32 GiB，超过了 4090 的 24 GB。fa1 和 fa2 能一直跑到 N = 16384。

### 精度（[A]，相对 fp64 参考的最大绝对误差）

- **fp32**：fa1 为 2.8e-7 到 4.3e-7，fa2 为 2.0e-7 到 4.6e-7，都低于 SDPA 在同一输入上的误差（比值 0.33–0.63）。
- **fp16**：fa1 和 fa2 的误差是 SDPA 误差的 0.98–1.00 倍，例如 N = 1000、D = 64 时都是 1.194e-4。这是因为标量版本全程用 fp32 累加，误差几乎全部来自最后舍入成 fp16 那一步。

### Roofline 推算（不是 ncu）

[R] 里每一行都由 `bench/roofline.py` 计算：先用 4N²D 和实测耗时算出达到的 TFLOP/s，再拿一个不考虑 cache 复用的字节模型算出 GB/s，最后和 4090 的规格峰值比较。规格峰值来自 NVIDIA Ada 白皮书：FP32 CUDA core 82.6 TFLOP/s，FP16 tensor core（FP32 累加）165.2 TFLOP/s，DRAM 1008 GB/s。

| impl | 占 FP32 峰值 | 占 DRAM 峰值（模型上界） | 判断 |
|---|---|---|---|
| naive | 2.1–2.4 % | 6.2–13.2 % | 两者都没饱和（16/16 行） |
| fa1 | 0.5–1.2 % | 3.0–9.9 % | 两者都没饱和（24/24 行） |
| fa2 (v2a) | 1.4–2.9 % | 3.9–15.1 % | 两者都没饱和（24/24 行） |
| torch_naive | 4.8–20.5 % | 55.8–87.5 % | memory bound（12/12 行），符合 §1 的分析 |
| sdpa_flash (fp16) | 63.5–100.3 %（tensor 峰值） | 1.9–45.2 % | compute bound（12/12 行） |

**结论：** 三个自己写的 kernel 既不是 memory bound，也不是 compute bound，而是 **latency / occupancy bound**。所以下一步不该先去省字节，而是先让 SM 忙起来。SDPA 在 N ≥ 4096 时能达到 99–100 % 的 tensor 峰值，说明 4090 上这个问题的天花板确实是 tensor core。SDPA 有几行略超过 100 %，是因为实际 boost 频率高于白皮书的标称值；`env_4090.txt` 里记录的最高 SM 频率是 3105 MHz。

fa2 达不到峰值，可能的原因如下。**这些都是推测，要等 ncu 验证：**
1. **每个 block 只有 1 个 warp（32 线程）。** 同时每个 block 要用 24 KB（D = 64）或 48 KB（D = 128）SMEM，一个 SM 上同时驻留的 warp 很少，延迟藏不住。
2. **`Qs[tx * D + d]` 有 32-way bank conflict。** 这一条是从地址算出来的，不是猜的：D 是 32 的倍数，同一个 warp 里所有线程在同一个 d 上访问的是同一个 bank 的不同地址。（`Ks`、`Vs` 所有线程读同一个地址，是广播，没有冲突。）
3. **D = 128 时每个线程用 241–254 个寄存器。** 这进一步限制了 occupancy，也能解释为什么 D = 128 时 fa2 反而比 naive 慢。

## 负面结果

- **v2a 在 D = 128 时比 naive 慢**（naive / fa2 = 0.60–0.64×，[B]）。原因见上面第 1 和第 3 条。这说明「融合」本身并不保证更快：如果 kernel 的 occupancy 很差，省下来的 HBM 流量换不回时间。
- **v2a 比 cuBLAS 版的 torch_naive 还慢 3.3–14×，比 SDPA 慢 45–131×。** 标量 FMA 版本的上限是 82.6 TFLOP/s，而实际只达到了 1.4–2.9 %。和 torch_naive 比，fusion 能省下的 HBM 流量远远抵不上我们在计算上的劣势。
- **v1 比 naive 还慢**（fa1 在所有形状下都慢于 naive，[B]）。§3 的估算已经预料到了：Bc = 32、O 用 fp32 时，v1 的 O 流量和 naive 的 S 流量同一量级。再加上 grid 只有 B·H = 128 或 64 个 block，每个 block 只有 1 个 warp，SM 也没占满。
- **ncu 在 AutoDL 上用不了**（`ERR_NVGPUCTRPERM`），所以上面的瓶颈判断还只是推算。

## 下一步

1. 在 lium 的 profiling 机器上跑 `profile/ncu.sh fa1 4096 64` 和 `fa2 4096 64`，用 SpeedOfLight、occupancy、bank conflict 计数和 stall reasons 验证上面三条推测。
2. 按 ncu 指出的方向改 v2：每个 block 放 4 个 warp，SMEM 加 padding（`D + 1`），然后做 v2b（WMMA）。

## 简历描述

> 从零用 CUDA 实现 FlashAttention 前向（naive / v1 / v2，PyTorch C++ extension），fp32 / fp16 与 fp64 参考对齐，fp16 误差与 PyTorch SDPA 持平（0.98–1.00×）。v2 交换 Q 与 K/V 的循环顺序，O 留在寄存器里，并按序列维度并行，在 RTX 4090 上比 v1 快 2.4–4.0×。N = 4096 时额外显存比 naive attention 少 129×（64 MiB vs 8.1 GiB），能跑到 N = 16k，而 naive 在 N ≥ 8k 时会超出显存。用 roofline 推算定位出标量版本属于 occupancy / latency bound（只达到 FP32 峰值的 1.4–2.9 %），作为下一步做 tensor core 版本的依据。

每个数字的来源：
- 「0.98–1.00×」：[A] 中 fp16 行的 `ratio_to_sdpa`。
- 「2.4–4.0×」：[B] 中 fa1 / fa2 的 `ms_median` 之比，在所有 dtype、D、N 下的范围是 2.38–3.98。
- 「129×、64 MiB vs 8.1 GiB」：[B] 中 `fp16,4,32,4096,64` 的 `extra_mem_MiB`，fa2 是 64，naive 是 8256。
- 「N ≥ 8k 超出显存」：按 4·32·N²·4 B 计算，N = 8192 时是 32 GiB，大于 24 GB。[B] 中对应的行标记为 `skipped_S_too_big`。
- 「1.4–2.9 %」：[R] 中 fa2 行的 `pct_peak_flops`。

**注意：** 这条描述里没有写「比 SDPA 快」，也没有写「ncu 分析」，因为现在既没有这样的数据，事实上也不成立。等 lium 上的 ncu 数据补齐之后，再把「roofline 推算」改成「ncu 分析」。
