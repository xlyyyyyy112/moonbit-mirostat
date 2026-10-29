# moonbit-mirostat

一个与模型无关的 MoonBit Mirostat 自适应采样实现。调用方每生成一个 token，传入一行 logits 和一个 `[0, 1)` 均匀随机数。库返回选中的 token、它在原始模型分布下的概率与惊讶度、保留的候选数量，以及更新后的反馈状态。

本包实现了 [Mirostat 论文](https://arxiv.org/abs/2007.14966)中的两个算法：V1 估计 Zipf 指数并自适应选择 top-k；V2 保留模型惊讶度不超过当前 `mu` 的 token。反馈规则为 `mu := mu - eta * (observed_surprise - tau)`。`tau`、`mu` 和惊讶度均以 bit 为单位。

## 适用场景

- MoonBit 文本生成循环可以控制生成 token 的惊讶度，而不必为每一步固定 top-k 或 top-p 阈值。
- WebAssembly 应用可以在模型推理后于本地完成采样，不必通过 JavaScript 执行 token 选择。
- 解码实验可以将相同的 logits 和随机数依次交给 V1、V2，比较保留词表和惊讶度轨迹。

本包不运行语言模型、不进行分词、不下载模型权重，也不管理随机数生成器。logits 和随机数由调用方提供。本包不承诺与某个模型运行时完全等价：运行时可能会在 Mirostat 前应用重复惩罚、语法掩码、温度等变换，这些变换会改变输入分布。

## 安装与最小使用

需要 MoonBit 编译器 `moonc >= 0.10.16`。当前 CI 在 Linux、macOS、Windows 上固定使用
`moonc v0.10.18+6c83a7f62`，Node 桥接测试使用 Node.js 22。

在已有 MoonBit 项目中安装已发布的库：

```sh
moon add xlyyyyyy112/moonbit-mirostat@0.1.1
```

在使用它的包的 `moon.pkg` 中加入：

```moonbit nocheck
///|
import {
  "xlyyyyyy112/moonbit-mirostat" @mirostat,
}
```

在可执行包的 `main` 函数中调用：

```moonbit nocheck
///|
fn main {
  let config = @mirostat.Config::new(3.0, eta=0.1).unwrap()
  let sampler = @mirostat.Sampler::new(config, version=@mirostat.Version::v2())
  let step = sampler.sample([4.0, 2.0, 1.0], 0.42).unwrap()
  println(step.token())
}
```

库的核心算法和 `jsonio` 支持 wasm、wasm-gc、JavaScript 和 native。
`bridge` 是给 Node/Hugging Face 使用的 CommonJS 导出库，只构建 JavaScript 目标；
它不是原生命令行程序。

## 从源码构建与运行

```sh
git clone https://github.com/xlyyyyyy112/moonbit-mirostat.git
cd moonbit-mirostat
moon update
```

仓库示例使用源码中的库，下面的 `cmd/main` 不需要模型权重、Python 或网络服务。

```sh
moon check --deny-warn
moon build --deny-warn
moon test --deny-warn
moon run cmd/main
moon run cmd/experiment
moon run cmd/generate
moon run cmd/batch
moon run cmd/weights
moon run cmd/process
moon run cmd/sweep
moon run cmd/json
```

`cmd/main` 会打印 V1/V2 各五步的 token、候选数量和反馈状态。完整 CI 复现命令为：

```sh
moon version --all
moon check --target all --deny-warn
moon build --target all --deny-warn
moon test --target all --deny-warn
moon test --release --deny-warn
moon build --target js --release --deny-warn
node scripts/test_hf_bridge.cjs
moon fmt --check
moon info
git diff --exit-code
```

native 构建需要平台 C 编译器及链接器；JS 桥接测试需要 Node.js。
格式和生成接口应使用 CI 的固定工具链更新，避免不同版本造成格式差异。

- `cmd/main`：为每个算法版本打印五次可复现的采样结果。
- `cmd/experiment`：用相同的静态 Zipf 分布和随机种子运行两个版本各 1000 次，并输出 CSV 汇总指标。这是确定性的算法测试样例，不是语言模型质量测试。
- `cmd/generate`：用小型确定性模型样例演示上下文回调和结束 token。
- `cmd/batch`、`cmd/weights`、`cmd/process`：分别演示原子批处理、权重输入和显式 logits 预处理。
- `cmd/sweep`：在同一受控样例上评估多个目标惊讶度。
- `cmd/json`：打印供 WebAssembly 宿主使用的 JSON 数据格式。

典型接入方式是为每个生成会话创建一个 sampler：

```moonbit nocheck
let config = @mirostat.Config::new(3.0, eta=0.1).unwrap()
let sampler = @mirostat.Sampler::new(config, version=@mirostat.Version::v2())
let step = sampler.sample([4.0, 2.0, 1.0], 0.42).unwrap()
println(step.token())
```

传入的随机数必须是有限值且位于 `[0, 1)`。模型 logits 至少要有一个有限值。负无穷表示屏蔽该 token；NaN 和正无穷会报错。token 索引从零开始，并保持输入 logits 的顺序。概率相同时，较小的 token 索引排在前面，因此固定输入和随机数会得到可复现的结果。`reset()` 会恢复初始 `mu` 和步数。`replay()` 接收对齐的 logits 行与随机数数组，`summarize()` 汇总平均惊讶度、候选数量和相对目标值的绝对误差。

若模型后端提供非负权重，可调用 `sample_weights()`，本包会对权重归一化。`preview()` 和 `preview_weights()` 可在不抽样、不改变状态的情况下查看候选数量、保留的模型概率质量和期望惊讶度。`ParkMiller` 为示例提供可移植且可复现的随机数生成器；`sample_with_rng()` 会自动从中取随机数。生产环境也可以继续传入宿主自己的随机数。

`checkpoint()` 和 `restore()` 可用于需要回滚的生成分支；`fork()` 会以相同反馈状态创建独立会话。`replay_atomic()` 在批次中的某一行失败时恢复状态，`replay()` 则保留此前成功的步骤。这两个函数都不管理调用方的随机数来源。

`generate()` 使用生成的 token 历史和可选结束 token 驱动上下文相关的模型回调。若实验中反复使用固定分布，`PreparedRow` 会缓存 softmax 和完整排序，`sample_prepared()` 可重复使用缓存。`TraceAccumulator` 在线计算汇总指标，无需保留整条轨迹。

模型服务在一个推理批次中解码多个请求时，可以调用 `sample_batch()`，传入对齐的 sampler、logits 行和随机数数组。`sample_batch_atomic()` 会在任一行失败时恢复所有 sampler。每个数组位置应对应不同的生成会话。

## 接入宿主运行时

如果宿主负责执行类别抽样，可调用 `candidates()` 获取 token ID、原始概率和归一化后的候选概率。选定 token 后，再调用 `observe_token()` 验证该 token 是否属于当前 Mirostat 候选集，并更新反馈状态。适用于模型运行时已经管理随机数生成器或实现融合采样内核的情况。

`jsonio` 包基于同一套公开 API 提供精简 JSON 协议：`init_json` 创建初始可移植状态；`session_json` 根据下一步 logits 行及之前的 `state_mu`、`state_steps` 处理一个采样步骤；`batch_json` 原子处理多个独立会话；`preview_json` 查看候选集但不消耗随机数；`weight_json` 为返回非负权重的后端提供一次性采样接口。这些函数不执行模型推理，也不访问文件或网络。

## 真实模型接入（CPU）

`scripts/run_hf_mirostat.py` 将 Hugging Face 因果语言模型接入 MoonBit sampler。Transformers 计算每一步的 logits；一个常驻 Node 进程调用编译为 JavaScript 的 MoonBit `jsonio.session_json`；生成器再将选出的 token 传回模型。每个请求使用一个生成会话和一条随机数流。目前此接入示例尚未实现批量解码。

已验证的模型是公开的 `sshleifer/tiny-gpt2`，固定版本为 `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`。Transformers 通过 `LogitsProcessor` 暴露下一 token 的分数，适配器将分数发送给常驻的 Node/MoonBit 桥接进程，再将选出的 token 返回生成流程。这验证了 sampler 可以接入真实的 Hugging Face 生成循环。该微型模型能力有限，不能据此声称生成文本质量良好或大型模型性能优越。

在 Windows 上，可在仓库之外创建独立 Python 环境并下载固定版本的模型（默认路径使用 E 盘）。此配置会复用系统 Python 中的 CPU 版 PyTorch；创建带 `--system-site-packages` 的虚拟环境前，运行机器的系统 Python 中必须已安装 `torch`：

```powershell
py -3.12 -m venv --system-site-packages E:\moonbit-mirostat-venv
E:\moonbit-mirostat-venv\Scripts\python.exe -m pip install -r scripts\requirements-hf.txt
$env:HF_HOME = 'E:\moonbit-mirostat\hf-cache'
E:\moonbit-mirostat-venv\Scripts\python.exe scripts\download_hf_model.py
moon build --target js --release
```

使用 Mirostat V2 生成短文本；也可以将 `--version v2` 改为 `--version v1`：

```powershell
E:\moonbit-mirostat-venv\Scripts\python.exe scripts\run_hf_mirostat.py `
  --prompt 'Explain why stable sorting is useful.' --version v2 `
  --max-new-tokens 48 --seed 7 --report E:\moonbit-mirostat\runs\v2.json
```

同一运行脚本也支持 top-p 基线，例如：

```powershell
E:\moonbit-mirostat-venv\Scripts\python.exe scripts\run_hf_mirostat.py `
  --prompt 'Explain why stable sorting is useful.' --sampler top-p `
  --temperature 0.8 --top-p 0.9 --max-new-tokens 48 --seed 7 `
  --report E:\moonbit-mirostat\runs\top-p.json
```

JSON 报告会记录基线温度和 `--top-p`。比较实验时，应使用相同的模型版本、提示词、随机种子、线程数和 token 上限。报告包含生成文本、token ID、逐步惊讶度和候选数量、每秒生成 token 数，以及 MoonBit 桥接耗时。桥接目前每一步都通过 JSON 发送完整的 float32 logits 行，因此会单独测量这部分开销；这是实验性接入方式，并非低延迟生产适配器。运行器目前仅支持 batch size 为 1，并要求 logits 为有限值。本项目不声称目标惊讶度总能等于观测均值，也不声称 Mirostat 一定提升文本质量。

模型下载脚本会在已下载权重旁记录模型 ID 和精确版本；仓库不包含模型权重。CPU 解码不需要 CUDA GPU 或付费推理 API。实际吞吐量和桥接开销见中文验证记录 [docs/model-integration-validation.md](docs/model-integration-validation.md)，其中也列出了该测试样例的局限。

针对这一接入示例，请分别报告每秒生成 token 数和 MoonBit 桥接的每 token 毫秒数。至少运行三次并比较中位数，每次使用相同的模型版本、提示词、随机种子、线程数和 token 上限。由于 JSON 桥接耗时不等于模型推理耗时，两项指标应分别报告。功能验收条件是每次运行都生成指定数量且 ID 在有效范围内的 token；Mirostat 运行还应为每个生成 token 记录一条轨迹。项目不设与硬件无关的速度门槛，也不把该微型模型作为文本质量测试。

若要排除模型推理开销，单独测试 JS 桥接和 sampler，可先构建 JS 目标，再运行 `node scripts/bench_sampling_bridge.cjs`。测试使用固定的 50,257 个 token 的 logits 行，并报告多轮调用的中位耗时；V2 优化结果和测试限制记录在中文验证文档中。

## 验证范围

测试覆盖稳定 softmax、屏蔽 logits、归一化、候选选择、V1 和 V2 反馈、状态回滚、批处理原子性、JSON 解码、固定种子回放、合成 Zipf 分布漂移，以及 WebAssembly 兼容执行。合成实验用于验证反馈循环在相应测试分布下能否接近可行的目标惊讶度，不代表文本质量、模型质量或特定词表规模下的性能结论。

## 数值与状态行为

Softmax 会先减去最大 logit 再执行指数运算。Zipf 斜率估计器使用前 `m` 个排序概率之间的相邻值；对于平坦分布或下溢分布，会使用较小的正数作为斜率下限。V1 会将估算出的 `k` 限制在词表范围内。V2 至少保留概率最高的 token，即使它的惊讶度超过 `mu`。所选 token 的惊讶度以截断和重新归一化之前的原始 softmax 概率计算。无效 logits 行或随机数会返回错误，且不改变 sampler 状态。在 replay 中，如果后续行失败，之前成功处理的行仍会保留。

V1 使用有界堆检查概率最高的 `m` 个 token，再选择自适应 top-k 候选。V2 扫描整个词表，并仅对低于惊讶度阈值的 token 排序。两种算法每一步仍需检查每个 logit。对于大词表、低延迟解码场景，应使用目标模型和参数对本包进行基准测试。

为便于受控比较，`sample_top_k`、`sample_top_p` 和 `sample_temperature` 使用相同的稳定概率计算和抽样函数。其他对比采样器还包括 min-p、epsilon 和 locally typical sampling。调用方可以在 Mirostat 前应用 `mask_logits`、`bias_logits` 或 `penalize_repetition`；这些变换是显式执行的，因此报告的惊讶度对应变换后的模型分布。`entropy`、`prefix_mass`、`expected_prefix_surprise` 和 `perplexity` 可用于比较分布及观测轨迹。

## 来源与许可证

这是依据 Sourya Basu 等人的论文 *Mirostat: A Neural Text Decoding Algorithm That Directly Controls Perplexity*（ICLR 2021）独立实现的 MoonBit 项目。作者的 [Python 研究代码](https://github.com/basusourya/mirostat)用于参考术语和算法范围；本仓库未复制其源文件。上游代码采用 MIT 许可证。本 MoonBit 项目采用 Apache-2.0 许可证，详见 [LICENSE](LICENSE)。
