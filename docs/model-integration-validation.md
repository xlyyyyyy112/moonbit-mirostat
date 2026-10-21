# Hugging Face 模型接入验证记录

验证日期：2026-10-21

## 验证目的

检查 MoonBit Mirostat 能否接入真实模型的逐 token 生成循环：Transformers 计算下一 token 的 logits，`LogitsProcessor` 将 logits 交给 Node/MoonBit 桥接，采样得到的 token 再送回模型继续生成。本次验证针对接口连通性和开销，不用于证明生成质量。

## 环境与输入

- 系统：Windows 11，CPU 推理，PyTorch 2.10.0+cpu，2 个 Torch 线程
- Transformers：4.57.1
- 模型：[`sshleifer/tiny-gpt2`](https://huggingface.co/sshleifer/tiny-gpt2)，固定 revision `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`
- 权重文件：2,514,146 字节；SHA-256：`b706b24034032bdfe765ded5ab6403d201d295a995b790cb24c74becca5c04e6`
- 提示词：`The capital of France is`；5 个输入 token
- 每次生成 16 个 token，随机种子 7；Mirostat 使用 `tau=4`、`eta=0.1`；top-p 使用 temperature `0.8`、`top_p=0.9`
- 每种采样方式重复运行 3 次；模型文件与 JSON 报告保存在仓库外的 `E:\moonbit-mirostat`

该模型体积很小，适合 CPU 集成测试，但不是有代表性的文本生成模型。它没有 chat template，因此使用普通文本提示词。
结果表记录的是随后热路径优化之前的 `tau=4` 接入基线；优化后的 V2 全词表复测单列在文末。

## 结果

| 采样方式 | 每次生成 token 数 | 吞吐量（3 次，token/s） | 中位吞吐量 | MoonBit 桥接耗时中位数（ms/token） | 平均 surprise |
| --- | ---: | --- | ---: | ---: | ---: |
| Mirostat V1 | 16/16 | 10.00、8.69、9.20 | 9.20 | 99.05 | 15.392 bits |
| Mirostat V2 | 16/16 | 9.75、9.45、9.15 | 9.45 | 96.63 | 15.392 bits |
| top-p 0.9 | 16/16 | 52.50、45.78、50.25 | 50.25 | 不经过 MoonBit 桥接 | — |

三个方案都使用相同模型版本、提示词、种子、线程数和生成长度。Mirostat 的 16 个 token 都有对应的采样轨迹；Python 端检查了每个返回 token ID 均处于模型词表范围内。三次重复结果一致：当前 tiny-gpt2 在 `tau=4` 下只保留 1 个候选，输出重复 `stairs`。top-p 输出了不同 token，但句子仍不连贯。

## 结论与验收口径

本次证明公开的 Hugging Face Transformers `LogitsProcessor` 接口可以提供模型 logits，项目能够完成“模型 logits → MoonBit 采样 → token 回灌模型”的真实生成闭环；不依赖模型厂商合作、付费 API 或 GPU。项目没有获得模型厂商背书，也不把这次小模型接入描述成生产部署。

本验证的功能验收条件为：使用固定模型 revision；每次生成达到指定 token 数；返回 token ID 不越界；Mirostat 每个生成 token 都产生一条状态/采样记录；报告保存模型与运行时版本、吞吐和桥接耗时。性能指标报告总生成吞吐量与桥接毫秒/token，使用至少三次相同配置运行的中位数。速度不设跨硬件的绝对门槛；当前 JSON 桥接明显慢于不经过桥接的 top-p 基线，因此不声称本实现提高了推理速度。

## 限制

tiny-gpt2 太小，不能用来判断 Mirostat 的实际语言质量或较大词表下的吞吐表现。当前 `tau=4` 产生单候选重复输出，说明该设置在这个模型上不合适；这不是成功的质量案例。以上数据只用于确认集成路径、功能返回和当前桥接开销。模型接入的复现命令见项目 README，下载器默认使用同一固定模型版本。

## 根据实验结果做的 V2 热路径优化

前面的真实模型运行显示 V2 在候选集较大时会排序整个候选列表；检查代码还发现，内部已生成的候选索引会再次进入面向外部输入的校验函数，该函数会额外分配与词表等长的去重数组。内部索引由采样器自己生成，重复校验没有必要。

现在 V2 通过一次 `2^-mu` 阈值计算筛选候选，并在内部抽样时按 token ID 顺序直接抽取，不再对内部候选排序，也不再创建去重数组。公开 `candidates()` 返回的列表仍按概率降序排列；公开 `draw_candidates()` 仍执行完整输入校验。由于抽样遍历顺序改变，相同 uniform 在新旧版本间可能选中不同 token，但候选集合和抽样概率不变，同一版本内仍可复现。

使用 Node v24.18.0 和 50,257 个相同 logits 的固定 Node/JS 桥接微基准。每次脚本先预热，再测 5 组、每组 24 次；分别在旧版和优化版上重复启动基准进程，比较每次进程中位数的中位数：

| 版本 | 优化前（ms/调用） | 优化后（ms/调用） | 变化 |
| --- | ---: | ---: | ---: |
| V1 | 22.720（3 次） | 22.059（4 次） | 基本不变，属于运行波动 |
| V2 | 13.441（3 次） | 10.882（4 次） | 下降约 19% |

这组微基准包含 JSON 解析和 MoonBit 采样，不包含模型推理；输入是 50,257 个相同 logits，不代表所有真实模型的分布。优化前后的代码来自同一仓库的当前版本和父提交。复现方式：

```powershell
moon build --target js --release --deny-warn
node scripts/bench_sampling_bridge.cjs
```

优化后另用 tiny-gpt2、`tau=15`、相同提示词/种子/线程数生成 16 个 token，重复三次：每次保留全部 50,257 个候选，平均 surprise 为 15.613 bits；总吞吐中位数 10.15 token/s，桥接耗时中位数 90.40 ms/token。输出仍不连贯，因此这只是检查全词表候选路径的端到端复测，不能当作文本质量提升证据。此前 tau=15 仅有单次运行结果，端到端前后对比不作为严格性能结论。
