# 项目工作流

一个项目跨越 CPU 准备、数据处理、GPU 训练、部署或交付时看本页。这里给阶段化决策和验收点，不维护命令模板；具体命令表面回到 CLI help，单领域边界分别看 resources、network、paths、notebook、image、compute workloads 和 model references。

## 1. 总体框架

把项目推进拆成三段：

| 阶段 | 目标 | 典型平台位置 | 验收 |
| --- | --- | --- | --- |
| A. 联网准备 | 下载代码、依赖、数据、权重，形成共享盘布局或镜像 | `CPU资源空间` 的可上网 CPU notebook | repo 可更新，依赖可导入，数据 / 权重路径可读，镜像 `READY` |
| B. CPU 处理 | 预处理、清洗、评测、打包、索引构建 | HPC，必要时 Ray | 小规模 probe 通过，正式规模产物完整，有 fingerprint |
| C. GPU 训练 / 部署 | 单节点调试、多节点训练、serving | `分布式训练空间` 的 GPU notebook、job、serving | 日志推进，metrics 有负载，产物 / 服务 smoke 通过 |

核心原则：公网和依赖准备前置；目标 GPU 空间只负责读共享盘、拉已准备镜像、运行目标程序。特殊 workspace 只有在硬件、权限或项目环境明确要求时才进入计划。

## 2. 阶段 A：准备

准备阶段要产出两类稳定结果：

- 共享盘布局：代码、数据、权重、checkpoint、输出目录、fingerprint 约定。
- 可复用镜像：项目依赖、系统依赖、Slurm / Ray runtime、服务 runtime。

验收不要只看命令返回成功。至少确认：

- 远端 repo 能更新到目标 commit。
- 关键 Python / system 依赖能在目标镜像里导入或执行。
- 数据和权重路径在目标项目共享盘下可读。
- 后续 workload 要复用的镜像状态为 `READY`。
- `INSPIRE.md` 已记录 path conventions、默认镜像和相关 workload profile 名称。

## 3. 阶段 B：CPU 处理

固定规模 CPU 批处理优先 HPC；流式、长守护或异构 worker 才考虑 Ray。

| 形态 | HPC | Ray |
| --- | --- | --- |
| 任务边界 | 明确开始和结束 | 长时间流式或服务型 |
| 并发模型 | 固定 task / instance | worker min / max 弹性伸缩 |
| 数据流 | GPFS 读写 | Ray 对象存储 + GPFS |
| 结束条件 | Slurm 程序退出 | Driver 退出 |

正式放量前先跑接近生产形状的 probe。小规模通过不代表正式规模稳定；正式处理要写 fingerprint，并用同项目 notebook 回读产物目录确认数量、大小和内容摘要。

## 4. 阶段 C：GPU 训练

进入训练前应已经具备：

- 代码在共享盘 repo 中，或镜像内已包含固定代码。
- 数据和权重在目标项目共享路径可见。
- 依赖在镜像中，或目标环境无需公网安装。
- 目标 GPU quota 和 compute group 实时可用。
- 单节点 probe 已验证 CUDA、NCCL、数据路径和入口脚本。

正式多节点训练看三条线：events 判断调度，logs 判断程序，metrics 判断资源是否真的工作。多节点中某个 pod 长期低 GPU / 低网络时，不要只盯 `RUNNING`，回到该 worker 日志和数据加载路径。

## 5. 部署或交付

模型服务先把模型目录注册到 model registry，再创建 serving。Serving 验收不止看状态：

- 服务实例齐全，没有长期 Pending。
- metrics 显示请求或模型加载后的资源曲线合理。
- `/health`、`/v1/models` 或业务 smoke test 通过。
- 无 key 请求被拒绝，带 key 请求成功。
- 模型版本、镜像版本、启动命令和端口写入 `INSPIRE.md` 或项目交付记录。

非服务型交付则回到共享盘产物：目录结构、fingerprint、大小、样例文件和下游读取 smoke 都要确认。
