# 远端路径与项目上下文

理解共享盘作用域、存储池、挂载隔离、path alias 和仓库 `INSPIRE.md` 时先看本页。调度条件和资源规格看 [resources-and-paths.md](resources-and-paths.md)；公网 / 内部源看 [network-and-sources.md](network-and-sources.md)。

## 1. 路径不是调度条件

Path alias 是远端路径 alias，不是 workload profile。它只回答“文件在哪里”，不回答“任务在哪里跑”。

| 概念 | 保存什么 | 不能替代 |
| --- | --- | --- |
| Workload profile | workspace、project、group、quota、image | 远端目录 |
| Path alias | 共享盘上的目录 | workspace、project、group、quota、image |
| `INSPIRE.md` | 人类可读项目约定 | CLI 配置和账号 secret |

账号级 `[path_aliases]` 提供默认远端路径；仓库级、账号隔离的 `[path_aliases]` 可以覆盖当前 repo 的同名 alias。不要维护单独的“远端工作目录”字段。

## 2. 路径作用域

| 作用域 | 路径形状 | 定位 |
| --- | --- | --- |
| 项目个人 | `/inspire/<tier>/project/<topic>/<path-user>/...` | 每项目、每账号一份，适合代码、脚本、调试输出 |
| 项目公共 | `/inspire/<tier>/project/<topic>/public/...` | 项目成员共享，适合数据集、权重、批量结果、checkpoint |
| 全局个人 | `/inspire/<tier>/global_user/<path-user>/...` | 跨项目个人盘，适合小工具和中转 |
| 全局公共 | `/inspire/hdd/global_public/...` | 全平台共享，普通账号通常只读 |

`<path-user>` 是共享盘返回的个人目录名，不一定等于登录 ID。路径事实以文件页 Browser API 发现结果为准。

## 3. 存储池

| 池 | 项目路径前缀 | 定位 |
| --- | --- | --- |
| SSD `gpfs_flash` | `/inspire/ssd/project/<topic>/` | 训练 hot path、活跃工作集、checkpoint 热点 |
| HDD `gpfs_hdd` | `/inspire/hdd/project/<topic>/` | 通用空间，写前看剩余容量 |
| qb-ilm `qb_prod_ipfs01` | `/inspire/qb-ilm/project/<topic>/` | 大容量，顺序读带宽接近 SSD |
| qb-ilm2 `qb_prod_ipfs02` | `/inspire/qb-ilm2/project/<topic>/` | 新且空余多，新增大数据默认优先考虑 |

`global_public` 只在 hdd。需要 SSD 或 qb-ilm 速度时，优先走项目个人或项目公共路径。

## 4. 挂载隔离

实例只挂自身所在项目的 fileset。其它项目的 `/inspire/{hdd,ssd,qb-ilm,qb-ilm2}/project/<others>/` 在该实例里通常不存在，`ls` 报 `No such file` 不是权限问题。

跨项目搬小文件时，在两个项目各起一个 notebook，用全局个人路径中转。大数据集或全量 checkpoint 超出个人 quota 时，联系项目管理员处理。

## 5. Alias 语义

默认 alias 由 `inspire init` 写入账号配置；当前 repo 需要覆盖时，用项目初始化或 path alias 命令写仓库级配置。

| Alias | 指向 |
| --- | --- |
| `me` | 当前项目、当前账号、初始化时选择的默认存储池 |
| `public` | 当前项目公共目录、初始化时选择的默认存储池 |
| `global-me` | 当前账号全局目录 |
| `<tier>.me` | 指定存储池下的项目个人目录，例如 `ssd.me`、`hdd.me`、`qb-ilm2.me` |
| `<tier>.public` | 指定存储池下的项目公共目录 |
| `<tier>.global-me` | 指定存储池下的全局个人目录 |

路径参数支持 `alias`、`alias:<subdir>` 和命名 alias 三种心智模型。具体命令语法看 help；reference 只强调语义：把代码、数据、权重和产物放到远端路径约定里，不把路径含义塞进 workload profile。

## 6. `INSPIRE.md` 必须维护

每个启智项目仓库都必须维护根目录 `INSPIRE.md`。它是项目级启智上下文，不是可有可无的建议文件。

适合写入：

- `Default Image`
- `Workload Profiles`
- `Path Conventions`
- `Public Directory Layout`
- `Existing Notebooks`
- `Ongoing Jobs`
- 特殊 workspace、特殊 compute group 或组内资源边界

不应写入：

- 账号配置、密码、代理密钥或平台 session。
- `.inspire/config.toml` 或 `.inspire/accounts/<account>/config.toml` 的具体内容。
- 只属于某个 harness 的本地 Agent 计划。
- 临时排查流水账、旧命令列表或已过期 release 状态。

`AGENTS.md` / `CLAUDE.md` 等文件可以记录本地 Agent 计划和执行风格；启智上下文必须集中在 `INSPIRE.md`，方便不同 harness、不同成员和未来 Agent 共用同一份项目事实。
