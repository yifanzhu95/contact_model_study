# Kamino on Polaris: full G1–G3 campaign / 完整 G1–G3 实验

The current operator entry is `campaign.py submit`. It submits **one PBS job** for the original G1 calibration, G2 parallel comparison and G3 parameter grid. There is no separate G0 job, inter-stage submission, login-node watcher or fixed-topology substitute. Changes are isolated on `codex/polaris-readiness`, based on integrated commit `a8dcc873e9f4e4f13dcf9973b30ed72f9d3bfa70`; shared experiment branches are not changed.

当前入口是 `campaign.py submit`，只提交**一个 PBS 作业**，完整执行原 G1 校准、G2 并行对比和 G3 参数组合。不另提 G0，不在阶段间提交作业，不依赖登录节点监控，也不以单一配置代替完整实验。改动隔离在 `codex/polaris-readiness`，基于集成提交 `a8dcc873e9f4e4f13dcf9973b30ed72f9d3bfa70`；共享实验分支不变。

## Workload / 实验范围

| Stage | Configurations | Episodes | Maximum control plans |
| --- | --- | ---: | ---: |
| G1 | N=32/64/128/256 × seeds 0/1/2; one GPU, ≤20 steps | 12 | 240 |
| G2 | N=64; 1 GPU/4 workers, 4 GPUs/4, 8, 16 workers; 24 paired episodes per topology, ≤100 steps | 96 | 9,600 |
| G3 | N=64; temperature 10/40/100 × sigma 0.02/0.05/0.10 × seeds 0/1/2; ≤100 steps | 27 | 2,700 |
| Total | 43 cells | 135 | 12,540 |

G1 runs sequentially on GPU 0. All four G2 topologies run sequentially, using the same pooled driver and deterministic per-episode goal/noise seeds; no other campaign workload competes with them. G3 runs independent sequential drivers, at most one per GPU and four at once. G1 and G3 share the sequential driver's seed convention; G2 uses the pooled convention. N=64 is a reviewed starting budget, not an optimum or an automatically selected calibration result. No cell or N level is silently dropped.

G1 在 GPU 0 上顺序执行。G2 的四种拓扑逐一执行，均使用进程池入口和相同的 episode 目标／噪声种子，不与本批其他实验争抢资源。G3 运行独立实验进程，每卡同时最多一组、总共最多四组。G1 与 G3 使用顺序入口的种子规则，G2 使用进程池规则。N=64 是已选定的起始预算，不是最优值，也不是自动校准选择。不会暗中省略配置或 N 档位。

All stages retain the cube scenes, cost weights, one MPPI iteration, CUDA graphs, no warm start, no video and saved numeric steps. The control interval is 64 ms, rollout timestep 4 ms, and effective horizon five control intervals (320 ms). PADMM uses the original cap **100** and tolerance **5e-4**, uniformly. The earlier reduced direct pilot's cap400 is not used. Experiments may finish early when the object drops or planning fails.

各阶段保留方块场景、代价权重、一次 MPPI 迭代、CUDA graph、不热启动、不录视频并保存数值步骤。控制间隔 64 ms，rollout 时间步 4 ms，有效预测长度为五个控制间隔（320 ms）。PADMM 统一使用原始上限 **100** 和容差 **5e-4**，不用之前单配置入口的 400。物体掉落或规划失败时，episode 可能提前结束。

## Operator: prepare once, submit once / 操作者：一次准备、一次提交

Use a **new source directory**, detached at the full deployment SHA supplied in the handoff. Do not modify any active checkout, previous failed output, or historical submission ledger. Reuse the already prepared Python3.12 environment with MuJoCo3.6.0, Newton1.6.0 and Warp1.17.0. No new dependencies were added. Replace the four example values below as needed; these are runtime settings, not source edits. Never share login credentials.

在**新目录**克隆源码，并固定到交付消息提供的完整 SHA。不修改正在使用的代码目录、旧失败结果或历史提交登记。复用已准备的 Python3.12 环境：MuJoCo3.6.0、Newton1.6.0、Warp1.17.0。本次未增加依赖。根据实际情况替换下方四项示例值；它们是运行配置，不是修改源码。不要分享登录凭证。

```bash
git clone --single-branch --branch codex/polaris-readiness \
  https://github.com/yifanzhu95/contact_model_study.git \
  /home/<user>/contact_model_study_polaris_full
git -C /home/<user>/contact_model_study_polaris_full switch --detach <deployment-SHA>

PROJ_DIR=/home/<user>/contact_model_study_polaris_full
ENV_DIR=/home/<user>/envs/contact-kamino-polaris
ALCF_PROJECT=FoundationModels4RobotManipulation
OUT_ROOT=/eagle/FoundationModels4RobotManipulation/<user>/kamino_polaris

"$ENV_DIR/bin/python" "$PROJ_DIR/experiments/polaris/campaign.py" submit \
  --project "$ALCF_PROJECT" --env "$ENV_DIR" --out-root "$OUT_ROOT" \
  --queue capacity --walltime 24:00:00
```

The account/project must already exist and permit this operator to charge it; `--project` is passed to PBS as `qsub -A`. The environment and writable Eagle output root must already exist. Use absolute paths without whitespace, commas, quotes or backslashes. A clean pinned source is mandatory for submission. If the environment is not ready, the authorized operator first obtains a compute node and runs `bash experiments/polaris/setup_environment.sh <absolute-repo> <absolute-env>`; do not install or run simulations on a login node.

项目额度必须已经存在，且允许操作者使用；`--project` 会以 `qsub -A` 传给 PBS。环境目录和可写的 Eagle 结果根目录应已存在。使用不含空白、逗号、引号、反斜杠的绝对路径，提交时源码须固定且干净。如果环境尚未准备，获授权操作者先取得计算节点，再运行 `bash experiments/polaris/setup_environment.sh <源码绝对路径> <环境绝对路径>`；不在登录节点安装或运行仿真。

The [official Polaris queue guide](https://docs.alcf.anl.gov/polaris/running-jobs/) allows 1–4 nodes in `capacity`, with a documented maximum of 168 hours. This campaign requests **one node, up to24 hours**; this is a bounded budget, not a measured completion-time guarantee. It exits when finished and reserves ten minutes for startup/finalization. The account holder must confirm queue access and available allocation; the software cannot grant them. `debug` supports at most one hour and is not the recommended full-campaign queue. There is no automatic queue switch, resubmission or extra allocation.

[Polaris 官方队列说明](https://docs.alcf.anl.gov/polaris/running-jobs/)中 `capacity` 允许 1–4 个节点，最长 168 小时。本实验申请**一个节点、最多24小时**；这是有限预算，不是实测完成时间保证。完成后即退出，并预留十分钟用于启动／封存。账户持有人需确认队列权限和可用额度，软件不能授予这些权限。`debug` 最多一小时，不建议用于完整实验。不自动换队列、重提或追加额度。

The submitter first requests one held, non-rerunnable job, saves its ID and plan hash, then releases it. Only after **`Submitted and released ONE job`** appears is normal execution independent of SSH. If submission/release fails or the response is unknown, keep the printed output directory and send its `submission.json`; do not blindly rerun or hand-edit the ledger. This protects against duplicate jobs, not against ordinary quality warnings.

提交入口先申请一个挂起、不自动重跑的作业，保存编号和计划哈希，再放行。仅当出现 **`Submitted and released ONE job`** 后，正常执行才不依赖 SSH。提交／放行失败或返回不明时，保留打印出的目录并提供 `submission.json`，不要反复重提或手改登记。这是在防止重复作业，不是在拦截普通数值告警。

Inside that allocation, the wrapper checks source, package versions, four A100s, scene hashes, PBS identity and Eagle writing; records the environment; executes all G1 → all G2 → all G3; and seals artifacts. GPU roles are assigned explicitly, without MPI. Baseline/calibration phases intentionally leave other GPUs idle so that the comparisons remain interpretable.

在同一个作业内，程序检查源码、依赖版本、四张 A100、场景哈希、PBS 身份及 Eagle 写入；记录环境；自动执行全部 G1 → 全部 G2 → 全部 G3；最后封存结果。GPU 任务明确分配，不用 MPI。基线／校准阶段有意让其他 GPU 空闲，以免干扰比较。

For a read-only argument/matrix preview, append `--check` to the submission command. It does not create directories, query queue eligibility, run CUDA, or submit jobs. It is not proof that the campaign will finish within24 hours.

在提交命令末尾加 `--check` 可只读预览参数和矩阵。它不建目录、不检查账户队列资格、不运行 CUDA，也不提交作业；不能证明实验可在24小时内完成。

## Numerical policy: explicit exploration / 数值策略：明确的探索模式

The campaign explicitly uses `--kamino-acceptance exploratory` in **every stage**. Finite residual overshoot, nonconvergence and iteration-limit hits are quality warnings and no longer manufacture NaN costs that remove those candidates. Reaching contact capacity is a warning, not proof of overflow. The original strict checks and per-plan diagnostics remain recorded; strict qualification is independent of whether an experiment runs to completion. This acceptance-policy change may change the selected actions; do not pool these results with old strict-filter runs as one cohort.

本批**所有阶段**显式使用 `--kamino-acceptance exploratory`。有限的残差超标、未收敛、触及迭代上限会记为质量告警，不再人为把相应候选代价改成 NaN 并剔除。接触数达到容量是告警，不等于证明溢出。原严格检查和逐次规划诊断仍然记录；严格资格与实验是否执行完成分别判定。接受策略的改变可能改变选出的动作，不能与旧严格过滤实验混为同一批统计。

Actual nonfinite raw state/cost/residual, failed reset and definite contact-buffer overflow remain hard candidate rejection conditions. Fine-step residual/state checks are accumulated on-device and compatible with graph capture. If all candidates are unusable, the episode ends as `planner_failure` **before** applying the fallback zero absolute joint target, and the rejected attempt is saved separately. Other episodes continue. The model default outside this explicit campaign remains `strict`.

真实的原始状态／代价／残差 NaN 或 Inf、重置失败、明确的接触缓冲区溢出仍会使候选被拒绝。细时间步的残差／状态检查在设备上累积，与 graph capture 兼容。若所有候选均不可用，episode 记为 `planner_failure`，在执行错误的零绝对关节目标**之前**结束，并单独保存被拒规划。其他 episode 继续。本批以外的模型默认仍为 `strict`。

Task failures, quality warnings, and isolated cell errors/OOM are recorded without stopping other planned cells or shrinking N. Recognized global environment/source/filesystem/driver failures stop the campaign. Missing utilization samples are measurement warnings; they do not kill experiments. Unknown runtime errors are preserved as cell errors rather than silently repaired. There is no automatic retry.

任务失败、质量告警和孤立配置报错／显存不足会被记录，不阻止其他配置，也不缩小 N。已识别的全局环境／源码／文件系统／驱动故障会停止整批。利用率采样缺失只是测量告警，不终止实验。未知运行错误保留为配置报错，不暗中修复。不自动重试。

## Results and fair performance reporting / 结果和公平性能统计

`status.json` separates execution completion, task outcomes, strict numerical qualification and performance claims. It includes all stage verdicts, G1's N groups and G2 comparisons. Each cell stores source/policy identity, requested/completed work, first-plan timing, later planning means, CPU request waiting, per-planner initialization time (pooled), GPU telemetry, diagnostics and numeric arrays. Initialization timing excludes neither all Python import time nor all lazy CUDA compilation: first-plan and full-process wall times are also retained, not relabeled as pure steady-state compute.

`status.json` 分别报告执行完成、任务结果、严格数值资格和性能结论，包含各阶段判定、G1 的 N 分组和 G2 对比。各配置保存源码／策略身份、请求和完成工作量、首个规划耗时、后续规划平均耗时、CPU 请求等待、进程池中各规划器初始化耗时、GPU 采样、诊断及数值数组。初始化计时并未覆盖全部 Python 导入或延迟 CUDA 编译，因此同时保留首规划及整个进程耗时，不冒充纯稳态计算时间。

A descriptive exploratory wall-time ratio is shown only for matching full work in the same allocation, with observed requested GPUs and complete telemetry. A qualified speedup additionally requires strict numerical qualification. Early drop, planner failure, unequal step counts or mixed resumed allocations never produce a speedup. Wall times and completed-plan counts remain visible even when ratios are ineligible. Three G3 seeds per setting are a pilot, not a strong statistical success-rate claim.

仅当同一作业内完成匹配的全部工作、观察到所需 GPU 且遥测完整时，才显示探索性的描述耗时比；经过严格资格的加速比还要求数值检查通过。提前掉落、规划失败、步数不等或混合续跑节点不能产生加速比。即使不适合计算比值，耗时和实际规划数仍保留。G3 每组仅三个种子是试点，不支持强统计成功率结论。

## Budget and explicit resume / 时间预算及显式续跑

A time limit or interrupt stops active children, retains completed cells and partial attempts, and marks the remainder uncompleted. A hard PBS kill/node loss may prevent final saving or sealing; no software can guarantee otherwise. No automatic new job is launched. To continue later, use the same source/environment and one command below, replacing `<previous-campaign-root>` with the earlier printed path.

达到时间预算或收到中断时，停止活动子进程，保留已完成配置及部分尝试，并将剩余标为未完成。PBS 强制杀进程或节点丢失可能使最终保存／封存来不及完成，软件无法保证例外。不自动创建新作业。后续若要续跑，使用相同源码／环境和下方一个命令，将 `<previous-campaign-root>` 替换为先前打印的目录。

```bash
"$ENV_DIR/bin/python" "$PROJ_DIR/experiments/polaris/campaign.py" submit \
  --project "$ALCF_PROJECT" --env "$ENV_DIR" --out-root "$OUT_ROOT" \
  --queue capacity --walltime 24:00:00 \
  --resume-from <previous-campaign-root>
```

Resume automatically asks PBS to confirm the prior job is finished, checks the identical source/matrix/environment snapshots, and copies hash-verified completed cells and all partial historical attempts into a **fresh** output directory. Completed cells with warnings or recorded task/planner failures are reused, not retried. Incomplete/error cells restart as whole cells in a new attempt; partial episodes are not mixed into completed benchmarks. Old evidence is never overwritten. A completed cell with corrupt/missing evidence blocks resume rather than being silently rerun. If PBS cannot confirm the old job is finished, resume refuses rather than risk overlapping writers. G2 ratios involving reused cells remain unqualified as cross-allocation comparisons.

续跑自动询问 PBS 以确认旧作业已结束，核对相同源码／矩阵／环境快照，将经过哈希复核的已完成配置和所有历史部分尝试复制到**新**结果目录。带告警、记录任务失败或规划失败的已完成配置会复用，不重跑。未完成／报错配置以新尝试整组重新开始，部分 episode 不混入完整基准。旧证据不覆盖。已完成配置的证据损坏／缺失时会拒绝续跑，不暗中重跑。PBS 无法确认旧作业结束时也拒绝续跑，以免并发写入。涉及复用配置的 G2 比值不作为同节点加速结论。

Return the **entire printed campaign directory**, not only a screenshot. It contains `plan.json`, `submission.json`, `submission_at_start.json`, `status.json`, preflight/environment records, logs, `g1/`, `g2/`, `g3/`, per-attempt outcomes/arrays and `artifact_manifest.json`. The seal excludes the login-owned live ledger and PBS stdout/stderr, which PBS may stage later. To inspect saved progress without submitting anything:

请返回**打印出的整个结果目录**，不只是截图。目录包含计划、提交记录、启动时提交快照、总状态、预检／环境记录、日志、G1／G2／G3 子目录、各次尝试的结果／数组及产物哈希清单。封存排除登录进程负责的实时登记和 PBS 稍后可能写入的标准输出／错误文件。如需只读查看已保存进度：

```bash
"$ENV_DIR/bin/python" "$PROJ_DIR/experiments/polaris/campaign.py" status --root <campaign-root>
```

## Package map / 文件对应关系

```text
experiments/polaris/
├── campaign.py             # one-submit full-matrix controller / 完整实验控制器
├── run_campaign.pbs        # one Polaris allocation and final seal / 单作业及封存
├── g1.csv, g3.csv          # original calibration and parameter rows / 原校准及参数行
├── m5_smoke.csv            # shared scene/control defaults / 公共场景和控制设置
├── run_stage.py            # reused matrix/command helpers; old strict CLI / 矩阵工具及旧严格入口
├── direct.py               # reused single-job journal/finalizer; reduced pilot CLI / 提交和封存工具及旧单配置入口
├── preflight.py            # compute-node source/environment/device checks / 节点预检
├── collect_manifest.py     # final artifact hashes / 结果哈希
├── setup_environment.sh    # one-time environment setup and CPU checks / 一次性安装
└── requirements-m5.txt     # unchanged pinned dependencies / 不变的依赖
ContactModelStudy/
├── Simulators/Kamino.py, _KaminoDiagnostics.py  # explicit policy plus raw diagnostics / 接受策略及原始诊断
├── Drivers/run_episodes.py, EpisodePool.py     # safe episode exit, seeds and timings / 安全结束、种子和计时
└── Utils/EpisodeRecorder.py                   # separate actions, rejection and quality / 动作、拒绝与质量记录
tests/test_polaris_campaign.py, test_kamino_diagnostics.py  # CPU/fake-PBS checks / 离线测试
```

Legacy `pipeline.py`, separate-stage PBS scripts and `direct.py submit` remain for historical/reference use, **not this handoff**. The former submits four held jobs with strict G0 gates; the latter runs only one fixed topology. Do not submit either alongside `campaign.py`, and do not edit their historical ledgers. Strict reference qualification remains separately selectable, but its old multi-job submitter still has the observed queue-limit incompatibility.

旧 `pipeline.py`、分阶段 PBS 脚本及 `direct.py submit` 保留用于历史／参考用途，**不是此次交付入口**。前者提交四个挂起作业并要求严格 G0 通过，后者只跑单一拓扑。不要与 `campaign.py` 同时提交，也不要编辑历史登记。严格参考资格仍可单独选择，但旧多作业入口仍存在实测队列限制的不兼容。

## Verification boundary / 验证边界

CPU tests and fake PBS tests check matrix coverage, single submission, warning continuation, hard candidate rejection, interruption, immutable artifacts and resume. They do not establish A100 convergence, real queue eligibility, task success or completion within24 hours. The previous Polaris run tested installation and GPUs but failed strict qualification; it is preserved as historical evidence, not a result of this new campaign.

CPU 和模拟 PBS 测试检查矩阵覆盖、单次提交、告警继续、硬候选拒绝、中断、结果保全及续跑。它们不能证明 A100 数值收敛、真实队列资格、任务成功或24小时内完成。之前 Polaris 运行验证了安装和 GPU，但未通过严格数值资格；其证据保留，不冒充新实验结果。

The M5 deployment environment intentionally omits the M1–M4-only `comfree_warp` dependency. An additional M4 preset test therefore cannot run in this environment; the same missing-dependency failure was reproduced on the unmodified preceding commit. It is not evidence of an M5 regression or a reason to install unrelated GPU backends for this campaign.

M5 部署环境有意不安装 M1–M4 才需要的 `comfree_warp` 依赖，因此额外的 M4 预设测试无法在该环境运行；在未修改的上一提交上也复现了相同缺依赖错误。这不是 M5 回归的证据，也不需要为本批安装无关的 GPU 后端。
