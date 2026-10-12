# Kamino deployment on Polaris / Kamino 的 Polaris 部署

This package implements G0 correctness checks, G1 single-GPU calibration, G2 four-GPU throughput evaluation, G3 a small research pilot, and a single-launch PBS job chain. Local implementation does not establish A100 compatibility; GPU execution and Polaris measurements are still pending.

本部署包已实现 G0 正确性检查、G1 单卡负载校准、G2 四卡吞吐量评估、G3 小规模科研试验，以及一次启动的 PBS 作业链。本地实现完成仍需在 A100 验证，GPU 执行和 Polaris 实测结果尚待取得。

The deployment starts with correctness checks and measured calibration before expanding to multi-GPU experiments and parameter searches. The budgets below define the initial pilot, not hardware-validated optimal settings. Deployment changes are isolated on `codex/polaris-readiness`, based on integrated commit `a8dcc873e9f4e4f13dcf9973b30ed72f9d3bfa70`; publishing this branch does not merge it into shared experiment branches.

部署首先进行正确性检查和实测校准，再扩大到多 GPU 实验和参数搜索。下方规模定义首轮试点，不是经过硬件验证的最优设置。部署改动隔离在 `codex/polaris-readiness`，基于集成提交 `a8dcc873e9f4e4f13dcf9973b30ed72f9d3bfa70`；发布该分支不会将它合并到共享实验分支。

## Hardware and parallel work / 硬件与并行方式

A Polaris node has one 32-core CPU and four NVIDIA A100 GPUs. In our proposed configuration, the CPU runs Python, schedules work, and advances MuJoCo evaluation episodes; GPUs run batched Kamino predictions used by MPPI. Requesting a node does not automatically make one Python process use four GPUs. See the [ALCF hardware overview](https://docs.alcf.anl.gov/polaris/).

Polaris 一个节点包含一颗 32 核 CPU 和四张 NVIDIA A100。本方案中，CPU 执行 Python、安排工作，并推进 MuJoCo 评估实验；GPU 执行 MPPI 所需的批量 Kamino 预测。申请节点不会自动让一个 Python 程序用上四张卡。硬件信息见 [ALCF 官方介绍](https://docs.alcf.anl.gov/polaris/)。

GPU work can grow in two ways. Increasing N gives one plan more candidate action sequences: N=256 means predicting 256 possible futures. Running independent episodes or parameter settings on different GPUs increases experiment throughput. Four GPUs with N=256 each give four independent planners, not one N=1024 planner.

增加 GPU 工作量有两种方式。增加 N，让一次规划包含更多候选动作序列：N=256 表示预测 256 种未来。让不同 GPU 同时运行独立 episode 或参数配置，则增加实验吞吐量。四张卡各运行 N=256，得到四个独立规划器，并不等于一个 N=1024 的规划器。

The integrated code provides `run_episodes_pooled.py` and `EpisodePool.py`. Queues connect GPU planners and CPU evaluation workers. Start with one planner process per GPU, then measure whether more CPU workers help. Multiple planners on one GPU may increase contention and memory use.

集成代码提供 `run_episodes_pooled.py` 和 `EpisodePool.py`，通过队列连接 GPU 规划进程和 CPU 评估进程。先给每张卡一个规划进程，再测量增加 CPU 工作进程是否有帮助。同一卡增加规划进程可能导致争抢和显存增加，收益需要实测。

## G0: correctness check / 正确性检查

The implemented G0 uses one visible A100, one cube episode, ten closed-loop plans, four candidates, one MPPI iteration, CUDA graphs, and no video. It checks state conversion, solver diagnostics, graph/eager agreement, source/environment identity, and result writing. Later stages use the remaining cards.

已实现的 G0 使用一张可见 A100、一个方块 episode、十次闭环规划、四个候选、一次 MPPI 迭代，启用 CUDA graph，不生成视频。它检查状态转换、求解器诊断、graph 与普通执行的一致性、代码/环境版本和结果保存。后续阶段使用其余 GPU。

Every expected Kamino fine step must converge. Reset failure, excessive residuals, iteration/contact-capacity limits, non-finite state/cost, or missing diagnostics block progression. Invalid candidates receive NaN costs before MPPI weighting. G0 requires all candidates to be valid; ordinary MPPI may still return an action with only some valid candidates.

每个预期的 Kamino 细时间步都必须收敛。重置失败、残差超标、触及迭代或接触容量上限、状态或代价出现非有限数值，以及诊断缺失，都阻止进入下一阶段。无效候选在 MPPI 计算权重前被写入 NaN。G0 要求所有候选有效；普通 MPPI 在部分候选有效时仍可能返回动作，因此需要分别检查。

## G1: single-GPU calibration / 单卡负载校准

After G0, test N=32, 64, 128, and 256 in ascending order on one GPU. Keep the cube scene, horizon, cost weights, temperature 40, noise sigma 0.05, and one MPPI iteration fixed. For each N use seeds 0, 1, 2, one episode per seed, and up to twenty plans per episode: twelve short episodes and at most 240 plans. Stop a failing level before increasing N.

G0 通过后，在一张卡上按顺序测试 N=32、64、128、256。固定方块场景、预测时域、代价权重、温度 40、噪声标准差 0.05 和一次 MPPI 迭代。每个 N 使用种子 0、1、2，每个种子一个 episode，每个 episode 最多二十次规划：合计十二个短 episode、最多 240 次规划。某一级失败时先分析原因。

Record wall time, planning time after compilation/warm-up, GPU utilization over time, peak device memory, solver iterations/residuals, and valid-candidate counts. Choose N using valid work per unit time and planning quality, not utilization alone. Changing N changes the algorithm's sampling budget, so this study does not establish pure hardware speedup or planning convergence.

记录总耗时、排除编译和预热后的规划耗时、GPU 利用率时间序列、显存峰值、求解迭代数与残差，以及有效候选数量。选择 N 时考虑单位时间完成的有效工作和规划质量，不能只看利用率。改变 N 也改变了算法采样预算，不能据此得出纯硬件加速比或规划收敛结论。

## G2: four-GPU throughput / 四卡吞吐量

Use the validated N from G1 and the pooled driver. Compare one GPU with four GPUs on the same batch of 24 episode IDs and the same base seed, with up to 100 plans per episode. Both measurements must use the pooled driver: the sequential driver uses a different goal random stream. Start with one planner per GPU and four CPU evaluation workers; on four GPUs, test eight and sixteen workers if measured time permits. Freeze N and all scientific settings.

使用 G1 验证过的 N 和进程池入口。固定同一批 24 个 episode 编号及同一个基础种子，每个 episode 最多 100 次规划，对比单卡与四卡。两次都使用进程池入口：顺序入口的目标随机序列不同。起步时每卡一个规划进程、总共四个 CPU 评估进程；若实测耗时允许，再在四卡下测试八个和十六个 CPU 评估进程。保持 N 和科研参数一致。

Measure valid episodes per hour, batch wall time, per-GPU utilization/memory, request waiting, and solver validity. Fourfold acceleration is not guaranteed. The extended preflight checks one or four visible A100s. Pooled results record topology, per-episode request wait time, device routing, source identity, and a completion sidecar. An interrupted or incomplete batch cannot pass merely because its process exits successfully.

衡量每小时完成的有效 episode 数、整批耗时、各卡利用率/显存、请求等待和求解有效性。不能预设四卡一定快四倍。扩展后的预检核对一张或四张可见 A100；进程池结果记录进程分配、每个 episode 的请求等待、GPU 分配、代码来源和整批完成状态。中断或不完整的实验不会因进程正常退出就被判为通过。

## G3: research pilot / 科研试点

Use all four GPUs for independent parameter cells. The pilot crosses temperature {10, 40, 100} with noise sigma {0.02, 0.05, 0.10}: nine settings, three seeds each, one cube episode per seed, up to 100 plans per episode. This is 27 episodes and at most 2,700 plans. G3 launches independent sequential drivers, at most one per GPU, so its seeds match G1. G2 instead uses the shared pool. Freeze N, horizon, contact settings, cost weights, and MPPI iteration count. These values are exploratory, not tuned recommendations.

让四卡并行处理不同参数组合。试点为温度 {10、40、100} 与噪声标准差 {0.02、0.05、0.10} 的组合：九组配置，每组三个种子，每种子一个方块 episode，每个 episode 最多 100 次规划。合计 27 个 episode、最多 2,700 次规划。G3 启动独立的顺序实验进程，每张卡同时最多一组，其种子方式与 G1 一致；G2 使用共享进程池。固定 N、时域、接触设置、代价权重及 MPPI 迭代次数。这些是探索值，尚未调优。

Report task success/error, planning time, and invalid-candidate rates together. Larger noise may create harder contacts and invalidate a cell even when Python finishes. Three seeds per cell are a pilot, insufficient for a strong success-rate conclusion. Use these results to design more repeats or a BO search. Validate other objects separately.

同时报告任务成功情况/误差、规划耗时和无效候选比例。较大噪声可能产生更困难的接触，即使 Python 结束，该组实验也可能无效。每组三个种子只适合试点，不足以支持强有力的成功率结论。应据此设计更多重复实验或 BO 搜索，其他物体单独验证。

## Isolated GitHub delivery / 独立 GitHub 分支交付

The dedicated deployment branch is `codex/polaris-readiness`. Keep it separate from `main`, `planner_dev`, `refactor_and_simplification`, and existing experiment branches. It contains instrumentation and packaging changes relative to the base; original versions remain on their own branches. Clone into a new directory and create a separate environment rather than switching or modifying an active experiment checkout. Offline transfer remains an optional alternative.

专用部署分支为 `codex/polaris-readiness`，与 `main`、`planner_dev`、`refactor_and_simplification` 及已有实验分支保持隔离。它相对基线包含诊断与打包改动，原版本仍保留在各自分支。克隆到新目录并建立独立环境，不切换或修改正在运行实验的目录。离线传递只是可选替代方式。

The operator clones into a new directory, then detaches at the exact deployment commit supplied with the handoff. Replace `<deployment-commit>` with that full SHA, not the base SHA above. A pinned commit prevents later branch updates from silently changing an experiment. These commands must not target an existing working checkout. The submitters still accept any matching clean local commit and do not require GitHub access during a compute job.

操作者将代码克隆到新目录，再固定到交付时提供的精确部署提交。将 `<deployment-commit>` 替换为该完整 SHA，而不是上面的基线 SHA。固定提交可防止后续分支更新悄悄改变实验。这些命令不能指向已有工作目录。提交入口仍接受匹配的干净本地提交，不要求计算作业期间访问 GitHub。

```bash
git clone --single-branch --branch codex/polaris-readiness \
  https://github.com/yifanzhu95/contact_model_study.git \
  /home/<user>/contact_model_study_polaris
git -C /home/<user>/contact_model_study_polaris switch --detach <deployment-commit>
git -C /home/<user>/contact_model_study_polaris rev-parse HEAD
git -C /home/<user>/contact_model_study_polaris status --short
```

The initial G2/G3 budget is fixed at N=64. This is the second calibration level: more candidates than N=32 while avoiding the initial cost of N=128/256 for the longer pilot episodes. It is an engineering starting point, not an A100 measurement, optimal setting, or convergence claim. Use the G1 evidence to revisit it later. Throughput measurements describe this instrumented deployment version, not the uninstrumented baseline.

首轮 G2/G3 固定为 N=64，即第二档校准规模：候选比 N=32 更多，同时先不将较长试验扩大到 N=128/256。它是工程上的起点，不是 A100 实测结论、最优参数或收敛证明；以后再依据 G1 证据调整。吞吐量测量对应这个带诊断的部署版本，不等同于未加入这些诊断的基线版本。

## Operator configuration / 操作者配置

No source-code or PBS-script edits are needed to supply the deployment account and paths. Provide four runtime values: `--project` is the existing ALCF allocation/account string, `--env` is a new dedicated environment directory, `--out-root` is an existing writable Eagle results directory, and the source directory is the new clone containing `pipeline.py`. The submitter automatically passes the project to every job with `qsub -A`; it cannot create an allocation, grant access, or confirm the remaining balance. Confirm permission to use `debug`, including four held dependent jobs, before submission. Use absolute paths without whitespace, commas, quotes or backslashes. Do not place output or configuration files in the source checkout.

部署账户和目录通过运行参数提供，无需编辑源码或 PBS 脚本。需要四项值：`--project` 是已有的 ALCF 额度/账户名称，`--env` 是新的独立环境目录，`--out-root` 是已有且可写的 Eagle 结果目录，源码目录则是包含 `pipeline.py` 的新克隆目录。提交入口自动通过 `qsub -A` 将项目传给每个作业；它不能创建额度、授予权限或确认剩余额度。提交前确认可以使用 `debug`，包括四个挂起且带依赖的作业。目录使用不含空白、逗号、引号或反斜杠的绝对路径，不在源码目录内保存结果或配置文件。

Initial login, choosing the four values, obtaining an interactive preparation node, and starting submission are operator actions. Environment installation and its checks run inside `setup_environment.sh`; after successful submission and release, G0–G3 sequencing, runtime preflight, validation, GPU assignment and result recording are automatic. `--check` only validates the plan and driver arguments: it does not prove GPU compatibility, queue eligibility or completion within the time limit. Failures stop progression; there is no automatic repair, resubmission or whole-chain resume.

首次登录、选择四项配置、取得交互式准备节点和启动提交由操作者完成。`setup_environment.sh` 自动安装环境并执行相应检查；提交与放行成功后，G0–G3 衔接、运行前检查、有效性判定、GPU 分配和结果记录自动完成。`--check` 只验证计划和入口参数，不能证明 GPU 兼容性、队列权限或能在时限内完成。失败会阻止继续，不自动修复、重提或整链续跑。

## G0 operator commands / G0 操作命令

The authorized operator supplies the ALCF project name, repository path, environment path, and Eagle output root. Replace every angle-bracket placeholder. Request an interactive compute node following the [official PBS workflow](https://docs.alcf.anl.gov/polaris/running-jobs/).

获授权操作者提供 ALCF 项目名、源码目录、环境目录和 Eagle 结果根目录。替换所有尖括号占位符。按照[官方 PBS 流程](https://docs.alcf.anl.gov/polaris/running-jobs/)申请交互式计算节点。

```bash
qsub -I -A <project> -q debug -l select=1:system=polaris \
  -l walltime=01:00:00 -l filesystems=home:eagle
```

On the allocated compute node, build the environment with Python 3.12, MuJoCo 3.6, Newton 1.6, and Warp 1.17. Build it on Polaris rather than copying the workstation environment.

在获分配的计算节点上建立环境：Python 3.12、MuJoCo 3.6、Newton 1.6、Warp 1.17。在 Polaris 建立环境，不复制工作站的现成环境。

```bash
bash experiments/polaris/setup_environment.sh \
  /home/<user>/contact_model_study_polaris \
  /home/<user>/envs/contact-kamino-py312
```

After setup and returning to the login shell, choose either the single-launch pipeline below or the separate G0 command here; do not submit both for the same experiment. The separate script requests PBS debug, one node, and thirty minutes. Create the Eagle output root first. The script prints the job ID and result path; inspect with `qstat -f <job-id>`.

准备完成并返回登录节点后，可选择下方的一次启动流程，或这里的单独 G0 命令；同一次实验不要两者都提交。单独脚本申请 PBS debug 队列、一个节点、三十分钟时限。提前建立 Eagle 结果根目录。脚本打印作业编号和结果路径，用 `qstat -f <job-id>` 查看状态。

```bash
bash experiments/polaris/submit_m5_smoke.sh \
  <project> \
  /home/<user>/contact_model_study_polaris \
  /home/<user>/envs/contact-kamino-py312 \
  /eagle/<project>/<user>/contact_model_study
```

Preserve the whole output directory: preflight, test logs, cell results/status/arrays, `smoke_verdict.json`, environment snapshots, artifact manifest, and PBS stdout/stderr. PBS stages stdout/stderr after the job, outside the in-job hash manifest. G0 progression requires `smoke_verdict.json` to report `passed=true`.

保留完整结果目录：预检、测试日志、各组结果/状态/数组、`smoke_verdict.json`、环境快照、产物清单和 PBS 标准输出/错误日志。PBS 在作业后放置的输出/错误文件不在作业内哈希清单中。进入下一阶段要求 `smoke_verdict.json` 显示 `passed=true`。

## Single-launch pipeline / 一次启动完整流程

After environment preparation, `pipeline.py submit` is the single entry for G0–G3. It submits four separate jobs with PBS `afterok` dependencies: submission happens together, execution remains sequential. Each job requests one node; G0 has thirty minutes and the other stages one hour each. There may be queue waits between stages. See the [ALCF PBS dependency documentation](https://docs.alcf.anl.gov/running-jobs/not_in_nav/pbs-qsub-options-table/).

完成环境准备后，`pipeline.py submit` 是 G0–G3 的统一入口。它提交四个独立作业，以 PBS 的 `afterok` 依赖连接：一起提交，但按顺序执行。各作业申请一个节点，G0 时限三十分钟，其余阶段各一小时。阶段之间可能等待排队。机制见 [ALCF 官方 PBS 依赖说明](https://docs.alcf.anl.gov/running-jobs/not_in_nav/pbs-qsub-options-table/)。

The reviewed first-deployment default is N=64 for G2/G3; omitting `--n-samples` uses that fixed value, and the commands below state it explicitly for clarity. It is not a claim that N=64 has passed. The automatic chain requires all of G1 to complete and pass, and N=64 to appear in its validated list. It does not choose the largest N or silently fall back to a smaller one. Explicit overrides to 32, 128 or 256 are supported for later reviewed runs. Separate-stage operation below can still be used for a reviewed smaller-N recovery after a failed higher level.

首轮部署已选定 G2/G3 默认使用 N=64；省略 `--n-samples` 也会采用这个固定值，下方命令为清楚起见仍显式写出。它不代表 N=64 已通过验证。自动作业链要求 G1 整体完成并通过，而且 N=64 出现在已验证列表中。它不会自动选择最大的 N，也不会悄悄改用较小值。以后经审阅可显式改为 32、128 或 256。若较高一级失败，经审阅后仍可用下方的分阶段方式，在已验证的小 N 上恢复工作。

Preview from the prepared Python environment with `--check`. It validates the four stages' driver options and prints their paths, fixed N and time limits; it creates no output directories and never calls PBS. A dirty workstation checkout is allowed for preview only. For real submission, use the clean local deployment commit, existing Eagle root and prepared environment; the project/repository/environment/output paths must not contain whitespace, commas, quotes or backslashes. Replace all placeholders; do not run this from the workstation.

在准备好的 Python 环境中加 `--check` 预览：它检查四阶段的入口参数，打印路径、固定 N 和时限，不建立结果目录，也不调用 PBS。预览允许工作站的未提交副本。正式提交必须使用干净的本地部署提交、已有 Eagle 目录及准备好的环境；项目名、源码/环境/结果路径不能包含空白、逗号、引号或反斜杠。替换所有占位符；不要在工作站执行正式提交。

```bash
<env>/bin/python <repo>/experiments/polaris/pipeline.py submit --check \
  --project <project> --env <env> --out-root <eagle-root> --n-samples 64

<env>/bin/python <repo>/experiments/polaris/pipeline.py submit \
  --project <project> --env <env> --out-root <eagle-root> --n-samples 64
```

All four jobs are first submitted with a user hold, meaning they cannot run. Once every ID is recorded, the script releases G3, G2, G1, then G0; PBS dependencies still prevent later stages from running early. Only after the command reports that all jobs were accepted and released is normal sequencing independent of the operator's computer or SSH connection. No monitoring daemon or further job submission is needed. Runtime wrappers check the registered job, source commit, fixed parameters, preceding full-stage verdict and final artifact seal before launching a stage.

四个作业首先以用户挂起状态提交，暂不能运行。全部作业编号记录成功后，脚本依次放行 G3、G2、G1，最后放行 G0；PBS 依赖仍会阻止后续阶段提前运行。只有命令确认全部作业已接受并放行后，正常衔接才不再依赖操作者的电脑或 SSH 连接。无需常驻监控程序，也无需再提交下一阶段。每阶段运行前核对已登记作业、代码提交、固定参数、前阶段的完整通过判定及最终产物清单中的状态文件哈希。

The command prints a new `<pipeline-root>`. Preserve `pipeline_plan.json`, `submission.json` and the `g0`–`g3` subdirectories. The submission ledger records scheduler commands, responses, IDs and release state. If submission/release fails or a response is unknown, the command stops without automatic retries or job deletion. G0 is kept held unless the final release was attempted; an ambiguous final release may already have started it. Inspect `submission.json` and `qstat -xf <job-id>` before any recovery. Never blindly rerun the submit command or release G0 from a partially submitted chain.

命令打印新的 `<pipeline-root>`。保留 `pipeline_plan.json`、`submission.json` 及 `g0`–`g3` 子目录。提交记录包含调度命令、返回信息、作业编号和放行状态。若提交/放行失败，或返回情况不明，命令会停止，不自动重试或删除作业。除非已经尝试最后放行，G0 都仍被挂起；最后一次放行若返回不明，G0 可能已经启动。恢复前核对 `submission.json` 和 `qstat -xf <job-id>`。不要盲目重跑提交命令，也不要在作业链只提交了一部分时自行放行 G0。

Inspect file-based progress with the command below. It does not submit jobs, query PBS or start a watcher. A hard kill can leave a stale running status or no final status; check PBS when the file evidence is incomplete. A scientific failure, time limit or infrastructure error stops progression. This first version has no automatic resubmission or whole-chain resume; recover deliberately using the separate stage scripts and their existing cell checkpoints, after confirming that the old jobs cannot still write to those directories. Submitting the pipeline again creates a new run, not a continuation.

用下方命令查看文件记录中的进度。它不提交作业、不查询 PBS，也不启动监控。强制终止可能留下过时的运行状态，或没有最终状态；文件证据不完整时需查看 PBS。科研检查失败、超时或基础设施异常都会阻止继续。首版不自动重提，也不提供整链续跑；应先确认旧作业不会继续写入这些目录，再用独立阶段脚本及其已有配置检查点恢复。再次提交完整流程会创建新实验，不是续跑。

```bash
<env>/bin/python <repo>/experiments/polaris/pipeline.py status --root <pipeline-root>
```

## Separate G1–G3 operation / 单独运行 G1–G3

Before submission, the operator can inspect all workloads with `--check`. This parses the scientific and pooled-driver options without running simulations or submitting jobs. The N=32 examples below only preview the plan; they do not establish that N=32 is valid on Polaris.

提交前，操作者可用 `--check` 查看全部工作内容。此操作解析科研参数和进程池参数，不运行仿真或提交作业。下面的 N=32 仅用于预览，不能据此判定它在 Polaris 有效。

```bash
python experiments/polaris/run_stage.py g1 --check --outdir /tmp/polaris-preview
python experiments/polaris/run_stage.py g2 --check --n-samples 32 --outdir /tmp/polaris-preview
python experiments/polaris/run_stage.py g3 --check --n-samples 32 --outdir /tmp/polaris-preview
```

Submit each stage separately using the previous stage's verdict path. G1 requires passing G0. For G2, replace `<validated-N>` with one number listed in G1's `validated_samples`: all three seeds must pass at that N. A failure at a larger N stops calibration, but a previously validated smaller N remains usable. G3 requires a fully passing G2 at the same N.

每个阶段分别提交，并传入前一阶段的判定文件。G1 要求 G0 通过。提交 G2 时，将 `<validated-N>` 替换为 G1 的 `validated_samples` 中的一个数：该 N 下三个种子必须全部通过。更大的 N 失败会停止校准，但此前已验证的较小 N 仍可使用。G3 要求同一 N 下 G2 整体通过。

```bash
bash experiments/polaris/submit_stage.sh g1 <project> <repo> <env> <eagle-root> \
  <g0-output>/smoke_verdict.json -
bash experiments/polaris/submit_stage.sh g2 <project> <repo> <env> <eagle-root> \
  <g1-output>/stage_verdict.json <validated-N>
bash experiments/polaris/submit_stage.sh g3 <project> <repo> <env> <eagle-root> \
  <g2-output>/stage_verdict.json <validated-N>
```

Each stage requests one node and one hour in PBS debug, with a 50-minute workload budget and time reserved for shutdown. Completion within that budget is not assumed. If the stage is incomplete, resubmit the same command with its existing output directory as an eighth argument. Completed cells are checked and skipped; incomplete or failed cells restart from the beginning in a new attempt directory. Partial episodes are retained as evidence, not continued mid-trajectory. Do not submit two jobs to the same output directory concurrently.

每个阶段申请 PBS debug 队列的一个节点、一个小时，其中工作预算为五十分钟，并留出退出时间。不能预设整阶段能在预算内完成。若未完成，在相同提交命令末尾加上已有结果目录作为第八个参数。已完成配置会重新核对后跳过，未完成或失败配置在新的 attempt 目录从头运行。部分 episode 保留作为证据，不从轨迹中途继续。同一结果目录同时只能运行一个作业。

```bash
bash experiments/polaris/submit_stage.sh g2 <project> <repo> <env> <eagle-root> \
  <g1-output>/stage_verdict.json <validated-N> <existing-g2-output>
```

Read `stage_verdict.json` for completion, pending cells, validated N values, and G2 speedup ratios. Each cell has a `cell.status.json` and numbered attempt directories containing driver logs, result arrays/JSON, GPU telemetry, and pooled statistics where applicable. GPU telemetry is sampled approximately once per second; its memory maximum is a sampled maximum and may miss short spikes. Utilization includes initialization. `planner_wait_overhead_s` includes queue and communication overhead; it is not a direct measurement of CPU idle time. Planning-time means exclude the first plan of each episode, which is only a conservative approximation of warmed execution.

读取 `stage_verdict.json` 查看完成状态、待运行配置、已验证 N 和 G2 加速比。各配置包含 `cell.status.json` 及编号 attempt 目录，保存日志、结果数组/JSON、GPU 采样和适用时的进程池统计。GPU 约每秒采样一次，显存最大值是采样最大值，可能漏掉短暂峰值；利用率包含初始化。`planner_wait_overhead_s` 包含排队和通信开销，不等于 CPU 空闲时间。规划时间均值排除每个 episode 的首次规划，只是对预热后执行耗时的保守近似。

## Current evidence / 当前证据

The selected offline suite passed 75 tests, with five GPU-marked tests excluded. These checks cover stage budgets, prerequisite/source gates (including a clean detached commit), incomplete results, per-GPU scheduling, resume validation, synthetic child-process execution, held-job submission/release ordering, partial scheduler failures, runtime pipeline gates, artifact sealing, and wrapper failure/signal propagation. Fake PBS command-line tools were used; no real scheduler was contacted. Diagnostic kernels were executed on CPU. The workstation reported an NVML driver/library mismatch. Actual CUDA execution, A100 memory limits, G1–G3 timings, and four-GPU solver behavior remain unmeasured. Authentication and submission are performed by the authorized operator.

所选离线测试共 75 项通过，排除了五项标记为需要 GPU 的测试。这些检查覆盖阶段规模、前置阶段/版本检查（包括固定在精确提交的干净副本）、不完整结果、每卡调度、续跑核对、模拟子进程执行、挂起作业提交/放行顺序、调度中途失败、运行时阶段检查、产物封存，以及包装脚本的错误/终止信号传递。测试使用模拟 PBS 命令行工具，没有连接真实调度器。诊断内核已在 CPU 上执行。工作站报告 NVML 驱动/库不匹配。实际 CUDA 执行、A100 显存限制、G1–G3 耗时和四卡求解器行为均未测量。登录认证与提交由获授权操作者执行。
