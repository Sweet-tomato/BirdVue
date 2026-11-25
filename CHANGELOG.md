# bird_service 变更记录

> 历史迭代版本不再保留本地副本，完整代码与演进过程以本仓库 git 提交历史为准。
> 追溯方式：`git log --oneline`、`git show <commit>`、`git log -p -- <path>`。

## 2026-11-25 设备经纬度多格式兼容

背景：现场设备上报的坐标不在顶层 `location`，实际出现在两处：`gpsFence`（`{"latitude":22.58303269793829,"longitude":113.9343140112498,"radius":50}`）和 `attachment`（`{"moonphase":"FULL_MOON","batteryPower":71,"latitude":"22.5828","longitude":"113.93482","timestamp":1795491998414}`），值是数字/字符串/JSON 字符串混杂，导致地理硬过滤一直拿不到坐标、中性回退。

改动：

- `api/_common.py`：新增 `normalize_location()`。按 `location` > `gpsFence` > `attachment` 顺序取第一个可解析坐标；支持 dict、JSON 字符串、双重编码 JSON、数字与数字字符串；也支持把这些字段放在 `deviceInfo` 对象内。
- `api/bird_frames_service.py`：`_build_observation` 改用 `normalize_location`，输出契约保持不变（`source=device`、坐标为 float），坐标直接从 `gpsFence`/`attachment` 取到时同样进入硬过滤。
- `pipeline/geographic_prior.py`：`valid_location` 兼容数字字符串形式的经纬度。
- `API.md`：补充 `gpsFence` / `attachment` 示例、解析优先级与 `deviceInfo` 说明。
- 新增 `tests/test_location_normalization.py` 12 个用例；全套 87 个单测通过。

## 2026-11-25 地理硬过滤下的开放提案（撤销 fe81654 的禁用）

背景：硬过滤后豆包既看不到坐标、也不允许提案，本地池没有真鸟（迷鸟、屏幕/远程直播画面）时无法救回。要求豆包能看到经纬度、保留开放提案能力。

改动：

- 撤销 `fe81654` 的“过滤生效即禁用提案”；`pipeline/doubao_report.py` 提案分支恢复，并把 `geo_filter_applied` 传入 `accept_confident_proposal`。
- 系统提示词：过滤生效时向豆包提供设备坐标（latitude/longitude）、所在 3° 网格与本地池物种数，并说明“图像明显不是本地鸟时可提 `outsideCandidates`，纯本地鸟种取舍不要提案”。
- `accept_confident_proposal` 新增 `geo_filter_applied` 参数：绕过地理过滤时证据门槛提高到 `proposalGeoBypassMinEvidence`（默认 3 条）；采纳后写 `review.openSetProposal.geoFilterBypassed=true` 与 `meta.openSetEscape.geoFilterBypassed=true`。
- `config/pipeline_config.json` 新增 `openSetEscape.proposalGeoBypassMinEvidence=3`。
- 文档：README、API.md、PRODUCTION.md 更新。

验证：北京坐标 + 显示器播放事件（`8027e6d6`）：豆包在 notes 中识别出“显示器播放画面”，一次判 `no_bird`，一次提出池外 Hair-crested Drongo 但质量分 40 被 50 分质量门拒绝；新增 2 个单测（绕过标记、证据门槛），75 个单测通过。

风险：豆包可能借提案绕过地理过滤给出非本地物种（屏幕图容易误提案），靠证据门槛 + AviList 校验 + 置信度封顶抑制；上线后需监控 `geoFilterBypassed=true` 的采纳频率。

## 2026-11-25 地理先验：按经纬度硬过滤候选池（P0）

背景：设备上传经纬度后，应把 BioCLIP25 的候选范围收窄到附近鸟种，Top5 只排本地鸟，减少外地近似种误命中，再交给豆包复核。

改动：

- `pipeline/geographic_prior.py`：新增 `allowed_species()`。`filterMode=hard` 时读取 3° 网格资产，返回所在格的物种集合；无坐标、覆盖范围外、或池小于 `minPoolSpecies` 时返回 `None` 中性回退。
- `pipeline/bird_cli.py`：`analyze_local` 在相似度计算后按允许集合把主候选/补充候选分数 mask 为 -1.0，Top5、margin、风险门、复核短名单全部只基于本地鸟种；`meta.geographicPrior` 增加 `filterMode`、`hardFiltering`、`filter{applied,reason,cellId,poolSpecies,maskedSpecies}` 摘要。
- `config/pipeline_config.json`：`geographicPrior.enabled=true`、`filterMode=hard`、`minPoolSpecies=20`、`maxBonus=0`（不再叠加旧的软加分）。
- 文档：README、PRODUCTION、API.md 同步更新 location 语义。

验证：北京坐标（39.9042, 116.4074）命中 cell `W115_S39`，pool 552 种，mask 掉 10,057 种；白头鹎样张 Top5 全部变为北京常见鸟且 top1 不变；无 location、(0,0)、minPoolSpecies 过大均中性回退；新增 `tests/test_geographic_filter.py` 6 个用例，73 个单测通过。

风险：资产只覆盖中国+北美；若设备在覆盖外则不生效。屏幕/远程直播画面不要传 `location`——用 `8027e6d6`（显示器里的极乐鸟）传北京坐标验证过：Top5 会被强制换成本地鸟种，正确鸟种被过滤掉。

## 2026-11-24 开放物种提案（豆包自报物种）

背景：`d709a2bf` 的豆包复核判定“画面是雀类，不是任何果鸠候选”，候选池里没有该物种，旧逻辑只能 `reject_all` 输出 Unknown。用户要求：豆包很确定时，允许它输出自己的答案。

改动：

- `config/pipeline_config.json` 的 `openSetEscape` 新增：`allowConfidentProposal=true`、`proposalMinimumVisibleFeatures=2`、`proposalMinimumQuality=50`、`proposalConfidenceCap=0.5`。
- `pipeline/bird_cli.py`：新增 `resolve_species_entry`，用 AviList 注册表把英文常用名 / 双名法学名解析成物种条目（缓存一次加载）。
- `pipeline/doubao_report.py`：开关打开时，schema 允许 `outsideCandidates` 填任意物种名，提示词增加“证据充分才可提案”的规则；首轮决策为 `reject_all|genus|family` 时，若提案能解析到 AviList 条目、带 ≥2 项可见证据和至少 1 个被否定的候选槽位，则直接采纳（仍为单次豆包调用），置信度封顶 0.5，并记录 `review.openSetProposal` 与 `meta.openSetEscape`（reason=`confident_proposal_accepted`）。提案未通过时保持原逻辑。

验证：`d709a2bf` 输出 Gouldian Finch（`Chloebia gouldiae`），`decisionSource=doubao_open_set_escape_review`；`8027e6d6` 回归仍为 Black Sicklebill；60 个单测通过。

风险：靠 AviList 解析、证据条数与置信度封顶抑制幻觉，但“豆包自报物种”本质上未做二次独立确认；Gouldian Finch 的真值仍未经你确认。

### 补充（同日晚）：允许提案推翻首轮候选

背景：prompt_lab 样本 `b4eeb6c6` 的用户真值是 Jungle Myna（`Acridotheres fuscus`），豆包在一次回放里首轮选了 Great Myna（`Acridotheres grandis`）、同时在 `outsideCandidates` 提出 Jungle Myna，但旧逻辑只在 `reject_all|genus|family` 时采纳提案，正确提案被丢弃。

改动：

- `config/pipeline_config.json` 新增：`proposalOverrideCandidate=true`、`proposalOverrideMinEvidence=3`、`proposalOverrideConfidenceCap=0.4`。
- `pipeline/doubao_report.py` 的 `accept_confident_proposal`：首轮 `candidate` 时也可采纳提案，条件为——提案解析到 AviList、证据 ≥3 条、`contradictedCandidateRanks` 包含首轮所选槽位号、且提案与首轮候选同科（避免跨科幻觉顶替，如把夜鹭换成椋鸟）。采纳后置信度封顶 0.4，`review.openSetProposal.overrodeCandidate` 记录被替换的首轮候选，`meta.openSetEscape.reason=confident_proposal_overrode_candidate`。
- 提示词补充：首轮候选仍明显不符时，允许提案并在 `contradictedCandidateRanks` 里写明该候选槽位号。

验证：新增 `tests/test_confident_proposal.py` 7 个用例（同科提案推翻、跨科拒绝、未否定所选槽位拒绝、证据不足拒绝、reject_all 跨科采纳、无法解析拒绝、开关关闭保留候选），全套 67 个单测通过。注：`b4eeb6c6` 线上复跑两次豆包都直接选 Great Myna、未再提案，因此推翻路径暂时只有单测覆盖，没有线上命中记录。

## 2026-11-24 复核输入修复（P0+P1）

问题：11-23 事件 `8027e6d6` 真值为 Black Sicklebill（`Epimachus fastosus`，本地候选第 4），链路误判成 Magnificent Riflebird。根因是送审输入缺图——主帧按清晰度选中了没有鸟的 frame0；事件帧分支只发 frame0 原图 + 一张仅覆盖颈背蓝绿羽毛的紧贴裁剪，frame1、frame2 原图根本没送；豆包只看到一片羽毛，只能给 `body_color`，其余七项不可观察。

改动：

- `api/bird_frames_service.py`：`_select_best_frame` 改为检测优先——逐帧检测，检出分/面积最高的帧作主帧，全部未检出才退回清晰度；`/v1/identify` 与后台 worker 都传入 runtime。附带影响：`bestImageIndex` 语义变为"检测最优帧"。
- `pipeline/bird_cli.py`：新增 `detect_frame_scores`；`_tight_crop` 边距 4%→18%；`_event_frame_crops` 跳过置信度 <0.35 的检测框（宁可发原图，不发误导性局部裁剪）。
- `pipeline/doubao_report.py`：事件帧送审改为发送全部原图（最多 3 张）+ 最多 2 张高置信特写，提示词同步改为"先独立观察每张原图与特写，再跨帧合并证据"。

验证：该事件重跑，主帧选为 frame2，豆包观察到长弯镰刀喙并选中 Black Sicklebill（`Epimachus fastosus`）；60 个单测通过；服务重启后 `/health/ready` 200。

## 2026-11-24 资产整理（第四阶段）

- 资料目录归并：`bird_service_archive/` → `assets/archive/`，`prompt_lab/` → `assets/prompt_lab/`；`PROJECTS.md` 更新为新布局。
- 删除：临时脚本 `inspect*.sh`、`cleanup3.sh`；空目录 `.venvs`；`.history`、`.lesshst`；`BirdNET-Pi-main.rar`（与解压目录重复）与 `BirdNET-Pi-main/`（第三方参考项目，音频链路只用 `models/birdnet` 的 tflite）。
- 保留：`bird_service/`、`assets/archive/bird_service.zip`（唯一模型兜底）、`venv/`（systemd 与启动脚本写死路径）、IDE 与系统目录。

## 2026-11-24 资产整理（第三阶段）

原则：落后版本不保留本地副本，迭代过程写清楚；能合并的文档合并。

### 文档合并

- `README.md` 重写为唯一入口：修正 `openSetEscape`（实际 `true`）与 `marginThreshold`（实际 `0.05`）的过期描述，补音频服务、CPU-only 现状、`birdctl` 运维方式。
- `PRODUCTION.md` 重写为当前运行手册，并入原 `PROD_VALIDATION.md` 的验证结论与数字；删除 `PROD_VALIDATION.md`。
- 删除 `config/doubao.env.example`：与 `config/service.env.example` 重复，且其中的 `ARK_API_KEY` 写法代码已不支持。

### 历史副本

- 删除 `bird_service_archive/` 下 `bird_service_opt_test/`、`bird_service_prod_test/`、`dead_code_20261111/`、`checksums/`、`CHANGE_20261111_P0/`、`ORGANIZE_LOG.md`；仅保留 `bird_service.zip`（3.2G，11-09 原始交付包，唯一含模型权重的兜底副本）。
- 删除 `bird_service_backups/`（源码快照 tar.gz、manifest、旧 `doubao_report.py.bak_1745`）；代码与提示词迭代已在 git 历史与 `prompt_lab/`。
- 原归档要点：2026-11-11 将 opt/prod 两份测试副本（167 + 153 个文件）sha256 校验通过后归档；`dead_code_20261111/` 收拢 `Dockerfile`、`Dockerfile.cpu`、`docker-compose.yml`、`README.txt`、`BUNDLE_FILES.json`、项目内 `.venv`、`start_service.sh.bak`、`rollout_service.log`，主因生产已切 systemd。
- 原 P0 变更要点：systemd 守护 + Bearer 鉴权上线，启动方式由 `nohup` 改为 systemd，新增 `config/service.env`（600）与 `deploy/bird-service.service`；回滚到无鉴权手工启动的步骤见该次提交。

### 提示词

- 删除 `bird_service_prompt_opt` worktree 与分支 `prompt-opt-cn`：分支停在 `50d9b3e` 之后 1 个提交，其「提示词中文化并精简」已由 `7b4ac72` 落地生产。
- `prompt_lab/` 保留为提示词实验记录（`EXPERIMENTS.md` + 原始 jsonl）：
  - 主线：v0 基线 84.9% → v1 `b8bcd69` 87.5% → v2 `7b4ac72` 85.7%，p50 -10.0%，输出 tok -6.2%。
  - 归因：提速来自输出限字，不是中文化；图片 token 占输入约 55%，文本瘦身收益天花板 <2%。
  - 定种意愿：E3（qualityScore≥80 强制定种）上生产；`forceAllowsUnknown=false` 会使精度崩到 57.1%，已否决。
  - 推理档位：`DOUBAO_REASONING_EFFORT=minimal` 上生产后豆包段 p50 -37.5%、端到端 p50 -25%，精度无统计差异。
  - 精度上限在本地候选池（Top5 硬上限），提示词层面已无低风险增益空间。

### 性状包与模型资产

- 删除 `text_overrides/` 下 4 个 `.bak` 旧版本；版本迭代：v5 原始（json 340K + npz 2.6M）→ pre_v6（337K + 2.6M）→ 当前 v5 扩展（3.8M + 37M，覆盖约 9k 物种，配套字段包 `common_birds_fields_v1.json`）。
- 当前性状包与 `.npz`、`models/geographic/`、`config/region_priors.json` 均为主链路资产，保留。
- `models/onnx/e19/`（3.0G）保留：文档化的可选编码器后端，生产默认 `BIRD_ENCODER=torch`。

## 2026-11-24 /root 资料整理（第二阶段）

按"只清一次性资料，不动主链路资产"原则清理 `/root`，合计释放约 5.5 GB。

保留的主链路资产：`models/bioclip25/birds/text_overrides/`（性状子句包及历史版本）、`models/geographic/` 与 `config/region_priors.json`（GPS 先验）、`models/onnx/e19/`（ONNX 图像编码器）、`deploy/fetch_gbif_distribution.py`、`deploy/build_geographic_asset.py`、`bird_service_prompt_opt/`（prompt-opt-cn 分支）、`prompt_lab/`（提示词实验数据）、`bird_service_archive/`、`bird_service_backups/`、`BirdNET-Pi-main*`。

### 已删除的一次性实验与结果

脚本已纳入提交 `7b1087f`，内容可通过 `git show 7b1087f:experiments/20261121/<文件名>` 查看。

| 路径 | 大小 | 说明 |
|---|---|---|
| `bird_batch_20261121/` | 407M | 1200 事件批量评测图与脚本 |
| `garbage_se_b/`、`garbage_se.onnx` | 699M | 垃圾检测训练实验（含 .venv） |
| `loadtest_20261121/` | 44K | 压测产物；`tests/loadtest.py` 已入库 |
| `小模型链路处理结果_20261121.{html,xlsx,zip}` | 15M | 一次性评测结果 |
| `豆包检出结果_20261123*.xlsx` | 15M | 一次性排查数据 |
| `bird_pipeline_state_machine.{html,excalidraw}` | 3.4M | 状态机导出图；`STATE_MACHINE.md` 已入库 |
| `yolo_crop_variants.jpg`、`proto_experiment.log`、`整理代码清理冗余脚本.md` | 320K | 实验残留与对话导出 |
| `/root/.cache/pip` | 4.4G | pip 缓存 |
| `/root/.venvs/report` | 49M | 旧报告 venv |
| `/root/.trae-html-share-packages` | 5.4M | IDE 共享包缓存 |

### 仓库内删除

- `USAGE.md`：写死 `/opt/bird_service` 的通用安装说明，与 `README.md` 重复。

## 2026-11-24 代码清理

- 将 `/root` 下 11 个一次性实验脚本纳入提交 `064362e`（内容仅存于 git 历史），随后从工作区与本地删除。
- 移除 `pipeline/taxonomy_registry.py` 中无生产引用的 `raw_candidate_rows`、`find_by_scientific_name`。
- 将 `/root` 下 11 个实验数据文件（缓存/结果/对话导出）纳入提交 `7de04f5`，随后从工作区与本地删除。

### 已删除的一次性脚本

内容可通过 `git show 064362e:experiments/20261123/<文件名>` 查看。

| 脚本 | 用途 |
|---|---|
| `build_trait_pack_v6.py` | 基于 DongNiao 对照语料扩展 text_overrides 性状包（约 9k 物种） |
| `update_trait_pack.py` | 新增山雀类 tier-D 性状行并改写 Tufted Titmouse 性状文本 |
| `build_fields_asset.py` | 从抽取缓存生成 `common_birds_fields_v1.json`（含 QC 门槛） |
| `extract_fields.py` | 通过豆包批量抽取易混物种的 8 项形态字段 |
| `fix_comparative_fields.py` | 规范化比较级字段值并复用字段 QC 门槛 |
| `fix_javan_myna.py` | Javan Myna 单物种数据修正 |
| `proto_analysis_v2.py` | 提示词原型实验结果分析 |
| `proto_experiment.py` | 提示词原型批量实验 |
| `render_prompt.py` | 提示词渲染 |
| `screen_test.py` | 屏幕四宫格切图并跑本地链路 |
| `screen_e2e.py` | 四帧本地 + 豆包复核端到端验证 |

### 已删除的实验数据文件

内容可通过 `git show 7de04f5:experiments/artifacts/<文件名>` 查看（Netron 文件名含空格，需加引号）。

| 文件 | 用途 |
|---|---|
| `proto_experiment_cache.json` | 原型检索实验的向量缓存，用于断点续跑 |
| `proto_experiment_results.json` | 原型检索实验汇总结果（easy/hard 分组命中率） |
| `proto_analysis_v2.json` | 对原型实验结果的二次分析（分物种、分一致/分歧子集） |
| `field_extract_cache.jsonl` | 豆包批量抽取形态字段的逐物种缓存 |
| `field_fix_cache.jsonl` | 比较级字段规范化的处理缓存 |
| `screen_recapture_result.json` | 屏幕四帧本地 + 豆包复核端到端结果 |
| `screen_recapture_result.pre_fields.json` | 性状包更新前的同项结果对照 |
| `doubao_prompt_example.json` | 豆包复核接口请求样例（json_schema、temperature 等） |
| `Netron SiLU split vs CBS BN GL_*.json` | 3 份 CodeBuddy 对话导出，记录 Netron 中 SiLU split 与 CBS BN GL 结构对比 |

## 迭代历史

| 提交 | 说明 |
|---|---|
| `6d8120b` | 建立 bird_service 生产代码 git 基线 |
| `4ae9a8f` | systemd 托管与 Bearer 鉴权 |
| `a4c2c4d` | 新增标准库压测脚本 `tests/loadtest.py` |
| `7d5e9ce` | `/v1/jobs` 立即返回 202，本地阶段移入后台 worker |
| `29c588a` | 模型部署端输出字段与按日分层持久化存储 |
| `50d9b3e` | 归档无引用文件至 `bird_service_archive/dead_code_20261111` |
| `b8bcd69` | 优化豆包复核决策策略并固定 temperature=0 |
| `7b4ac72` | 豆包提示词中文化并精简输出 |
| `98b5aa0` | 回调推送 + 出参嵌套结构重构 + 候选提取统一入口 |
| `fe08189` | 开启 open-set 单链路候选注入 + GPS 地理先验接线 |
| `54f5032` | 修正 location 完成态回显说明，顶层保持 14 字段契约 |
| `da655aa` | 上线 GBIF 网格地理先验 + GPS 软加分生效 |
| `cad1070` | 补充识别流水线状态机说明（确定性 DAG，非 Agent） |
| `1e15431` | 默认推理档位改为 minimal、输出上限 2048 |
| `d37e896` | 新增独立声学识别服务并校准本地决策阈值 |
| `064362e` | 记录 2026-11-23 一次性实验脚本（仅供历史追溯） |
| `7de04f5` | 记录一次性实验数据文件（仅供历史追溯） |
| `7b1087f` | 记录 2026-11-21 批量评测、压测与垃圾检测实验脚本（仅供历史追溯） |
| `afc45ce` | 删除重复 USAGE.md，更新 CHANGELOG（第二阶段） |
