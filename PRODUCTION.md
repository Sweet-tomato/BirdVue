# 生产运行手册

最后更新：2026-11-24 CST

## 一、部署形态

- systemd 单元 `bird-service`（源 `deploy/bird-service.service`），`Restart=always`，日志追加 `logs/uvicorn.log`。
- 启动脚本 `start_service.sh`，通过 `EnvironmentFile` 加载 `config/service.env`（权限 600，不入库）。
- 监听 `0.0.0.0:8000`，uvicorn 单 worker，CPU 推理。
- 音频服务 `api/bird_audio_service.py`（`:8001`）代码、模型、启动脚本就绪，**尚未部署 systemd 单元**；手工启动用 `bash start_audio.sh`。

## 二、常用命令

```bash
systemctl status bird-service
systemctl restart bird-service
curl -s http://127.0.0.1:8000/health/live
curl -s http://127.0.0.1:8000/health/ready
curl -s http://127.0.0.1:8000/health/state

./birdctl log            # 查看日志
./birdctl status         # 两个服务状态
./birdctl restart image  # 仅图片服务（screen 方式，systemd 运行时勿用）
```

## 三、关键配置

`config/service.env`：

- 豆包：`OPENAI_BASE_URL1`、`OPENAI_VLM_MODEL1`、`OPENAI_API_KEY1`、`DOUBAO_REASONING_EFFORT=minimal`、`DOUBAO_MAX_OUTPUT_TOKENS=2048`
- 鉴权：`BIRD_API_KEY`（留空不鉴权）
- 复核：`BIRD_REVIEW_ENABLED=1`
- 并发：`BIRD_MAX_CONCURRENCY`、`BIRD_QUEUE_WORKERS`、`BIRD_LOCAL_CONCURRENCY`、`BIRD_MAX_PENDING`
- 落盘与日志：`BIRD_PERSIST_ENABLED`、`BIRD_RETENTION_DAYS`、`BIRD_LOG_RETENTION_DAYS`

`config/pipeline_config.json`：

- `localDecision.marginThreshold=0.05`（20261121 批次标定）
- `openSetEscape.enabled=true`（单链路，每事件一次豆包）
- `geographicPrior`：`enabled=true`、`filterMode=hard`、`minPoolSpecies=20`，资产为 GBIF 3° 网格（中国+北美 473 格）；设备传 `location` 时硬过滤候选池，未传或覆盖范围外中性回退。豆包能看到坐标与过滤说明，非本地鸟仍可走 `outsideCandidates` 提案并审计 `geoFilterBypassed`
- `review.defaultMode=auto`

运行参数：`BIRD_WORKERS`、`BIRD_CPU_THREADS`（默认 4）、`BIRD_DETECT_DEDUP`（默认 1）、`BIRD_STATE_BACKEND`（`memory|sqlite|redis`）。
`BIRD_WORKERS>=2` 时必须配 `sqlite|redis`，否则轮询落到别的 worker 会 404。

## 四、已验证性能（2026-11-11 隔离实例 A/B）

同一份代码只切环境变量，顺序 `baseline → 生产方案 → baseline 反向复测`：

| 并发档 | 基线均值 → 生产方案 | 净比值 | 平均耗时降幅 |
|---|---|---|---|
| 1 | 0.544 → 0.678 | 1.25x | -19.8% |
| 4 | 0.671 → 0.937 | 1.40x | -24.7% |
| 6 | 0.663 → 0.887 | 1.34x | -20.5% |

- 精度：9 图 × 5 字段共 45 项逐字段对比，差异 0；含贴边、焦点冲突、未检出等敏感样本。
- 中规模 2 worker + SQLite：12 次跨进程轮询 12×200 / 0×404，并发 6 完成 RPS 1.03（单 worker 0.89、基线 0.66）。
- 收益主要来自 CPU 线程配额 12→4；检测去重逐位等价、默认开启。
- GPU 路径未实测（实机无 `nvidia-smi`）。

## 五、回滚

- 代码：`git log --oneline` 找到变更前提交，`git revert <commit>` 或 `git checkout <commit> -- <path>`；本地不再保留旧副本。
- 参数：`BIRD_DETECT_DEDUP=0 BIRD_CPU_THREADS=12 BIRD_STATE_BACKEND=memory BIRD_WORKERS=1` 即退回基线行为。
- 服务：`systemctl restart bird-service`。
- 模型兜底：`/root/assets/archive/bird_service.zip`（11-09 原始交付包，含模型权重）。

## 六、数据、日志与磁盘

- 落盘：`data/jobs/YYYY/MMDD/<requestId>/{request.json,frame*.jpg,result.json}`。
- 日志：`logs/api.log` 按日切分；`logs/uvicorn.log` 由 systemd 追加。
- 清理：按 `BIRD_RETENTION_DAYS`（30）清理请求目录，按 `BIRD_LOG_RETENTION_DAYS`（30）清理日志。
