# Bird Identification Service

图片识别（BioCLIP 2.5 + 豆包一次性复核）与听音识鸟（BirdNET + YAMNet）两个独立服务。

## 目录

| 路径 | 说明 |
|---|---|
| `api/bird_frames_service.py` | 图片识别接口，默认 `:8000` |
| `api/bird_audio_service.py` | 音频识别接口，默认 `:8001` |
| `pipeline/` | BioCLIP 推理、候选融合、性状、豆包复核、风险门、音频链路 |
| `models/bioclip-2.5-vit-h14/` | BioCLIP 2.5 ViT-H-14 权重（FP32 safetensors，686 张量），SHA256 `ac2e37c2f89ef8e6b889176a9a3f418970ad9db15a218bd29e3321e95c46ae97` |
| `models/bioclip25/birds/` | AviList v2025b 候选向量、taxonomy、性状包 v5 |
| `models/` 其余 | YOLOX 检测、ONNX 编码器、GPS 网格资产、BirdNET/YAMNet |
| `config/pipeline_config.json` | 阈值、开关、相对路径 |
| `config/service.env` | 运行时密钥与参数，权限 600，不入库 |
| `STATE_MACHINE.md` | 识别流水线状态机（确定性 DAG，非 Agent） |
| `API.md` | 接口契约与错误码 |
| `PRODUCTION.md` | 生产运行手册 |
| `CHANGELOG.md` | 迭代记录（历史版本不保留本地副本） |

## 启动

生产由 systemd 守护：

```bash
systemctl status bird-service
```

手工方式：

```bash
bash start_service.sh   # 图片，:8000
bash start_audio.sh     # 音频，:8001
```

日常运维用 `./birdctl log|status|restart`；systemd 运行中不要再用 birdctl 启动，避免双开。

配置放 `config/service.env`（模板 `config/service.env.example`），启动脚本自动加载。

## 接口

- 同步：`POST /v1/identify`（multipart，1–3 张 JPEG/PNG/WebP）
- 异步：`POST /v1/jobs` → 202，`GET /v1/results/{requestId}`
- 音频：`POST /v1/audio/identify`（独立端口 8001，只接受音频 URL）
- 健康：`GET /health/live`、`/health/ready`、`/health/state`
- 鉴权：除 `/health*` 外需 `Authorization: Bearer $BIRD_API_KEY`

## 当前行为（2026-11-24）

- 单链路：每个事件只调用一次豆包；`openSetEscape.enabled=true`，在 Top5 外注入 1 个候选，不二次复核。
- 本地直通：`margin(top1-top2) ≥ 0.05` 且无阻塞项时 `local_direct`；质量分门禁已废弃，`qualityScore` 仅作观测。
- 地理先验：设备传 `location`（WGS84）时按 3° 网格**硬过滤候选池**（中国+北美 473 格），Top5 只排本地鸟种；未传、覆盖范围外或本地池 <20 种时中性回退为全量候选。豆包复核能看到坐标与过滤说明，若图像明显不是本地鸟（屏幕/远程画面、迷鸟）仍可在 `outsideCandidates` 提案，采纳后审计 `geoFilterBypassed`。
- 音频：独立服务，图片链路不读音频；当前未部署 systemd unit。
- 持久化：`data/jobs/YYYY/MMDD/<requestId>/`，按 `BIRD_RETENTION_DAYS`（默认 30）清理。

## 硬件现实

当前生产为 CPU-only（无 GPU），单事件约 20–45 s。GPU（L4/A10/4090）方案未实测。

## 编码器

`BIRD_ENCODER=torch`（默认）| `onnx_fp32` | `onnx_int8`；ONNX 分片在 `models/onnx/e19`，可用 `BIRD_ONNX_DIR` 覆盖。
`onnx_fp32` 与 torch Top5 一致；`onnx_int8` 更快但 Top5 可能变化。

## 未包含

Docker 部署、多 GPU、知识图谱/识别卡片/图像原型（均关闭）、事件存储与鉴权中心。
