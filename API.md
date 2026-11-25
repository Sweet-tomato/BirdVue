# Bird Identification Service API

服务地址示例：`http://<server-ip>:8000`。所有接口返回 JSON，统一鉴权头：

```http
Authorization: Bearer <BIRD_API_KEY>
```

`/openapi.json`、`/docs`、`/redoc` 与业务接口共用同一把 key；`/health/*` 保持公开。

## 1. 健康检查

```http
GET /health/live
GET /health/ready
```

`ready` 返回 `{"status":"ready"}` 后再提交识别请求；模型加载中返回 503。

## 2. 异步提交（业务主线）

### 请求

```http
POST /v1/jobs
Content-Type: application/json
Authorization: Bearer <BIRD_API_KEY>
```

请求体（调用方 Java 契约；`deviceId`/`eventId`/`capturedAt` 类型不限，服务端只校验、不改写，完成态原样回显到顶层）：

```json
{
  "files": [
    "https://cdn.example.com/evt-0001/frame0.jpg",
    "https://cdn.example.com/evt-0001/frame1.jpg",
    "https://cdn.example.com/evt-0001/frame2.jpg"
  ],
  "deviceId": "BV4G266701000015",
  "eventId": 1013488,
  "capturedAt": 1794450432666
}
```

也兼容纯 URL 数组：`["https://cdn.example.com/evt-0001/frame0.jpg"]`。

- 1–3 个公网 `http/https` 图片 URL，第一个是主帧，其余是同一事件的连续帧。
- 单张 ≤10 MB，格式 JPEG/PNG/WebP。
- 服务端做 SSRF 防护：拒绝私网、环回、链路本地地址；下载失败体现在轮询结果的 `error` 字段。

#### 可选 location（按经纬度硬过滤候选池）

```json
{
  "files": ["https://cdn.example.com/evt-0001/frame0.jpg"],
  "capturedAt": "2026-05-01T08:00:00+08:00",
  "location": {"latitude": 22.5, "longitude": 114.0}
}
```

- WGS84 经纬度；`capturedAt` 建议 ISO-8601 字符串（月份权重按事件时间计算，毫秒时间戳只走全年权重）。
- 传了合法坐标且落在资产覆盖内：BioCLIP 只对 3° 网格内的本地鸟种计算候选，Top5 / margin / 复核短名单全部只来自本地鸟种；`meta.geographicPrior.hardFiltering=true`，`meta.geographicPrior.filter` 给出 `cellId` / `poolSpecies` / `maskedSpecies` / `reason`。
- 覆盖范围：中国 + 北美，473 格（东经 70–140 北纬 15–55，以及北美）。范围外、坐标非法、本地池小于 `minPoolSpecies`（默认 20）时**中性回退**，行为与不传 location 完全一致，不发 422。
- 豆包复核会收到设备坐标与“候选池已按本地鸟类过滤”的说明；若图像明显不是本地鸟，豆包可以在 `outsideCandidates` 中提出池外物种，程序校验 AviList 后仍可采纳，审计字段 `review.openSetProposal.geoFilterBypassed=true`、`meta.openSetEscape.geoFilterBypassed=true`。
- 完成态顶层保持 14 字段业务契约，`location` 不回显到顶层；生效摘要见 `meta.geographicPrior`（`enabled`/`applied`/`reason`/`filterMode`/`hardFiltering`/`filter` 等）。

#### 音频识别（独立服务）

本接口只处理图片，不接受 `audioUrls`（传入会被忽略）。听音识鸟是独立服务，见第 7 节。

```bash
curl -X POST http://<server-ip>:8000/v1/jobs \
  -H "Authorization: Bearer <BIRD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '["https://cdn.example.com/evt-0001/frame0.jpg"]'
```

### 提交响应（202）

```json
{
  "requestId": "305cf7d055524e708b0ba2cae008d4af",
  "status": "reviewing",
  "local": null,
  "resultUrl": "/v1/results/305cf7d055524e708b0ba2cae008d4af",
  "pollAfterMs": 2000
}
```

### 回调推送（可选）

提交时传 `callbackUrl`（对象体字段，或纯数组时用 `?callbackUrl=` 查询参数），任务进入终态后服务端主动 POST 最终结果，业务侧无需轮询：

```json
{
  "files": ["https://cdn.example.com/evt-0001/frame0.jpg"],
  "deviceId": "BV4G266701000015",
  "eventId": 1013488,
  "callbackUrl": "https://biz.example.com/api/bird/callback"
}
```

回调请求头：

| Header | 说明 |
|---|---|
| `X-Bird-Request-Id` | 对应 `requestId`，用于幂等去重 |
| `X-Bird-Event` | `job.done` 或 `job.error` |
| `X-Bird-Signature` | `sha256=<hex>`，对原始 body 用 `BIRD_CALLBACK_SECRET` 做 HMAC-SHA256；未配置密钥时不发送 |

- 完成态 body 与轮询完成态完全一致（顶层关联字段 + `report` 28 字段 + `meta`）；失败态 body 为 `{"requestId": "...", "status": "error", "error": "..."}`。
- 仅接受公网 `http/https` 回调地址，私网/环回地址返回 400 `invalid_callback_url`（内网回调需服务端显式开启 `BIRD_CALLBACK_ALLOW_PRIVATE=1`）。
- 失败重试：默认最多 3 次、指数退避（5s/10s），单次超时 10s；可用 `BIRD_CALLBACK_MAX_ATTEMPTS`、`BIRD_CALLBACK_TIMEOUT_SECONDS` 调整。业务侧请按 `requestId` 做幂等。
- 回调与轮询并存，任一通道拿到结果即可。

## 3. 轮询结果

```http
GET /v1/results/{requestId}
```

处理中：

```json
{"requestId": "305cf7d055524e708b0ba2cae008d4af", "status": "processing",
 "frameCount": 1, "local": {"top5": ["..."], "scene": {"route": "bird"}}, "error": null}
```

完成态（`status` 为 `ok`）：

```json
{
  "requestId": "9a2186faf8c0455ebf11081729a34e8d",
  "status": "ok",
  "deviceId": "feeder-001",
  "eventId": "probe-apipost-002",
  "capturedAt": null,
  "frameCount": 1,
  "latencyMs": 32601,
  "canIdentify": true,
  "bird": true,
  "scientificName": "Acridotheres tristis",
  "commonName": "Common Myna",
  "confidence": 0.95,
  "report": {
    "qualityScore": 100,
    "canIdentify": true,
    "describe": "Medium-sized myna perched on a building ledge…",
    "categories": "Acridotheres tristis",
    "alias": "",
    "name": "Common Myna",
    "gender": "Unknown",
    "age": "Unknown",
    "plumage": "Unknown",
    "healthStatus": 8000,
    "healthIssues": [],
    "healthDescription": "No visible signs of injury, sickness, or distress.",
    "rarity": "Unknown",
    "rarityReason": "No calibrated local abundance data was supplied.",
    "colorTags": [2006, 2007, 2002, 2009],
    "patternTags": [5007, 5013],
    "shapeTags": [3004, 3022, 3028],
    "behaviorTags": [4011],
    "flock": false,
    "bird": true,
    "person": false,
    "squirrel": false,
    "raccoon": false,
    "rattus": false,
    "snake": false,
    "opossum": false,
    "animal": true,
    "bestImageIndex": 0,
    "confidence": 0.95
  },
  "meta": {
    "decisionSource": "doubao_forced_review",
    "reviewMode": "on",
    "reviewTrigger": "forced",
    "latencyMs": 32601,
    "qualityScore": 100,
    "sceneRoute": "bird",
    "top5": ["Acridotheres tristis", "Acridotheres javanicus", "Acridotheres fuscus",
             "Acridotheres albocinctus", "Manorina melanocephala"],
    "review": {"decision": "candidate", "candidateRank": 5, "confidence": null},
    "traits": {"status": "applied", "count": 3, "clausesKept": 9,
               "clausesDroppedOffPool": 11, "slotsOmittedOffPool": 1},
    "knowledgeGraph": {"applied": false},
    "usage": {"input_tokens": 5079, "output_tokens": 2189, "total_tokens": 7268},
    "vlmStatus": "ok",
    "error": null
  }
}
```

字段说明：

- 顶层 `deviceId`/`eventId`/`capturedAt` 原样回显提交值，缺失时为 `null`；`frameCount` 为提交帧数。
- `scientificName` 取 `report.categories`，`commonName` 取 `report.name`，`confidence` 取 `report.confidence`。
- `report` 固定 28 字段，顺序即契约顺序，`bestImageIndex` 位于 `animal` 与 `confidence` 之间。
- `meta` 为复核过程信息：`decisionSource`/`reviewMode`/`reviewTrigger` 说明判定路径，`traits` 为候选性状子句统计，`geographicPrior` 为地理先验摘要（未传 location 时为 `null`），`usage` 为豆包 token 用量，`vlmStatus` 为 `ok`/`error`/`skipped`。
- 失败时 `status` 为 `error`，错误信息在 `error` 字段。

## 4. 同步识别（仅调试）

```http
POST /v1/identify
```

请求体、鉴权与 `/v1/jobs` 完全相同（对象或纯数组），返回与轮询完成态相同的结构，但会占用一次请求的连接直到识别结束（全链路约 30–40s），业务侧请使用异步接口。

## 5. 错误码

| HTTP | 含义 |
|---|---|
| 400 | 数组为空或超过 3 个 URL（`expect_1_to_3_urls`）；回调地址非法或非公网（`invalid_callback_url`） |
| 401 | 缺少或错误的 Bearer key |
| 404 | `job_not_found`，requestId 不存在或已过期 |
| 422 | JSON 解析失败或不是字符串数组 |
| 429 | 任务队列已满（64），稍后重试 |
| 503 | 服务加载中，稍后重试 |

## 6. 持久化与日志

- 任务目录：`data/jobs/YYYY/MMDD/<requestId>/`，含 `request.json`、图片与 `result.json`，保留 `BIRD_RETENTION_DAYS`（默认 30）天，每小时巡检整批清理。
- 运行日志：`logs/api.log` 按日切分，保留 30 天。
- 任务状态在内存中，进程重启会丢失未完成任务；识别结果请业务侧按 requestId 自行落库。

## 7. 音频识别服务（独立端口 8001）

听音识鸟与图片识别**完全分离**：两个独立进程、独立端口、独立模型加载，互不读取对方输入，也不合并结果。

```http
POST /v1/audio/identify
Content-Type: application/json
Authorization: Bearer <BIRD_API_KEY>
```

请求体（只接受音频，不接受图片）：

```json
{"audioUrls": ["https://cdn.example.com/evt-0001/audio0.wav"]}
```

- 恰好 1 条公网 `http/https` 音频 URL，≤20 MB、≤60 秒，格式 WAV/FLAC/OGG/MP3（AAC/M4A 需先转码）。
- BirdNET v2.4 逐 3 秒窗识别；YAMNet 先过滤人声、车流/引擎、风声/雨声、宠物叫声，再叠加谱特征门控。
- 同步返回，不进入图片任务队列：

```json
{
  "requestId": "b3f1c0d2e4a6490c8f7d6e5a4b3c2d1e",
  "status": "ok",
  "top5": [
    {"scientificName": "Pica pica", "commonName": "Eurasian Magpie",
     "confidence": 0.9527, "taxonomyMatched": true, "windows": [[0.0, 3.0]]}
  ],
  "acceptedWindows": 4,
  "totalWindows": 20,
  "acceptedRatio": 0.2,
  "noiseSummary": {},
  "noiseDominant": null,
  "model": "BirdNET_GLOBAL_6K_V2.4_Model_FP16",
  "latencyMs": 1432
}
```

- `status` 取值：`ok`（有鸟声候选）、`noise_only`（只有人声/车流/风/宠物等噪音）、`no_bird_signal`（无有效鸟声）。
- `noiseDominant` 为占主导的噪音类别：`human`/`pet`/`vehicle`/`wind_water`/`media_tools`。
- 健康检查：`GET /health/live`、`/health/ready`、`/health/state`。
- 错误码：`400 invalid_audio_url` / `audio_download_failed` / `expect_1_audio_url`，`401 unauthorized`，`503 service_loading`。
- 结果落盘：`data/audio/YYYY/MMDD/<requestId>/{request.json,audio0.*,result.json}`，按 `BIRD_RETENTION_DAYS` 清理。

```bash
curl -X POST http://<server-ip>:8001/v1/audio/identify \
  -H "Authorization: Bearer <BIRD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"audioUrls": ["https://cdn.example.com/evt-0001/audio0.wav"]}'
```

