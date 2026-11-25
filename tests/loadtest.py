#!/usr/bin/env python3
"""bird_service 压测脚本（仅标准库，直接对本地 uvicorn 发压）。

模式:
  health        GET  /health/ready          （服务基础处理能力）
  jobs-submit   POST /v1/jobs               （异步提交，观察 202/429 与提交时延）
  jobs-e2e      POST /v1/jobs + 轮询结果     （端到端时延：提交→完成）
  identify      POST /v1/identify           （同步识别，重负载）

用法:
  python3 tests/loadtest.py --mode health --concurrency 10 --duration 20 --outdir logs/lt_health
  python3 tests/loadtest.py --mode jobs-e2e --count 20 --concurrency 2 --outdir logs/lt_e2e
"""
import argparse, http.client, json, os, sys, threading, time, uuid
from pathlib import Path

HOST, PORT = "127.0.0.1", 8000


def build_multipart(image_bytes, filename="frame.jpg"):
    boundary = "----birdload" + uuid.uuid4().hex
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"deviceId\"\r\n\r\nloadtest\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"eventId\"\r\n\r\n{uuid.uuid4().hex}\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"capturedAt\"\r\n\r\n{time.strftime('%Y-%m-%dT%H:%M:%S')}\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"{filename}\"\r\nContent-Type: image/jpeg\r\n\r\n".encode() + image_bytes + b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class Ctx:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.results = []
        self.samples = []
        self.deadline = time.time() + args.duration if args.duration else None
        self.remaining = args.count
        self.body = None
        self.ctype = None
        if args.mode != "health":
            img = Path(args.image).read_bytes()
            self.body, self.ctype = build_multipart(img)

    def take(self):
        if self.deadline is not None:
            return time.time() < self.deadline
        with self.lock:
            if self.remaining > 0:
                self.remaining -= 1
                return True
            return False

    def record(self, row):
        with self.lock:
            self.results.append(row)


def request(conn, method, path, body=None, ctype=None, timeout=30):
    t0 = time.perf_counter()
    headers = {"Content-Type": ctype, "Connection": "keep-alive"} if ctype else {"Connection": "keep-alive"}
    conn.request(method, path, body=body, headers=headers)
    resp = conn.getresponse()
    data = resp.read()
    ms = (time.perf_counter() - t0) * 1000
    return resp.status, data, ms


def worker(ctx, tid):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=ctx.args.timeout)
    while ctx.take():
        row = {"ts": round(time.time(), 3), "thread": tid}
        try:
            if ctx.args.mode == "health":
                st, _, ms = request(conn, "GET", "/health/ready")
                row.update(status=st, ms=round(ms, 1))
            elif ctx.args.mode == "jobs-submit":
                st, data, ms = request(conn, "POST", "/v1/jobs", ctx.body, ctx.ctype)
                row.update(status=st, ms=round(ms, 1))
                if st == 202:
                    row["requestId"] = json.loads(data).get("requestId")
            elif ctx.args.mode == "identify":
                st, _, ms = request(conn, "POST", "/v1/identify", ctx.body, ctx.ctype, timeout=ctx.args.timeout)
                row.update(status=st, ms=round(ms, 1))
            else:  # jobs-e2e
                st, data, ms = request(conn, "POST", "/v1/jobs", ctx.body, ctx.ctype)
                row.update(status=st, submit_ms=round(ms, 1))
                if st == 202:
                    rid = json.loads(data)["requestId"]
                    row["requestId"] = rid
                    t0 = time.perf_counter()
                    final, polls = "timeout", 0
                    while time.perf_counter() - t0 < ctx.args.timeout:
                        time.sleep(ctx.args.poll)
                        polls += 1
                        s2, d2, _ = request(conn, "GET", f"/v1/results/{rid}")
                        if s2 != 200:
                            final = f"poll_{s2}"
                            break
                        body = json.loads(d2)
                        if body.get("status") in ("done", "error", "failed"):
                            final = body["status"]
                            break
                    row.update(ms=round((time.perf_counter() - t0) * 1000, 1), final=final, polls=polls)
        except Exception as exc:  # noqa: BLE001
            row.update(status=-1, error=f"{type(exc).__name__}: {exc}", ms=None)
            try:
                conn.close()
            except Exception:
                pass
            conn = http.client.HTTPConnection(HOST, PORT, timeout=ctx.args.timeout)
        ctx.record(row)


def sampler(ctx, stop):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=5)
    while not stop.is_set():
        try:
            _, d, ms = request(conn, "GET", "/health/state")
            state = json.loads(d)
            ctx.samples.append({"ts": round(time.time(), 3), "health_ms": round(ms, 1),
                                "jobs": state["store"].get("jobs"), "ready": True})
        except Exception as exc:  # noqa: BLE001
            ctx.samples.append({"ts": round(time.time(), 3), "error": str(exc), "ready": False})
        stop.wait(2)


def pct(values, p):
    if not values:
        return None
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(round((p / 100) * (len(values) - 1)))))
    return round(values[k], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["health", "jobs-submit", "jobs-e2e", "identify"])
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--duration", type=float, default=0)
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--image", default="data/loadtest/assets/real_a.jpg")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    if not args.duration and not args.count:
        sys.exit("必须指定 --duration 或 --count")

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(args)
    stop = threading.Event()
    t = threading.Thread(target=sampler, args=(ctx, stop), daemon=True)
    t.start()
    started = time.time()
    threads = [threading.Thread(target=worker, args=(ctx, i)) for i in range(args.concurrency)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    elapsed = time.time() - started
    stop.set()
    t.join(timeout=3)

    with (out / "requests.jsonl").open("w", encoding="utf-8") as fh:
        for row in ctx.results:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "samples.jsonl").open("w", encoding="utf-8") as fh:
        for row in ctx.samples:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(ctx.results)
    ok = [r for r in ctx.results if r.get("status") in (200, 202, "done")]
    lat = [r["ms"] for r in ok if r.get("ms") is not None]
    by_status = {}
    for r in ctx.results:
        key = str(r.get("final") or r.get("status"))
        by_status[key] = by_status.get(key, 0) + 1
    summary = {
        "mode": args.mode, "concurrency": args.concurrency, "elapsed_s": round(elapsed, 1),
        "requests": total, "ok": len(ok), "error_rate": round((total - len(ok)) / total, 4) if total else None,
        "rps": round(total / elapsed, 2) if elapsed else None,
        "latency_ms": {"p50": pct(lat, 50), "p90": pct(lat, 90), "p95": pct(lat, 95), "p99": pct(lat, 99),
                       "max": max(lat) if lat else None},
        "status_breakdown": by_status,
        "server_jobs_max": max([s.get("jobs") or 0 for s in ctx.samples], default=0),
        "server_jobs_last": (ctx.samples[-1].get("jobs") if ctx.samples else None),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
