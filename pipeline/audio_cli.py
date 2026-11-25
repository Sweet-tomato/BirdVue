"""Audio bird identification with environmental-noise gating.

Pipeline: decode -> mono 48k -> band-pass -> 3s windows -> BirdNET v2.4 +
YAMNet noise gate -> window acceptance -> species aggregation -> AviList mapping.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("bird_audio_cli")


DEFAULT_CONFIG = {
    "enabled": True,
    "modelFile": "../models/birdnet/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite",
    "labelsFile": "../models/birdnet/BirdNET_GLOBAL_6K_V2.4_Model_FP16_Labels.txt",
    "noiseModelFile": "../models/yamnet/yamnet.tflite",
    "noiseClassMapFile": "../models/yamnet/yamnet_class_map.csv",
    "noiseLabelsFile": "audio_noise_labels.json",
    "taxonomyFile": "../models/bioclip25/birds/avilist_v2025b_species.json",
    "sampleRate": 48000,
    "noiseSampleRate": 16000,
    "chunkSeconds": 3.0,
    "hopSeconds": 1.5,
    "maxSeconds": 60,
    "minBirdScore": 0.10,
    "noiseMargin": 1.2,
    "minSnrDb": 3.0,
    "noiseGateScore": 0.35,
    "topK": 5,
    "sensitivity": 1.0,
    "numThreads": 2,
}


def _resolve(config: dict, key: str) -> Path:
    path = Path(config[key])
    if path.is_absolute():
        return path
    base = Path(config.get("_path") or (Path(__file__).resolve().parent.parent / "config" / "pipeline_config.json")).parent
    return (base / path).resolve()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples
    from math import gcd

    factor = gcd(source_rate, target_rate)
    from scipy.signal import resample_poly

    return resample_poly(samples, target_rate // factor, source_rate // factor).astype(np.float32)


def _band_pass(samples: np.ndarray, rate: int) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, [100.0, 12000.0], btype="bandpass", fs=rate, output="sos")
    return sosfiltfilt(sos, samples).astype(np.float32)


def _frame_features(window: np.ndarray, rate: int) -> dict:
    """Cheap spectral statistics used for gating only."""
    size = 1024
    count = max(1, (len(window) - size) // (size // 2) + 1)
    frames = np.lib.stride_tricks.as_strided(
        window,
        shape=(count, size),
        strides=(window.strides[0] * (size // 2), window.strides[0]),
        writeable=False,
    )
    spec = np.abs(np.fft.rfft(frames * np.hanning(size), axis=1)) + 1e-10
    freqs = np.fft.rfftfreq(size, 1.0 / rate)
    power = spec**2
    total = float(power.sum())
    flatness = float(np.exp(np.mean(np.log(spec), axis=1)).mean() / (np.mean(spec, axis=1).mean() + 1e-10))
    low_ratio = float(power[:, freqs < 500.0].sum() / (total + 1e-10))
    frame_rms = np.sqrt(np.mean(frames**2, axis=1)) + 1e-10
    noise_floor = float(np.percentile(frame_rms, 10))
    snr_db = 20.0 * math.log10(float(np.percentile(frame_rms, 90)) / noise_floor + 1e-10)
    # Harmonicity: normalized autocorrelation peak in the 300-3000 Hz lag range.
    signal = window - window.mean()
    fft_size = 1 << int(np.ceil(np.log2(2 * len(signal) - 1)))
    spectrum = np.fft.rfft(signal, fft_size)
    autocorr = np.fft.irfft(spectrum * np.conj(spectrum), fft_size)[: len(signal)]
    autocorr = autocorr / (autocorr[0] + 1e-10)
    low_lag = max(1, int(rate / 3000.0))
    high_lag = min(len(autocorr) - 1, int(rate / 300.0))
    harmonicity = float(np.max(autocorr[low_lag:high_lag + 1])) if high_lag > low_lag else 0.0
    return {
        "flatness": round(flatness, 4),
        "lowBandRatio": round(low_ratio, 4),
        "snrDb": round(snr_db, 2),
        "harmonicity": round(harmonicity, 4),
    }


class BirdNetRecognizer:
    def __init__(self, model_path: Path, labels_path: Path, sensitivity: float, threads: int):
        from ai_edge_litert.interpreter import Interpreter

        self.labels = [line.strip() for line in labels_path.read_text().splitlines() if line.strip()]
        self.interpreter = Interpreter(model_path=str(model_path), num_threads=threads)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]
        self.expected_samples = int(self.interpreter.get_input_details()[0]["shape"][-1])
        self.sensitivity = float(sensitivity)
        self.lock = threading.Lock()

    def predict(self, window: np.ndarray) -> np.ndarray:
        data = window[: self.expected_samples]
        if len(data) < self.expected_samples:
            data = np.pad(data, (0, self.expected_samples - len(data)))
        data = data.astype(np.float32).reshape(1, -1)
        with self.lock:
            self.interpreter.set_tensor(self.input_index, data)
            self.interpreter.invoke()
            logits = np.array(self.interpreter.get_tensor(self.output_index)[0], dtype=np.float32)
        return _sigmoid(self.sensitivity * logits)


class YamnetNoiseTagger:
    def __init__(self, model_path: Path, class_map_path: Path, categories: dict, threads: int):
        from ai_edge_litert.interpreter import Interpreter

        rows = list(csv.DictReader(class_map_path.open(encoding="utf-8")))
        names = [row["display_name"].strip() for row in rows]
        lookup = {}
        for index, name in enumerate(names):
            lookup.setdefault(name.casefold(), index)
        self.categories = {}
        for category, wanted in categories.items():
            indices = []
            for name in wanted:
                index = lookup.get(name.casefold())
                if index is not None:
                    indices.append(index)
            self.categories[category] = indices
        self.interpreter = Interpreter(model_path=str(model_path), num_threads=threads)
        self.interpreter.allocate_tensors()
        details = self.interpreter.get_input_details()[0]
        self.input_index = details["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]
        self.expected_samples = int(details["shape"][-1])
        self.lock = threading.Lock()

    def score(self, samples: np.ndarray, hop: int) -> list[dict]:
        results = []
        for start in range(0, max(1, len(samples) - self.expected_samples + 1), hop):
            chunk = samples[start : start + self.expected_samples]
            if len(chunk) < self.expected_samples:
                chunk = np.pad(chunk, (0, self.expected_samples - len(chunk)))
            with self.lock:
                self.interpreter.set_tensor(self.input_index, chunk.astype(np.float32))
                self.interpreter.invoke()
                scores = np.array(self.interpreter.get_tensor(self.output_index)[0], dtype=np.float32)
            categories = {
                name: round(float(np.max(scores[idx])) if idx else 0.0, 4)
                for name, idx in self.categories.items()
            }
            results.append({"start": start, "categories": categories})
        return results


class AudioEngine:
    def __init__(self, config: dict, recognizer: BirdNetRecognizer, noise: YamnetNoiseTagger | None, taxonomy: dict):
        self.config = config
        self.recognizer = recognizer
        self.noise = noise
        self.taxonomy = taxonomy

    def analyze(self, path: Path) -> dict:
        started = time.perf_counter()
        config = self.config
        rate = int(config["sampleRate"])
        import soundfile as sf

        data, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        max_samples = int(config["maxSeconds"]) * source_rate
        if len(mono) > max_samples:
            mono = mono[:max_samples]
        samples = _resample(mono, int(source_rate), rate)
        duration = len(samples) / rate
        if duration < 0.5:
            return self._result("no_bird_signal", duration, 0, 0, [], {}, started)
        filtered = _band_pass(samples, rate)

        chunk = int(config["chunkSeconds"] * rate)
        hop = max(1, int(config["hopSeconds"] * rate))
        windows = []
        index = 0
        while index + int(hop * 0.5) <= len(filtered):
            piece = filtered[index : index + chunk]
            if len(piece) < chunk:
                piece = np.pad(piece, (0, chunk - len(piece)))
            windows.append({"start": index / rate, "end": min(duration, (index + chunk) / rate), "audio": piece})
            index += hop

        noise_windows = []
        if self.noise is not None:
            noise_rate = int(config["noiseSampleRate"])
            noise_samples = _resample(samples, rate, noise_rate)
            noise_hop = max(1, int(self.noise.expected_samples / 2))
            for entry in self.noise.score(noise_samples, noise_hop):
                center = (entry["start"] + self.noise.expected_samples / 2) / noise_rate
                noise_windows.append({"center": center, "categories": entry["categories"]})

        accepted = []
        noise_summary = {}
        for window in windows:
            mid = (window["start"] + window["end"]) / 2.0
            categories = {}
            if noise_windows:
                nearest = min(noise_windows, key=lambda item: abs(item["center"] - mid))
                categories = nearest["categories"]
            features = _frame_features(window["audio"], rate)
            scores = self.recognizer.predict(window["audio"])
            bird_score = float(np.max(scores))
            noise_score = max(categories.values(), default=0.0)
            dominant = max(categories, key=categories.get) if categories else None
            accepted_window = self._accept(bird_score, noise_score, features)
            window.update({
                "birdScore": bird_score,
                "noiseScore": noise_score,
                "noiseCategory": dominant,
                "features": features,
                "accepted": accepted_window,
                "scores": scores if accepted_window else None,
            })
            if accepted_window:
                accepted.append(window)
            for name, value in categories.items():
                noise_summary[name] = max(noise_summary.get(name, 0.0), value)

        total = len(windows)
        if not accepted:
            status = "noise_only" if max(noise_summary.values(), default=0.0) >= config["noiseGateScore"] else "no_bird_signal"
            return self._result(
                status, duration, total, 0, [], noise_summary, started,
                windows=[self._window_debug(w) for w in windows],
            )

        best = np.zeros(len(self.recognizer.labels), dtype=np.float32)
        hit_windows: dict[int, list[list[float]]] = {}
        for window in accepted:
            scores = window["scores"]
            best = np.where(scores > best, scores, best)
            for index in np.nonzero(scores >= config["minBirdScore"])[0]:
                hit_windows.setdefault(int(index), []).append([round(window["start"], 2), round(window["end"], 2)])
        top_k = int(config["topK"])
        order = np.argsort(best)[-top_k:][::-1]
        top = []
        for index in order:
            score = float(best[index])
            if score < config["minBirdScore"]:
                continue
            label = self.recognizer.labels[index]
            record = self.taxonomy.get(label.casefold())
            top.append({
                "scientificName": label,
                "commonName": record["commonName"] if record else label,
                "confidence": round(score, 4),
                "stableId": record.get("stableId") if record else None,
                "family": record.get("family") if record else None,
                "genus": record.get("genus") if record else None,
                "taxonomyMatched": bool(record),
                "windows": hit_windows.get(int(index), []),
            })
        return self._result(
            "ok", duration, total, len(accepted), top, noise_summary,
            started, windows=[self._window_debug(w) for w in windows],
        )

    def _accept(self, bird_score: float, noise_score: float, features: dict) -> bool:
        config = self.config
        if bird_score < config["minBirdScore"]:
            return False
        if noise_score >= config["noiseGateScore"] and bird_score < config["noiseMargin"] * noise_score:
            return False
        if features["flatness"] > 0.55 and features["harmonicity"] < 0.2:
            return False
        if features["lowBandRatio"] > 0.85:
            return False
        if features["snrDb"] < config["minSnrDb"]:
            return False
        return True

    @staticmethod
    def _window_debug(window: dict) -> dict:
        return {
            "start": round(window["start"], 2),
            "end": round(window["end"], 2),
            "accepted": window["accepted"],
            "birdScore": round(window["birdScore"], 4),
            "noiseScore": round(window["noiseScore"], 4),
            "noiseCategory": window["noiseCategory"],
            "features": window["features"],
        }

    def _result(self, status, duration, total, accepted_count, top, noise_summary, started, windows=None):
        result = {
            "status": status,
            "model": "BirdNET_GLOBAL_6K_V2.4_Model_FP16",
            "durationSeconds": round(duration, 2),
            "totalWindows": total,
            "acceptedWindows": accepted_count,
            "acceptedRatio": round(accepted_count / total, 4) if total else 0.0,
            "top5": top,
            "noiseSummary": {k: round(v, 4) for k, v in sorted(noise_summary.items(), key=lambda item: -item[1]) if v >= 0.05},
            "noiseDominant": max(noise_summary, key=noise_summary.get) if noise_summary and max(noise_summary.values()) >= 0.05 else None,
            "latencyMs": int((time.perf_counter() - started) * 1000),
        }
        if windows is not None and status in {"noise_only", "no_bird_signal"}:
            result["windows"] = windows
        return result


def _load_taxonomy(path: Path) -> dict:
    data = json.loads(path.read_text())
    lookup = {}
    for record in data.get("species", []):
        entry = {
            "commonName": record.get("commonName"),
            "stableId": record.get("stableId"),
            "family": record.get("family"),
            "genus": record.get("genus"),
        }
        for name in (record.get("scientificName"), record.get("modelScientificName")):
            if name:
                lookup.setdefault(name.casefold(), entry)
    return lookup


def load_engine(config: dict) -> AudioEngine:
    merged = dict(DEFAULT_CONFIG)
    merged.update(config or {})
    recognizer = BirdNetRecognizer(
        _resolve(merged, "modelFile"),
        _resolve(merged, "labelsFile"),
        merged["sensitivity"],
        int(merged["numThreads"]),
    )
    taxonomy = _load_taxonomy(_resolve(merged, "taxonomyFile"))
    noise = None
    try:
        categories = json.loads(_resolve(merged, "noiseLabelsFile").read_text())
        noise = YamnetNoiseTagger(
            _resolve(merged, "noiseModelFile"),
            _resolve(merged, "noiseClassMapFile"),
            categories,
            int(merged["numThreads"]),
        )
    except Exception as error:  # noise gate is optional; BirdNET still runs
        log.warning("yamnet_noise_gate_unavailable error=%s", error)
        noise = None
    return AudioEngine(merged, recognizer, noise, taxonomy)
