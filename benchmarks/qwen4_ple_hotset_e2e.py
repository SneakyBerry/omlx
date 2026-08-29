#!/usr/bin/env python3
"""Cold-prefill/decode A/B harness for the Qwen4 exp PLE hot-set.

Protocol (from mlx-community/qwen4_exp#3235): config edit, purge, fresh
server per trial, real long prompt, alternating arms, identical prompts.

Arm "off": ple_hot_set_bytes=0 (bare mmap). Arm "on": --hot-bytes.
Run while the machine is idle. Paste the JSONL + summary back to the PR.
"""

import argparse
import datetime
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

API_HEADERS = {}
REPO_VENV = [sys.executable, "-m", "omlx.cli"]


def http_get(url, timeout=10):
    req = urllib.request.Request(url, headers=API_HEADERS)
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def set_arm(model_key, hot_bytes):
    path = Path.home() / ".omlx" / "model_settings.json"
    data = json.loads(path.read_text())
    models = data.get("models", data)
    models[model_key]["qwen4_ple_ssd_offload"] = True
    models[model_key]["ple_hot_set_bytes"] = hot_bytes
    path.write_text(json.dumps(data, indent=2))


def start_server(model_dir, port, log_path):
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(
            [*REPO_VENV, "serve", "--model-dir", str(model_dir),
             "--port", str(port)],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    deadline = time.time() + 600
    while time.time() < deadline:
        try:
            if http_get(f"http://127.0.0.1:{port}/health").get("status") in (
                "ok", "healthy",
            ):
                return proc
        except Exception:
            time.sleep(2)
    raise RuntimeError("server did not become healthy in 600s")


def stop_server(proc):
    subprocess.run(["pkill", "-TERM", "-P", str(proc.pid)], check=False)
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    subprocess.run(["pkill", "-f", "omlx serve"], check=False)
    time.sleep(5)


def run_request(port, model, prompt, max_tokens):
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **API_HEADERS},
    )
    t0 = time.perf_counter()
    t_first = t_last = None
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode(errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            text = (chunk.get("choices") or [{}])[0].get("text") or ""
            if text:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
    completion_tokens = usage.get("completion_tokens")
    decode = (
        (completion_tokens - 1) / (t_last - t_first)
        if completion_tokens and t_last and t_last > t_first
        else 0.0
    )
    return {
        "ttft_s": round(t_first - t0, 3) if t_first else None,
        "decode_tps": round(decode, 2),
        "total_s": round(time.perf_counter() - t0, 2),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion_tokens,
    }


def rss_kib(pid):
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True
    )
    return int(out.stdout.strip() or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--prompt-tokens", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--hot-bytes", type=int, default=536870912)
    ap.add_argument("--arm", choices=["both", "off", "on"], default="both")
    ap.add_argument("--api-key", default="")
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=Path.home() / ".omlx" / "models",
    )
    ap.add_argument("--model-key", default="Qwen3.8-Flash-Next-oQ4e-mtp")
    args = ap.parse_args()

    global API_HEADERS
    API_HEADERS = (
        {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    )

    passage = (
        "The steady state of a cache is defined not by its size but by the "
        "traffic that flows through it, and traffic remembers where it has been. "
    )
    prompt = (passage * (args.prompt_tokens * 4 // len(passage) + 1))[
        : args.prompt_tokens * 4
    ]

    stamp = datetime.date.today().isoformat()
    out_path = Path(__file__).with_name(f"ple_hotset_ab_{stamp}.jsonl")
    log_dir = Path("/tmp/omlx_ab_logs")
    log_dir.mkdir(exist_ok=True)

    subprocess.run(["sudo", "-v"], check=True)
    subprocess.run(["pkill", "-f", "omlx serve"], check=False)
    time.sleep(3)

    results = []
    try:
        for trial in range(1, args.trials + 1):
            arm = (
                ("off" if trial % 2 else "on")
                if args.arm == "both"
                else args.arm
            )
            hot_bytes = 0 if arm == "off" else args.hot_bytes
            set_arm(args.model_key, hot_bytes)
            subprocess.run(["sudo", "purge"], check=True)
            time.sleep(2)
            proc = start_server(
                args.model_dir, args.port,
                log_dir / f"ab_{arm}_t{trial}.log",
            )
            rec = {"arm": arm, "trial": trial, "hot_bytes": hot_bytes}
            try:
                t_load0 = time.perf_counter()
                rec["warmup"] = run_request(
                    args.port, args.model_key, "Hello", 8
                )
                rec["warmup_s"] = round(time.perf_counter() - t_load0, 2)
                rec["measured"] = run_request(
                    args.port, args.model_key, prompt, args.max_tokens
                )
                rec["server_rss_mib"] = round(rss_kib(proc.pid) / 1024)
                vm = subprocess.run(
                    ["vm_stat"], capture_output=True, text=True
                ).stdout
                free_pages = [
                    int(line.split(":")[1].strip().rstrip("."))
                    for line in vm.splitlines()
                    if "Pages free" in line
                ]
                rec["free_ram_mib"] = round(free_pages[0] * 4096 / 1048576)
            finally:
                stop_server(proc)
            results.append(rec)
            with out_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
    finally:
        subprocess.run(["pkill", "-f", "omlx serve"], check=False)

    print(f"\nwrote {out_path}")
    for arm in ("off", "on"):
        rows = [r for r in results if r["arm"] == arm and "measured" in r]
        if not rows:
            continue

        def med(k, rows=rows):
            return sorted(r["measured"][k] for r in rows)[len(rows) // 2]
        print(
            f"{arm:4s} n={len(rows)} ttft_med={med('ttft_s')}s "
            f"decode_med={med('decode_tps')}tok/s "
            f"load_med={sorted(r['warmup_s'] for r in rows)[len(rows)//2]}s"
        )


if __name__ == "__main__":
    main()
