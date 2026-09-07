"""Pull GRPO run artifacts from the Cynaptics node: WEIGHTS FIRST, then reports.

Mirrors scripts/pull_grpo_now.ps1 but uses the same paramiko auth that works on
this box (ssh.exe has no askpass). Polls the node until ship/<OUT_NAME>_*.tar.gz
exists and its size is stable across 3 checks (>= 60s), then SFTP-downloads it
to the 'pulled 2' directory immediately. After weights are safe, waits for the
judge/dev reports and downloads everything else, including the PDF.
"""
import hashlib
import os
import sys
import time
from collections import deque

import paramiko

HOST, PORT, USER, PW = "10.214.5.55", 22013, "gpu17", "SLA8ZkUFgvAi4YbByESu"
SHIP = "/home/gpu17/ToolCall/artifacts/ship"
DEST = os.path.join(r"E:\Multi modal reasoning tool\atr", "scripts", "pulled 2")
OUT_NAME = os.environ.get("OUT_NAME", "grpo-real")
MIN_BYTES = 5 * 1024 * 1024
STABLE_POLLS = 3
POLL_SLEEP = 20
MAX_AGES = int(os.environ.get("MAX_WAIT_MIN", "240"))

os.makedirs(DEST, exist_ok=True)


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30,
              banner_timeout=30, auth_timeout=30)
    return c


def ssh(c, cmd, timeout=60):
    _, out, err = c.exec_command(cmd, timeout=timeout)
    code = out.channel.recv_exit_status()
    o = out.read().decode("utf-8", "replace")
    e = err.read().decode("utf-8", "replace")
    return code, o.strip(), e.strip()


def newest_tarball(c):
    code, o, _ = ssh(c, f"ls -t {SHIP}/{OUT_NAME}_*.tar.gz 2>/dev/null | head -1")
    if code != 0 or not o:
        return None
    return o.splitlines()[-1].strip()


def remote_size(c, path):
    code, o, _ = ssh(c, f"stat -c %s '{path}' 2>/dev/null")
    if code != 0 or not o:
        return -1
    try:
        return int(o.strip())
    except ValueError:
        return -1


def wait_for_weights(c, deadline):
    print(f"[pull] waiting for weights ship/{OUT_NAME}_*.tar.gz ...")
    last_size, stable = -1, 0
    while time.monotonic() < deadline:
        target = newest_tarball(c)
        if target:
            sz = remote_size(c, target)
            if sz > MIN_BYTES:
                if sz == last_size:
                    stable += 1
                else:
                    stable = 0
                print(f"[pull]   {target}: {sz} bytes (stable x{stable})")
                last_size = sz
                if stable >= STABLE_POLLS:
                    return target
            else:
                stable = 0
        time.sleep(POLL_SLEEP)
    raise TimeoutError(f"weights tarball did not stabilise within "
                       f"{MAX_AGES} min (last saw {last_size} bytes)")


def download(c, remote_path, local_path):
    sftp = c.open_sftp()
    sftp.get(remote_path, local_path)
    sftp.close()
    size = os.path.getsize(local_path)
    sha = hashlib.sha256(open(local_path, "rb").read()).hexdigest()
    print(f"[pull]   got {local_path} ({size} bytes) sha256={sha[:16]}...")
    return sha


def wait_for_reports(c, deadline):
    print("[pull] weights safe locally. Waiting for judge/dev reports ...")
    while time.monotonic() < deadline:
        _, d1, _ = ssh(c, f"ls {SHIP}/dev-report_*.json 2>/dev/null | head -1")
        _, d2, _ = ssh(c, f"ls {SHIP}/judge-scores_*.jsonl 2>/dev/null | head -1")
        if d1 and d2:
            print("[pull]   reports present on node.")
            return
        time.sleep(30)
    print("[pull]   reports not ready within window; pulling what exists.")


def main():
    c = connect()
    try:
        deadline = time.monotonic() + MAX_AGES * 60

        target = wait_for_weights(c, deadline)
        name = os.path.basename(target)
        local = os.path.join(DEST, name)
        download(c, target, local)
        print(f"[pull] === WEIGHTS PULLED FIRST: {local}")
        if os.environ.get("PULL_ONLY_WEIGHTS") == "1":
            print("[pull] PULL_ONLY_WEIGHTS set - done.")
            return 0

        wait_for_reports(c, deadline)
        pats = ["eval-dev_*.tar.gz", "eval-judge_*.tar.gz",
                "dev-report_*.txt", "dev-report_*.json",
                "judge-report_*.txt", "judge-scores_*.jsonl",
                "ATR-Eval-Report_*.pdf", "MANIFEST_*.txt"]
        for pat in pats:
            _, o, _ = ssh(c, f"ls -t {SHIP}/{pat} 2>/dev/null | head -1")
            if o:
                rpath = o.splitlines()[-1].strip()
                if os.path.basename(rpath) == os.path.basename(target):
                    continue
                download(c, rpath, os.path.join(DEST, os.path.basename(rpath)))
        print(f"[pull] done. Check '{DEST}'")
    finally:
        c.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())