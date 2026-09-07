#!/usr/bin/env python3
"""STOP the running GRPO at the CURRENT step and let the pipeline archive+eval.

Usage:  python scripts/stop_grpo_now.py [STOP_PATH]
        STOP_PATH defaults to artifacts/grpo-real/STOP (repo root relative).

Connects to the Cynaptics node, touches $REPO/artifacts/grpo-real/STOP. The
running trainer checks that file at the top of every step, saves a full
checkpoint at the current step, exits cleanly, and the pipeline proceeds to
archive weights -> evals -> reports. The local puller grabs the weights first.

Self-contained (paramiko only): the password below is the CURRENT reservation
password - re-check when the reservation changes.
"""
import getpass
import os
import sys

import paramiko

HOST, PORT, USER = "10.214.5.55", 22013, "gpu17"

# Same credential source as the local helper scripts in the package dir.
def _password() -> str:
    return os.environ.get("ATR_SSH_PW") or "SLA8ZkUFgvAi4YbByESu"


def ssh(cmd: str, timeout: int = 120) -> tuple[str, str]:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        c.connect(HOST, port=PORT, username=USER, password=_password(), timeout=30)
        _, o, e = c.exec_command(cmd, timeout=timeout)
        return o.read().decode("utf-8", "replace"), e.read().decode("utf-8", "replace")
    finally:
        c.close()


def main() -> None:
    stop = sys.argv[1] if len(sys.argv) > 1 else "artifacts/grpo-real/STOP"
    if not stop.startswith("/"):
        stop = f"/home/gpu17/ToolCall/{stop}"
    out, err = ssh(f"mkdir -p \"$(dirname {stop})\" && touch {stop} && ls -la {stop}")
    print(out)
    if "STOP" in out:
        print(f"STOP signal set at {stop}. Trainer saves the current step and exits.")
    else:
        print("WARN: did not confirm STOP file. stderr:", err, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()