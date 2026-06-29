#!/usr/bin/env python3 -u
"""
modelfactory job launcher for smoke test.
Submit this as:  代码解释器=python, 代码路径=.../scripts/smoke_job_launcher.py
"""
import subprocess
import sys
import os

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "smoke_train_with_services.sh")

print(f"=== Launcher: running {SCRIPT} ===", flush=True)
print(f"    CWD={os.getcwd()}", flush=True)
print(f"    SELF={__file__}", flush=True)

proc = subprocess.Popen(
    ["bash", SCRIPT],
    stdout=sys.stdout,
    stderr=sys.stderr,
    env=os.environ,
)
sys.exit(proc.wait())
