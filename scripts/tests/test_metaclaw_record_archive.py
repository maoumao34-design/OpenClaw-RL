"""The record must survive a purge, because a finished run has to be auditable.

`purge_record_files()` truncates the record on every `pause_submission`, i.e.
on every train step. That keeps the live file scoped to the rollout in flight,
which is what the official code wants -- but it also means a finished run
leaves nothing behind.

That is not hypothetical. On 2026-09-14 the trajectory splice was missing every
earlier turn, and the bytes that would have shown why were already gone; the
mechanism was only recovered because a snapshot had been taken mid-run by luck.
docs/change_ledger.md asks that every run be accounted for afterwards, and that
rule cannot be followed against a file that deletes itself while the run is
still going.

So the patch archives before truncating. This test runs the generated
`purge_record_files` against real files on disk and checks the two properties
that matter: the live file really is emptied (the official behaviour is
preserved), and nothing is lost (the archive accumulates across repeated
purges, in order).

Usage:
    OPENCLAW_RL_OFFICIAL=<path> python scripts/tests/test_metaclaw_record_archive.py
"""

import os
import re
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS_DIR)
OFFICIAL = os.environ.get(
    "OPENCLAW_RL_OFFICIAL", os.path.join(os.path.dirname(ROOT), "OpenClaw-RL-official")
)


class StubLogger:
    def __init__(self):
        self.messages = []

    def info(self, fmt, *a):
        self.messages.append(fmt % a if a else fmt)

    warning = info
    error = info


def extract_purge(source):
    m = re.search(
        r"^(    def purge_record_files\(self\):.*?)(?=^    (?:@|async |def ))",
        source, re.S | re.M,
    )
    if not m:
        raise AssertionError("could not find purge_record_files in the generated server")
    ns = {"os": os, "logger": StubLogger()}
    exec(compile("class _S:\n" + m.group(1), "<generated>", "exec"), ns)
    return ns["_S"], ns["logger"]


def main():
    src = os.path.join(OFFICIAL, "openclaw-opd", "openclaw_opd_api_server.py")
    if not os.path.exists(src):
        print(f"  -- skipped: {src} not present (set OPENCLAW_RL_OFFICIAL)")
        return
    dest = tempfile.mkdtemp(prefix="mcrec")
    subprocess.run(
        ["bash", os.path.join(SCRIPTS_DIR, "prepare_patched_openclaw_opd.sh"),
         OFFICIAL, dest],
        check=True, capture_output=True,
    )
    with open(os.path.join(dest, "openclaw_opd_api_server.py"), encoding="utf-8") as f:
        Cls, log = extract_purge(f.read())

    work = tempfile.mkdtemp(prefix="mcrecwork")
    rec = os.path.join(work, "run_20260914.jsonl")
    prm = os.path.join(work, "run_20260914_prm.jsonl")
    arc = os.path.join(work, "run_20260914_archive.jsonl")
    prm_arc = os.path.join(work, "run_20260914_prm_archive.jsonl")

    srv = Cls()
    srv._record_file = rec
    srv._prm_record_file = prm

    n = 0

    def ck(cond, label):
        nonlocal n
        n += 1
        if not cond:
            raise AssertionError(f"FAILED: {label}")
        print(f"  ok  {label}")

    print("[a purge empties the live file but keeps the bytes]")
    open(rec, "w", encoding="utf-8").write('{"turn": 1}\n{"turn": 2}\n')
    open(prm, "w", encoding="utf-8").write('{"prm": 1}\n')
    srv.purge_record_files()
    ck(open(rec, encoding="utf-8").read() == "",
       "the live record is emptied, as the official code requires")
    ck(open(arc, encoding="utf-8").read() == '{"turn": 1}\n{"turn": 2}\n',
       "every line is in the archive")
    ck(open(prm_arc, encoding="utf-8").read() == '{"prm": 1}\n',
       "the PRM record is archived too, to its own file")

    print("\n[repeated purges accumulate in order]")
    open(rec, "w", encoding="utf-8").write('{"turn": 3}\n')
    srv.purge_record_files()
    open(rec, "w", encoding="utf-8").write('{"turn": 4}\n')
    srv.purge_record_files()
    ck(open(arc, encoding="utf-8").read()
       == '{"turn": 1}\n{"turn": 2}\n{"turn": 3}\n{"turn": 4}\n',
       "three purges leave all four lines, in the order they were written")
    ck(open(rec, encoding="utf-8").read() == "", "and the live file is still empty")

    print("\n[non-vacuity: without the archive this test would fail]")
    # Prove the archive is doing the work: the live file alone has lost
    # everything, so an unpatched purge could not satisfy the check above.
    ck(os.path.getsize(rec) == 0 and os.path.getsize(arc) > 0,
       "the surviving bytes are in the archive, not in the live file")

    print("\n[a purge with nothing to carry writes nothing]")
    before = open(arc, encoding="utf-8").read()
    srv.purge_record_files()
    ck(open(arc, encoding="utf-8").read() == before,
       "purging an already-empty record does not append a blank entry")

    print("\n[training is never blocked by bookkeeping]")
    srv._record_file = os.path.join(work, "no_such_dir", "x.jsonl")
    srv._prm_record_file = ""
    srv.purge_record_files()          # must not raise
    ck(any("could not archive" in m or "purge" in m for m in log.messages),
       "an unwritable path is logged and skipped rather than raised")

    print(f"\nall {n} assertions passed")


if __name__ == "__main__":
    main()
