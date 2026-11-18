"""Serial U/T with a fresh-process restart gate after each arm's first round."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from cas_checkpoint import durable
from resource_guard import check


ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT.parent
FIRST_V1A_COMMIT = 9216


def json_read(path):
    return json.loads(path.read_text())


def main():
    contract = json_read(STAGE / "locks/bfm-ctft-v1a-signed.json")
    assert contract["approved"] and contract["version"] == "1.1"
    assert contract["formal_N_per_arm"] == 131072
    out = STAGE / "runs/formal-pair-v1a-01"
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    wall_limit = 144000 - contract["prior_stage_job_s"] - 36000 - 900
    assert wall_limit > 2 * 5.789 * 3600
    paths = sorted((ROOT / "research/bfm_ctft_v1").glob("*.py"))

    def hashes():
        return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    source = hashes()
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    record = dict(status="RUNNING", pid=os.getpid(), runtime=revision, version="1.1", arms=[],
                  N_per_arm=131072, shared_warm_rows=8192, new_rows_per_arm=122880,
                  prior_stage_job_s=contract["prior_stage_job_s"], max_pair_wall_s=wall_limit,
                  source_sha256=source, started_unix=time.time())
    requested = False

    def request_stop(signum, frame):
        nonlocal requested
        requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def save():
        record["elapsed_s"] = time.monotonic() - started
        record["stage_job_s_conservative"] = contract["prior_stage_job_s"] + record["elapsed_s"]
        durable(out / "ledger.json", (json.dumps(record, indent=2) + "\n").encode())

    def run_phase(arm, directory, label, option):
        assert not requested and time.monotonic() - started < wall_limit
        check(STAGE)
        assert hashes() == source
        argv = [sys.executable, "-u", str(Path(__file__).with_name("formal_train.py")),
                "--arm", arm, "--output", str(directory), option]
        report = record["arms"][-1]
        with (out / f"{arm}-{label}.log").open("xb") as log:
            child = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES="0",
                                              OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                              MKL_NUM_THREADS="1"))
            phase = dict(label=label, pid=child.pid, started_unix=time.time(),
                         argv=argv)
            report["phases"].append(phase)
            save()
            stop_at = None
            while child.poll() is None:
                if requested or time.monotonic() - started > wall_limit:
                    if stop_at is None:
                        child.send_signal(signal.SIGTERM)
                        stop_at = time.monotonic()
                        record["status"] = "STOP_REQUESTED"
                    elif time.monotonic() - stop_at > 900:
                        record["status"] = "STOP_SAVE_PENDING"
                        save()
                        raise RuntimeError("Owned process remains during save; needs inspection")
                time.sleep(5)
                save()
            phase.update(returncode=child.returncode, ended_unix=time.time())
            save()
            return child.returncode == 0

    for arm in ("U", "T"):
        d = STAGE / "runs" / f"formal-{arm}-v1a-01"
        assert not d.exists()
        record["arms"].append(dict(arm=arm, output=str(d.relative_to(STAGE)), phases=[]))
        ok = run_phase(arm, d, "first-round", "--qualify-first-round")
        if ok:
            committed = json_read(d / "progress.json")["collected"]
            ok = committed == FIRST_V1A_COMMIT and (d / "first-round-uninterrupted-canary.json").is_file()
        if not ok or requested:
            record["status"] = "STOPPED_FIRST_ROUND_NO_CONTINUATION"
            save()
            return
        ok = run_phase(arm, d, "resume", "--resume")
        if ok:
            ok = (json_read(d / "first-round-recovery-qualified.json")["status"] == "PASS"
                  and json_read(d / "progress.json")["collected"] == contract["formal_N_per_arm"])
        if not ok or requested:
            record["status"] = "STOPPED_ARM_NO_AUTOMATIC_NEXT_ARM"
            save()
            return
    record["status"] = "U_T_V1A_FIXED_ENDPOINTS_COMPLETE_EVALUATION_NOT_STARTED"
    save()


if __name__ == "__main__":
    main()
