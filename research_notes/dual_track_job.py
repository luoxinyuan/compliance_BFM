"""One isolated supervised job, serial GPU lock, immutable sidecar and hard timeout.

This records a job; it does not authorize scientific settings or enlarge budgets.
Formal entrypoints must separately enforce dual_track_admission.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--runtime-repo", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--category", choices=("frozen", "fb", "common", "buffer"), required=True)
    parser.add_argument("--purpose", choices=("engineering", "frozen_ppo", "fb_train", "student_ppo"), required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    assert cmd and 0 < args.timeout <= 80*3600
    assert args.name.replace("-", "").isalnum()
    repo = args.runtime_repo or args.stage / "repo"
    if args.purpose != "engineering":
        from dual_track_admission import require_launchable
        assert args.contract is not None
        track = dict(frozen_ppo="frozen", fb_train="fb", student_ppo="student")[args.purpose]
        require_launchable(json.loads(args.contract.read_text()), track)
    assert not subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"]), "Commit before launch"
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    sidecar = args.stage / "jobs" / (args.name + ".json")
    assert not sidecar.exists(), "Immutable job identity"
    limits = dict(frozen=112*3600, fb=80*3600, common=32*3600, buffer=16*3600)
    previous = [json.loads(p.read_text()) for p in (args.stage / "jobs").glob("*.json")]
    assert all(r.get("status") != "running" for r in previous), "Another stage job is active"
    spent = sum(r.get("elapsed_s", r.get("timeout_s", 0)) for r in previous if r.get("category") == args.category)
    assert spent + args.timeout <= limits[args.category]
    assert sum(r.get("elapsed_s", r.get("timeout_s", 0)) for r in previous) + args.timeout <= 240*3600
    lock = (args.stage.parent.parent / ".dual-track-gpu0.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=index,name,uuid,memory.used,utilization.gpu", "--format=csv,noheader"], text=True).strip()
    assert "RTX PRO 5000" in gpu, gpu
    uuid = gpu.split(",")[2].strip()
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"], text=True)
    assert not any(uuid in row for row in apps.splitlines()), "Authorized GPU is occupied"
    out = args.stage / "runs" / args.name
    out.mkdir(exist_ok=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env["PYTHONPATH"] = str(repo) + ":" + str(repo / "research_notes")
    started = time.time()
    row = dict(name=args.name, category=args.category, runtime=revision, command=cmd,
               started=started, timeout_s=args.timeout, supervisor_pid=os.getpid(),
               gpu_baseline=gpu, status="running", output=str(out),
               purpose=args.purpose, formal_training=args.purpose != "engineering")
    write(sidecar, row)
    try:
        with (out / "stdout.log").open("w") as log:
            child = subprocess.Popen(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            row["pid"] = child.pid
            write(sidecar, row)
            try:
                rc = child.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                child.terminate()
                rc = child.wait(timeout=30)
                row["timed_out"] = True
        row.update(returncode=rc, status="completed" if rc == 0 else "failed")
        if rc:
            write(out / "ERRATUM.json", dict(job=args.name, runtime=revision, returncode=rc,
                                            evidence="stdout.log", prior_outputs_affected=False))
    finally:
        row["elapsed_s"] = time.time() - started
        write(sidecar, row)
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
