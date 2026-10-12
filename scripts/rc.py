"""Run `python <args>` and print its exit code as the last log line ("RC <n>"), exiting with it -- so a detached
`scripts/spark.py run --name ID -- scripts/rc.py scripts/train.py ...` job leaves its exit status in runs/ID.log
(`spark.py status` shows it).
    PYTHONPATH=. python scripts/rc.py scripts/train.py sft ..."""
import subprocess
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: rc.py <script.py> [args ...]")
    rc = subprocess.run([sys.executable, *sys.argv[1:]]).returncode
    print(f"RC {rc}", file=sys.stderr, flush=True)
    sys.exit(rc)
