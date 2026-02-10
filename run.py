# run.py - Overnight execution script
# Usage: nohup python run.py > log.txt 2>&1 &
import subprocess
import sys

steps = [
    ("Generating associations", "src/01_generate.py"),
    ("Building networks", "src/02_build_network.py"),
    ("Calculating metrics", "src/03_metrics.py"),
    ("Running analysis & plots", "src/04_analysis.py"),
    ("FMN Ego-Networks", "src/05_fmn_multiplex.py"),
]

print("=== RLHF WORD ASSOCIATION NETWORKS ===")
for i, (desc, script) in enumerate(steps, 1):
    print(f"\n[{i}/{len(steps)}] {desc}...")
    result = subprocess.run([sys.executable, script], cwd=".")
    if result.returncode != 0:
        print(f"Error in {script}")
        sys.exit(1)

print("\n=== COMPLETED ===")
