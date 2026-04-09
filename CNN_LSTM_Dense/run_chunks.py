import subprocess
import sys
from pathlib import Path

# ===== first test settings =====
TOTAL_CHUNKS = 2
CHUNK_ROUNDS = 5
FRACTION_FIT = 0.3
FRACTION_EVALUATE = 0.0   # safer first test

CHECKPOINT_DIR = Path("chunk_checkpoints")
LOG_ROOT = Path("chunk_logs")
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_ROOT.mkdir(exist_ok=True)

current_model = ""
start_round = 0

for chunk_idx in range(1, TOTAL_CHUNKS + 1):
    out_model = CHECKPOINT_DIR / f"global_chunk_{chunk_idx:03d}_CNN_LSTM_Dense.keras"
    out_dir = LOG_ROOT / f"chunk_{chunk_idx:03d}"

    cmd = [
        sys.executable,
        "fl_chunk_run.py",
        "--chunk-rounds", str(CHUNK_ROUNDS),
        "--fraction-fit", str(FRACTION_FIT),
        "--fraction-evaluate", str(FRACTION_EVALUATE),
        "--chunk-index", str(chunk_idx),
        "--start-round", str(start_round),
        "--out-model", str(out_model),
        "--out-dir", str(out_dir),
    ]

    if current_model:
        cmd += ["--in-model", current_model]

    print(f"\n=== Running chunk {chunk_idx}/{TOTAL_CHUNKS} ===")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)

    current_model = str(out_model)
    start_round += CHUNK_ROUNDS

print("\nAll chunks completed.")
print(f"Final model: {current_model}")