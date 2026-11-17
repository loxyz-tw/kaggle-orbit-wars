#!/usr/bin/env bash
# 2-player BC model of the final dispatch submission: entity transformer h640 / 3 layers / 8 heads,
# trained on 2-player games only, demonstrators with leaderboard score >= 1500, won games
# up-weighted x2. Operating point = <out>.ep2 (second epoch). RESUME-safe: re-run the same command.
#   Usage: [STREAM='bc_chunks/b15_*_c*.npz'] [OUT=ckpt/bc_2p_h640] ./train_bc_2p_h640.sh
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
STREAM="${STREAM:-bc_chunks/b15_*_c*.npz}"
OUT="${OUT:-ckpt/bc_2p_h640}"
mkdir -p "$(dirname "$OUT")" logs
echo "=== 2p h640 train start $(date) ==="
"$PY" bc_train_clone.py --hidden 640 --layers 3 --heads 8 --no-focus --per-step \
    --players 2 --min-score 1500 --stream "$STREAM" \
    --win-weight 2.0 \
    --batch 256 --lr 4e-4 --lr-schedule cosine --warmup 1000 --grad-clip 1.0 --epochs 2 \
    --ckpt-every-chunks 30 --resume --out "$OUT" > logs/train_bc_2p_h640.log 2>&1 \
    || { echo "=== train FAILED (exit $?), see logs/train_bc_2p_h640.log ===" >&2; exit 1; }
echo "=== 2p h640 train DONE $(date) ==="
