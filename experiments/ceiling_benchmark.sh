#!/bin/bash
# Ceiling benchmark: k=0 vs k=1 vs k=2 vs k=5 对比
# 回答: "refiner 的天花板到底多高？"

SOURCE="weights/lowdim/predictor/vae/vae_pusht_lowdim/checkpoints/latest.ckpt"
REFINER="weights/lowdim/diffusion/pusht/cnn/latest.ckpt"
TASK="pusht"
N_EPISODES=${1:-30}

echo "============================================"
echo "Ceiling Benchmark — $TASK"
echo "n_episodes = $N_EPISODES"
echo "============================================"

for K in 0 1 2 5; do
    echo ""
    echo "--- k = $K ---"
    dp-infer-d3rl --task $TASK \
        --source $SOURCE \
        --refiner $REFINER \
        --refinement-steps $K \
        --n-episodes $N_EPISODES 2>&1 | grep -E "Success rate|Mean reward|Avg k|Avg inference|k distribution|Episode [0-9]"
done

echo ""
echo "============================================"
echo "Done. Compare k=0 (source only) vs k=5 (strongest refine)."
echo "If k=5 success_rate ≤ k=0 success_rate → refiner ceiling is low → pipeline rethink needed."
echo "============================================"
