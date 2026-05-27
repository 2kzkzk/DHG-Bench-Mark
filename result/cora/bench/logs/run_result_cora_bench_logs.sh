#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
BENCH_LOG_DIR="$REPO_ROOT/logs/cora/bench"
EPISODE_BANK_DIR="$REPO_ROOT/result/fewshot_episode_banks"

mkdir -p "$BENCH_LOG_DIR" "$EPISODE_BANK_DIR"
: > "$BENCH_LOG_DIR/result.log"
rm -f "$BENCH_LOG_DIR/result.csv" "$BENCH_LOG_DIR/result.json"

COMMON_ARGS=(
  --task_type fewshot_node_cls
  --dname cora
  --fs_class_split 3/2/2
  --fs_train_way 3
  --fs_val_way 2
  --fs_test_way 2
  --fs_shot 1
  --fs_query 15
  --embedding_hidden 128
  --fs_metric cosine
  --fs_temperature 10.0
  --fs_fair_config True
  --fs_reuse_episode_bank True
  --fs_save_episode_bank True
  --fs_episode_bank_dir "$EPISODE_BANK_DIR"
  --fs_log_dir "$REPO_ROOT/logs"
  --fs_train_episodes 300
  --fs_val_episodes 100
  --fs_test_episodes 600
  --lr 0.01
  --wd 0
  --dropout 0.2
  --num_seeds 5
)

MODELS=(
  HGNN
  HCHA
  HNHN
  AllSetformer
  AllDeepSets
  UniGCNII
  UniGIN
  HyperGCN
  EDHNN
  ZEN
  RawFeatureProto
)

cd "$REPO_ROOT/dhgbench"

for MODEL in "${MODELS[@]}"; do
  EXTRA_ARGS=()
  if [[ "$MODEL" == "AllDeepSets" ]]; then
    EXTRA_ARGS+=(--aggregate mean)
  fi
  if [[ "$MODEL" == "ZEN" ]]; then
    EXTRA_ARGS+=(--zen_mode no_projection)
  fi

  {
    echo "========================================================================"
    echo "Running fewshot_node_cls model=$MODEL"
    echo "The run prints split hash, episode bank paths, and episode bank hashes."
    python main.py --method "$MODEL" "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
  } 2>&1 | tee -a "$BENCH_LOG_DIR/result.log"
done

echo "========================================================================" | tee -a "$BENCH_LOG_DIR/result.log"
echo "Few-shot benchmark complete." | tee -a "$BENCH_LOG_DIR/result.log"
echo "Summary log: $BENCH_LOG_DIR/result.log" | tee -a "$BENCH_LOG_DIR/result.log"
echo "CSV: $BENCH_LOG_DIR/result.csv" | tee -a "$BENCH_LOG_DIR/result.log"
echo "JSON: $BENCH_LOG_DIR/result.json" | tee -a "$BENCH_LOG_DIR/result.log"
