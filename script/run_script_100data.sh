#!/bin/bash --login
#SBATCH -p gpuA                 # 队列/分区
#SBATCH -G 1                    # GPU 数量
#SBATCH -n 12                   # CPU 核心数
#SBATCH -t 2-0:00:00           # 最长运行时间 (天-时:分:秒)
#SBATCH --mem=24G               # 内存
#SBATCH -J MSV_Paper            # 任务名
#SBATCH -o %x.%j.out            # 标准输出
#SBATCH -e %x.%j.err            # 标准错误

set -euo pipefail

# -------- 可配置变量 --------
ENV_NAME=${ENV_NAME:-cgtest}                               # conda 环境名
WANDB_MODE=${WANDB_MODE:-offline}                          # online|offline|disabled
REPO_ROOT=${REPO_ROOT:-/mnt/iusers01/fatpou01/compsci01/k09562zs}
PROJECT_ROOT=${PROJECT_ROOT:-$REPO_ROOT/scratch/Multi_scale_vision}
CONFIGS_DIR=${CONFIGS_DIR:-$PROJECT_ROOT/configs}
EXPERIMENTS_ROOT=${EXPERIMENTS_ROOT:-$PROJECT_ROOT/experiments}
RESULTS_DIR=${RESULTS_DIR:-$PROJECT_ROOT/results}

# 数据路径
DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/scratch/Ball_counting_CNN/ball_data_collection}
TRAIN_CSV=${TRAIN_CSV:-$REPO_ROOT/scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_train.csv}
VAL_CSV=${VAL_CSV:-$REPO_ROOT/scratch/Ball_counting_CNN/Tools_script/ball_counting_dataset_val.csv}

# 实验网格 (空格分隔，传给 run_all_experiments.py)
SEEDS=${SEEDS:-"42 52 62 72 82"}

# 训练参数
TOTAL_EPOCHS=${TOTAL_EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-32}
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-11}
NUM_WORKERS=${NUM_WORKERS:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
SAVE_EVERY=${SAVE_EVERY:-10}
RUN_TRAIN=${RUN_TRAIN:-1}
RUN_EVAL=${RUN_EVAL:-1}
RUN_VIZ=${RUN_VIZ:-1}

# -------- 环境初始化 --------
echo "=== 作业信息 ==="
echo "节点: ${SLURMD_NODENAME:-N/A}  GPU: ${SLURM_GPUS:-0}  CPU: ${SLURM_NTASKS:-N/A}  内存: ${SLURM_MEM_PER_NODE:-N/A}"
echo "开始时间: $(date)"

source ~/.bashrc || true
conda activate "$ENV_NAME"

python -c 'import sys; print("Python:", sys.version)'
python -c 'import torch, platform; print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available(), platform.platform())' || {
	echo "当前环境 $ENV_NAME 中未找到 torch" >&2
	exit 1
}

export WANDB_MODE="$WANDB_MODE"
export PYTHONUNBUFFERED=1

# -------- 配置注入 --------
cd "$PROJECT_ROOT"

echo "=== 更新 configs 下实验参数 ==="
python - <<'PY'
import json
import os

configs_dir = os.environ["CONFIGS_DIR"]
data_root = os.environ["DATA_ROOT"]
train_csv = os.environ["TRAIN_CSV"]
val_csv = os.environ["VAL_CSV"]
experiments_root = os.environ["EXPERIMENTS_ROOT"]

total_epochs = int(os.environ["TOTAL_EPOCHS"])
batch_size = int(os.environ["BATCH_SIZE"])
sequence_length = int(os.environ["SEQUENCE_LENGTH"])
num_workers = int(os.environ["NUM_WORKERS"])
learning_rate = float(os.environ["LEARNING_RATE"])
weight_decay = float(os.environ["WEIGHT_DECAY"])
save_every = int(os.environ["SAVE_EVERY"])

for filename in os.listdir(configs_dir):
	if not filename.endswith(".json"):
		continue
	path = os.path.join(configs_dir, filename)
	with open(path, "r") as f:
		cfg = json.load(f)

	cfg["data_root"] = data_root
	cfg["train_csv"] = train_csv
	cfg["val_csv"] = val_csv
	cfg["experiments_root"] = experiments_root
	cfg["total_epochs"] = total_epochs
	cfg["batch_size"] = batch_size
	cfg["sequence_length"] = sequence_length
	cfg["num_workers"] = num_workers
	cfg["learning_rate"] = learning_rate
	cfg["weight_decay"] = weight_decay
	cfg["save_every"] = save_every

	with open(path, "w") as f:
		json.dump(cfg, f, indent=2)

print("Configs updated:", configs_dir)
PY

echo "=== 运行参数 ==="
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "CONFIGS_DIR=$CONFIGS_DIR"
echo "EXPERIMENTS_ROOT=$EXPERIMENTS_ROOT"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "DATA_ROOT=$DATA_ROOT"
echo "TRAIN_CSV=$TRAIN_CSV"
echo "VAL_CSV=$VAL_CSV"
echo "SEEDS=$SEEDS"

# -------- 启动实验 --------
status=0

if [[ "$RUN_TRAIN" == "1" ]]; then
	echo "=== 开始训练网格 (E0-E4 x seeds) ==="
	python run_all_experiments.py \
		--configs_dir "$CONFIGS_DIR" \
		--seeds $SEEDS
fi

if [[ "$RUN_EVAL" == "1" ]]; then
	echo "=== 汇总指标 ==="
	python evaluate.py \
		--experiments_root "$EXPERIMENTS_ROOT" \
		--results_dir "$RESULTS_DIR"
fi

if [[ "$RUN_VIZ" == "1" ]]; then
	echo "=== 生成论文图表 ==="
	python visualize_results.py \
		--experiments_root "$EXPERIMENTS_ROOT" \
		--results_dir "$RESULTS_DIR"
fi

echo "=== 关键输出 ==="
echo "- $RESULTS_DIR/main_table.csv"
echo "- $RESULTS_DIR/metrics_summary.csv"
echo "- $RESULTS_DIR/final_accuracy_bar.png"
echo "- $RESULTS_DIR/macro_f1_bar.png"
echo "- $RESULTS_DIR/accuracy_over_time.png"
echo "- $RESULTS_DIR/cm_visual_only.png"
echo "- $RESULTS_DIR/cm_stepwise_fusion.png"

echo "=== 完成，状态: $status，结束时间: $(date) ==="
exit $status
