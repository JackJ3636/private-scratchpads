set -euo pipefail

# ── Config ──────────────────────────────────────────────────────
MODEL="Qwen/QwQ-32B"
OUTPUT="runs/traces_qwq.jsonl"
DATASETS="math gpqa"
N_PER_DATASET=200
VLLM_PORT=8000
TP_SIZE=2           # tensor parallel across 2x A100
# ────────────────────────────────────────────────────────────────

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Create log dir
mkdir -p runs/logs

# Load modules (adjust to your ARC environment)
module purge
module load Anaconda3
module load CUDA/12.1

# Activate conda env (create beforehand: conda create -n ps python=3.11)
source activate ps

# ── Launch vLLM server in background ───────────────────────────
echo "Starting vLLM server for $MODEL (TP=$TP_SIZE)..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size "$TP_SIZE" \
    --port "$VLLM_PORT" \
    --dtype auto \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    &> runs/logs/vllm_${SLURM_JOB_ID}.log &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for server to be ready
echo "Waiting for vLLM server..."
for i in $(seq 1 120); do
    if curl -s http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "vLLM ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died. Check runs/logs/vllm_${SLURM_JOB_ID}.log"
        exit 1
    fi
    sleep 5
done

# Verify server is actually up
if ! curl -s http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
    echo "ERROR: vLLM failed to start within 600s"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# ── Run generation ─────────────────────────────────────────────
echo "Starting generation..."
python src/generate.py \
    --model "$MODEL" \
    --api-base "http://localhost:${VLLM_PORT}/v1" \
    --datasets $DATASETS \
    --n-per-dataset "$N_PER_DATASET" \
    --raw \
    --resume \
    --output "$OUTPUT"

echo "Generation complete at $(date)"

# ── Cleanup ────────────────────────────────────────────────────
echo "Shutting down vLLM..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "Job $SLURM_JOB_ID finished at $(date)"
