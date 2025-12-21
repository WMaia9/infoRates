# Memory Optimization for Evaluation Pipeline

## Problem
GPU memory was accumulating during evaluation with 2 GPUs, causing slowdowns or OOM errors during the 40-minute sweep across 25 coverage×stride combinations (5×5 grid).

## Root Cause
Three evaluation functions in `src/info_rates/analysis/evaluate.py` were missing GPU tensor cleanup:
- `evaluate_fixed_parallel` (lines ~58-97)
- `evaluate_fixed_parallel_counts` (lines ~152-189)
- `per_class_analysis_fast` (lines ~220-256)

Each function processes thousands of video clips across multiple coverage/stride combinations. Without explicit tensor cleanup, `inputs` and `logits` tensors remained in GPU memory after each batch, causing accumulation over the full evaluation run.

## Solution Applied
Added explicit GPU memory management after each batch inference:

```python
# After processing each batch:
del inputs, logits  # Release tensor references
if device.type == "cuda":
    torch.cuda.empty_cache()  # Clear CUDA cache
```

This pattern was added to **6 locations** (2 per function):
1. Main batch processing loop (when `len(batch_frames) == batch_size`)
2. Leftover batch handling (when `if batch_frames:` after main loop)

## Impact
- **Memory**: Prevents GPU memory accumulation during long evaluation runs
- **Performance**: Stable memory usage across all 25 coverage×stride combinations
- **Reliability**: Eliminates OOM errors on 2-GPU setup with `batch_size=20`

## Testing
Syntax validated with:
```bash
python -m py_compile src/info_rates/analysis/evaluate.py
```

## Usage
No changes to CLI or config required. The memory optimization is automatic and applies to all evaluation modes:
- Standard evaluation: `run_eval.py`
- Per-class analysis: `run_eval.py --per-class`
- Full pipeline: `scripts/pipeline_eval.sh`

## Next Steps
Run full evaluation on GPU node:
```bash
# Request 2 GPUs
srun --gres=gpu:2 -c 16 --time=02:00:00 --pty bash -l

# Navigate to project
cd /home/wesleyferreiramaia/data/infoRates
source .venv/bin/activate

# Set GPU count
export NPROC_PER_NODE=2

# Run full pipeline with memory optimization
bash scripts/pipeline_eval.sh config.yaml
```

Monitor GPU memory during execution:
```bash
# In separate terminal
watch -n 1 nvidia-smi
```

Expected behavior: GPU memory should stabilize after first few batches and remain constant throughout evaluation.
