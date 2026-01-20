# VGGT-SLAM Slurm Guide

## Submit a 24h requeueable job
1) Pick a run directory and dataset root:
   ```bash
   export RUN_DIR=/path/to/runs/film_m2f_$(date +%Y%m%d_%H%M%S)
   export DATASET_ROOT=/path/to/dataset_root
   # optional: export LABELS_ROOT=/path/to/labels_root
   ```
2) Submit (extra training flags can be appended after `--`):
   ```bash
   sbatch --export=ALL,RUN_DIR,DATASET_ROOT,LABELS_ROOT slurm/train_requeue.slurm -- --batch-size 2 --epochs 60
   ```
   - Resources: 1x GPU (prefers A100, falls back to A40), 40G RAM, 24h wallclock.
   - Logs: `$RUN_DIR/logs/train_<jobid>.log`
   - Checkpoints: `$RUN_DIR/checkpoints/film_m2f_epochXXXX_stepXXXXXXXX_<ts>.pt`

## Monitor progress
- Check queue: `squeue -u $USER`
- Tail latest log and checkpoint: `./slurm/monitor.sh $RUN_DIR`

## Requeue and resume behavior
- Slurm sends `USR1` 120s before preemption/time-limit. The job traps it and touches `$RUN_DIR/REQUEST_SAVE`.
- The trainer sees `REQUEST_SAVE`, writes an atomic checkpoint on rank0, and clears the file.
- `#SBATCH --requeue` automatically resubmits; the next launch uses `--resume auto` to pick the newest checkpoint in `$RUN_DIR/checkpoints`.
- You can manually request a save mid-run: `touch $RUN_DIR/REQUEST_SAVE`
- You can force a checkpoint + requeue now: `scancel --signal=USR1 <jobid>`

## Manual restart outside Slurm
```bash
python -u scripts/train_film_m2f_optimized.py \
  --dataset-root "$DATASET_ROOT" \
  --run-dir "$RUN_DIR" \
  --ckpt-dir "$RUN_DIR/checkpoints" \
  --resume auto \
  --batch-size 2 --epochs 60
```
