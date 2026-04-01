# Multi_scale_vision

Paper-oriented experiment guide for embodied counting with multimodal sequence models.

This README is written for one goal: run all core experiments once, then directly obtain the tables/figures needed in a paper.

## 1. Research Question

We test whether multimodal temporal interaction improves counting performance.

Main logic:
1. Are single modalities useful? (`joint_only`, `visual_only`)
2. Is multimodal better than strongest single modality? (`early_fusion`)
3. Is dual-stream temporal modeling better than single-stream fusion? (`late_fusion` vs `early_fusion`)
4. Is stepwise fusion better than late fusion? (`stepwise_fusion` vs `late_fusion`)

## 2. Main Experiment Matrix (must-run)

| ID | Model | Input | Temporal Structure | Fusion | Purpose |
|---|---|---|---|---|---|
| E0 | `joint_only` | joints | single joint LSTM | none | joint signal standalone predictive power |
| E1 | `visual_only` | images | single visual LSTM | none | visual-only baseline |
| E2 | `early_fusion_single_stream` | images + joints | single LSTM | concat before LSTM | standard multimodal baseline |
| E3 | `dual_stream_late_fusion` | images + joints | visual LSTM + joint LSTM | fuse at final step | value of dual-stream temporal modeling |
| E4 | `dual_stream_stepwise_fusion` | images + joints | visual LSTM + joint LSTM | fuse at every step | proposed method |

## 3. Fixed Training Setup (for fair comparison)

Use exactly the same setup for E0-E4 except model architecture.

- Data loader: `Data_loader/Data_loader_embodiment.py`
- Input image mode: RGB
- Input size: 224 x 224
- Sequence length: 11
- Joint dimension: 2
- Hidden dimension: 256
- LSTM layers: 2
- Batch size: 16
- Optimizer: Adam
- Learning rate: 1e-4
- Weight decay: 1e-5
- Scheduler: CosineAnnealingLR
- Epochs: 50 (or 100, choose one and keep fixed)
- Seeds: 42, 52, 62, 72, 82

## 4. Metrics (paper report)

Primary metrics:
1. `Final Accuracy` (main metric)
2. `Macro-F1`
3. `Balanced Accuracy`

Secondary metrics:
1. `Validation CE Loss`
2. `Params`
3. `Epoch Time` (or Throughput)

Report as mean +/- std across seeds.

## 5. Figures for Paper

Required figures:
1. Final Accuracy bar plot (E0-E4)
2. Macro-F1 bar plot (E0-E4)
3. Accuracy-over-time curve (E2, E3, E4)
4. Confusion Matrix for E1 (`visual_only`)
5. Confusion Matrix for E4 (`stepwise_fusion`)

Optional but recommended:
1. UMAP/t-SNE of final hidden state (E1 vs E4)
2. Stability plot (last-3-step prediction consistency for E2/E3/E4)

## 6. Output Files Required by Paper

After experiments, you should have:

1. `results/metrics_summary.csv`
2. `results/main_table.csv`
3. `results/final_accuracy_bar.png`
4. `results/macro_f1_bar.png`
5. `results/accuracy_over_time.png`
6. `results/cm_visual_only.png`
7. `results/cm_stepwise_fusion.png`
8. `results/run_manifest.json` (all runs, seeds, checkpoints, commit hash)

## 7. Recommended Project Structure

```text
Multi_scale_vision/
├── Data_loader/
│   └── Data_loader_embodiment.py
├── Models/
│   ├── Model_alexnet_embodiment.py
│   └── baselines.py
├── trainer.py
├── evaluate.py
├── visualize_results.py
├── configs/
│   ├── exp_joint_only.json
│   ├── exp_visual_only.json
│   ├── exp_early_fusion.json
│   ├── exp_late_fusion.json
│   └── exp_stepwise_fusion.json
├── experiments/
│   ├── joint_only/
│   ├── visual_only/
│   ├── early_fusion/
│   ├── late_fusion/
│   └── stepwise_fusion/
└── results/
```

Notes:
- `Model_alexnet_embodiment.py` currently contains the stepwise-fusion model.
- `Models/baselines.py` contains E0-E3 baseline implementations.

## 8. Execution Plan (single full pass)

Step 1: Train all runs
1. For each experiment E0-E4
2. For each seed in [42, 52, 62, 72, 82]
3. Save checkpoint + history + validation predictions

Step 2: Evaluate all runs
1. Compute Final Accuracy, Macro-F1, Balanced Accuracy, CE Loss
2. Aggregate mean/std by experiment
3. Save `results/main_table.csv` and `results/metrics_summary.csv`

Step 3: Produce figures
1. Generate required figures listed in Section 5
2. Export high-resolution PNG (>=300 dpi if used in paper)

Step 4: Reproducibility package
1. Save run manifest with seed, config, timestamp, checkpoint path, git commit
2. Ensure all paper figures are generated from files in `results/`

## 8.1 Commands

Run all experiments (E0-E4 x seeds):

```bash
python run_all_experiments.py --configs_dir ./configs --seeds 42 52 62 72 82
```

Aggregate metrics and generate paper tables:

```bash
python evaluate.py --experiments_root ./experiments --results_dir ./results
```

Generate paper figures:

```bash
python visualize_results.py --experiments_root ./experiments --results_dir ./results
```

## 9. Main Table Template

| Model | Final Acc | Macro-F1 | Balanced Acc | Val CE Loss | Params |
|---|---:|---:|---:|---:|---:|
| E0 Joint-only |  |  |  |  |  |
| E1 Visual-only |  |  |  |  |  |
| E2 Early-fusion |  |  |  |  |  |
| E3 Late-fusion |  |  |  |  |  |
| E4 Stepwise-fusion |  |  |  |  |  |

## 10. Expected Conclusion Pattern

Target pattern to support the paper claim:
1. `E2 > max(E0, E1)`
2. `E3 > E2`
3. `E4 > E3`

If this holds consistently across seeds, your main claim is strongly supported.
