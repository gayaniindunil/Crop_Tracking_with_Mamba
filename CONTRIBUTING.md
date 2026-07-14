# Contributing

Thanks for considering a contribution to PhenoMamba.

## Getting set up

```bash
git clone https://github.com/gayaniindunil/Crop_Tracking_with_Mamba.git
cd Crop_Tracking_with_Mamba
pip install -r requirements.txt
```

## Workflow

1. Check [open issues](../../issues) for something to work on, or open one to discuss a change before starting on anything large.
2. Create a branch off `main`: `git checkout -b your-feature-name`.
3. Keep commits small and focused — one logical change per commit, with a message that explains *why*, not just *what*.
4. Open a pull request against `main`. Describe what changed and how you verified it (a script run, a training curve, a before/after visualization).

## Code style

- Follow the existing module layout: model code in `src/model/`, training in `src/engine/`, data loading in `src/data_processor/`, visualization in `src/visualizer/`.
- No commented-out code or dead experiments in merged PRs — delete instead of commenting out.
- Don't commit large binaries (checkpoints, videos, datasets). These are `.gitignore`d — if you need to share results, attach them to the PR description or a release instead.

## Reporting bugs / proposing features

Open an [issue](../../issues) with:
- What you expected vs. what happened (for bugs), or the motivation (for features)
- Steps to reproduce, if applicable
- Environment (OS, Python version, GPU/CUDA version) for anything training-related
