# PhenoMamba: Crop Growth-Stage Detection & Tracking with Mamba

[![CI](https://github.com/gayaniindunil/Crop_Tracking_with_Mamba/actions/workflows/ci.yml/badge.svg)](https://github.com/gayaniindunil/Crop_Tracking_with_Mamba/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A spatiotemporal state-space model (Mamba/SSM) for jointly **detecting**, **classifying the growth stage of**, and **tracking** individual crops (strawberries) across video frames.

Most crop-monitoring pipelines treat detection, phenology (growth-stage) classification, and tracking as separate stages bolted together. PhenoMamba instead uses a single spatiotemporal Mamba backbone with dedicated task heads, so temporal context (how a fruit changes over time) directly informs detection and stage classification instead of being handled only at a downstream tracking step.

## Why Mamba

Selective state-space models (Mamba) scale linearly with sequence length, unlike the quadratic cost of attention. That makes them a good fit for video: PhenoMamba can look across many frames of a growing crop without the memory blow-up a comparable transformer-based tracker would incur.

## Architecture

```text
video clip (b, T, 3, H, W)
        │
        ▼
PhenoMambaBackbone            hierarchical stages, each a stack of ST-SSM blocks
  ├─ SpatialCrossScan          selective scan over spatial tokens per frame
  └─ DualArrowTemporalScan     bidirectional selective scan across time, with
                               deformable alignment between frames
        │
        ▼
PAFPN neck                    top-down path-aggregation over the feature pyramid
        │
        ├─ DetectionHead            anchor-free objectness + distribution-focal box regression
        ├─ OrdinalPhenologyHead     ordinal classification of growth stage per detection
        └─ InstanceEmbedHead        per-instance embedding for cross-frame association
                │
                ▼
        CrossFrameAssociation   links detections across frames into tracks using the
                                learned embeddings (run at inference on ROI tokens)
```

Model presets (`tiny` / `small` / `base`) are defined in [`src/model/phenomamba.py`](src/model/phenomamba.py).

## Project layout

```text
src/
├── model/            PhenoMamba backbone, SSM blocks, scans, heads, association (this is the core contribution)
├── engine/            training loop (DetectionTrainer): AMP, grad clipping, early stopping, checkpointing
├── data_processor/    dataset loading + preprocessing (growth-stage strawberry dataset)
├── visualizer/        bounding box / grid-target / prediction visualization utilities
└── init.py            training entry point
configs/               experiment configs
docs/                  notes and write-ups
tests/                 test suite (WIP — see Roadmap)
```

## Getting started

```bash
git clone https://github.com/gayaniindunil/Crop_Tracking_with_Mamba.git
cd Crop_Tracking_with_Mamba
pip install -r requirements.txt
```

The selective-scan recurrence has a pure-PyTorch fallback ([`src/model/ssm.py`](src/model/ssm.py)), so this runs on CPU out of the box. For a large speedup on GPU, optionally install the fused CUDA kernel:

```bash
pip install mamba-ssm causal-conv1d
```

(see [state-spaces/mamba](https://github.com/state-spaces/mamba) if that build fails on your platform — the model works without it).

Train:

```bash
python src/init.py
```

Training writes TensorBoard event files to `outputs/tensorboard`:

```bash
tensorboard --logdir outputs/tensorboard
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the current task list, and open [Issues](../../issues) for ones that are ready to pick up.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for how to get set up and propose changes.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
