# Roadmap

Tasks are sized so each is roughly a day's work — pick one, open an issue for it if it's not already tracked, and send a PR. Move items between sections as priorities change.

## Done

- [x] PhenoMamba backbone: spatial cross-scan + dual-arrow temporal scan SSM blocks
- [x] PAFPN neck, detection / ordinal-phenology / instance-embedding heads, cross-frame association
- [x] Training loop with gradient clipping, early stopping, checkpointing, TensorBoard logging
- [x] CI smoke test (forward pass on the `tiny` preset)

## Now

- [ ] Fix `src/init.py`'s `sys.path` setup (currently adds the project root, not `src/`, so `python -m src.init` fails with `ModuleNotFoundError: visualizer`; only `python src/init.py` works) and add a proper `src/__init__.py`
- [ ] Config-driven training: move hyperparameters (preset, lr, batch size, epochs) out of code and into `configs/*.yaml`, loaded by `src/init.py`
- [ ] Evaluation script reporting detection mAP and tracking accuracy (MOTA/IDF1) on a held-out split
- [ ] Document the dataset JSON schema expected by `GrowthStrawberryDataset` in `docs/`
- [ ] Unit tests for each head (`DetectionHead`, `OrdinalPhenologyHead`, `InstanceEmbedHead`) in isolation, not just the full-model smoke test

## Next

- [ ] Inference/demo script: run a trained checkpoint over a video file and write annotated output (reusing `visualizer/`)
- [ ] Publish a small pretrained checkpoint (e.g. as a GitHub Release asset) so others can reproduce results without training from scratch
- [ ] Benchmark the pure-PyTorch `selective_scan_ref` vs. the fused `mamba_ssm` kernel path and document the speedup
- [ ] Ablation configs for `align_stages` and `max_shift` (temporal alignment settings)

## Later / ideas

- [ ] Multi-crop support beyond strawberries (the heads are crop-agnostic; the dataset loader is not)
- [ ] Export to ONNX / TorchScript for deployment
- [ ] Data augmentation pipeline (temporal jitter, color/lighting for outdoor field conditions)
