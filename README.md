# ARC-AGI Data Augmentation & Fine-tuning

This repository contains utilities to augment ARC-AGI grid tasks, visualize them, and fine-tune small LoRA models using the Tinker API. The goal is to expand the dataset via geometric and color transforms, make it easy to inspect examples, and run experiments that evaluate model behaviour on ARC-style tasks.

Key components
- src/sample_augmentation.py — augmentation pipeline that produces augmented JSON datasets (rotations, flips, color changes).
- src/utils/visualizer.py — small helper(s) to render grids and tasks for quick inspection.
- tinker_finetuning/ — training and evaluation scripts that use Tinker (LoRA training, sampling/evaluation).
- notebooks/ — explorations and quick demos.

This README covers setup, how to run the augmentation pipeline, how to train/evaluate with Tinker, and a few troubleshooting notes.

## Quick links
- Augmentation: `src/sample_augmentation.py`
- Visualizer: `src/utils/visualizer.py`
- Training: `tinker_finetuning/train.py`
- Evaluation: `tinker_finetuning/eval.py`
- Notebooks: `notebooks/exploring_data.ipynb`

---

## Requirements & setup

Recommended: create and activate a virtual environment, then install requirements:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Environment variables
- TINKER_API_KEY — required to use Tinker training/sampling APIs.

Confirm your environment by running a tiny smoke test, e.g.:

```bash
python -c "import tinker; print('tinker ok')"
```

---

## Data augmentation pipeline

Input
- A pair of JSON files containing ARC-AGI tasks (challenges and solutions). The repository includes sample data in `data/arc-agi-2025/raw/`.

Output
- A JSON file containing augmented tasks saved to the directory provided by `--save_base_dir`.

Transformations applied
- Rotations: 90°, 180°, 270°
- Flips: horizontal, vertical
- Color remapping: change colors used in grids

Run augmentation

```bash
python3 -m src.sample_augmentation \
  --base_dir data/arc-agi-2025/raw \
  --challenges_filename arc-agi_training_challenges.json \
  --solutions_filename arc-agi_training_solutions.json \
  --save_base_dir data/arc-agi-2025/sample_augmented
```

This will create augmented JSON files under the `save_base_dir` you provide.

---

## Visualizing tasks

Use the visualizer in `src/utils/visualizer.py` to render a single example or an entire task. The function `plot_task` (or `visualize_grid`) accepts a list of examples and displays input/output grids side-by-side.

You can also open the notebook `notebooks/exploring_data.ipynb` for interactive viewing; the notebook includes a small sys.path fix cell so `src` imports work when running the notebook from the `notebooks/` folder.

---

## Training (LoRA fine-tuning) with Tinker

Prerequisites
- Set your Tinker API key:

```bash
export TINKER_API_KEY=your_api_key_here
```

- Install requirements (already shown above).

Example training command (from project root)

```bash
python -m tinker_finetuning.train \
  --model_name Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --training_prompts_path data/arc-agi-2025/prompts/arc-agi_training_prompts.json \
  --validation_prompts_path data/arc-agi-2025/prompts/arc-agi_validation_prompts.json \
  --epochs 5 \
  --batch_size 64 \
  --learning_rate 5e-4 \
  --max_seq_length 32500 \
  --lora_rank 32 \
  --checkpoint_dir ./tinker_checkpoints/arc-agi_qwen3_30b_instruct \
  --run_name arcagi-run-qwen3-30b-instruct \
  --wandb \
  --wandb_project arcagi \
  --wandb_run_name arcagi-run-qwen3-30b-instruct \
  --wandb_tags arc,agi,training
```

Notes
- When the training code saves LoRA weights for sampling, prefer using the helper `save_weights_and_get_sampling_client()` or `save_weights_for_sampler()` so you get a `sampler_weights` path that is compatible with the sampling API.

---

## Evaluation / Sampling

The repo includes `tinker_finetuning/eval.py` which accepts a `--checkpoint_path` and an evaluation prompts file and writes JSON results.

Example eval command:

```bash
python -m tinker_finetuning.eval \
  --model_name meta-llama/Llama-3.3-70B-Instruct \
  --checkpoint_path tinker://YOUR_CHECKPOINT_PATH \
  --eval_prompts_path data/arc-agi-2025/prompts/arc-agi_evaluation_challenges.json \
  --output_path ./eval_results.json \
  --max_tokens 512
```

Important: the sampling API requires a `sampler_weights` style checkpoint path for LoRA-sampled clients (see training notes above). If you only want to sample from a base model, supply `base_model` when creating the sampling client.

---

## Submission results

- Summary: Our ARC‑AGI 2025 submission achieved a +23% improvement over the base Qwen3 4B model on the evaluation metric used (reported relative gain vs. Qwen3). Detailed per-task metrics and the evaluation script output are available in `eval_results.json` included with the submission artifacts.


## Notebooks

Open `notebooks/exploring_data.ipynb` to inspect data, run the visualizer, and experiment with the dataset. The notebook includes a small sys.path fix cell so `src` imports work when running the notebook from the `notebooks/` folder.

---

## Troubleshooting

- Import errors in notebooks: add the project root to `sys.path` (the notebook includes this snippet already).
- Sampling errors about invalid tinker paths: ensure you use `save_weights_for_sampler()` or `save_weights_and_get_sampling_client()` to obtain a `tinker://.../sampler_weights/...` path.
- "List is not defined" in notebooks: import typing names (the notebook's top cell now imports `List`).

---

## Contributing

Feel free to open issues or PRs. Suggested improvements:
- More augmentation strategies (color palette transfers, scaling that preserves semantics)
- Unit tests for augmentation transforms
- Integration tests for training/eval pipelines

---

## License

This project is provided under the MIT License. See LICENSE file for details (if present).

---

If you'd like, I can also:
- Add a minimal `pyproject.toml` or `setup.cfg` to make `src` installable (so notebooks and scripts can import the package without sys.path hacks), or
- Patch the notebooks to replace remaining broken import lines with the absolute import and the sys.path snippet.

Let me know which follow-up you'd prefer.
