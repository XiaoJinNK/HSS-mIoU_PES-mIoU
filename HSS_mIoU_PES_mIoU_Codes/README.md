# Semantic-Structural Segmentation Metrics

Anonymous code release for evaluating semantic segmentation predictions with complementary semantic and structural metrics.

The repository contains:

- `metrics/`: implementations of standard mIoU, hierarchical semantic soft IoU, and persistence-edge structural scoring.
- `scripts/`: end-to-end evaluation, matched-pair validation, metric-difference analysis, and six-dataset mIoU checking.
- `tools/semantic/`: semantic metric experiments, semantic corruption studies, validation runs, context-conditioned variants, and similarity-matrix construction.
- `tools/structural/`: controlled structural perturbation, calibration, parameter sweep, and stability utilities.
- `tools/human_eval/`: human preference-response analysis and simulation helpers.
- `survey_materials/`: public human-questionnaire materials. Hidden model names and local paths have been removed.

Large files are intentionally not included: dataset copies, model weights, prediction dumps, cached outputs, paper PDFs, and historical experiment folders.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`numba` is optional; it accelerates the structural metric when available.

## Evaluate Predictions

Use label maps with matching relative names under the ground-truth and prediction directories. PNG/TIFF/JPEG/NumPy masks are supported.

```bash
python scripts/evaluate_predictions.py \
  --gt_dir /path/to/ground_truth \
  --pred_dirs /path/to/model_a_preds /path/to/model_b_preds \
  --pred_names ModelA ModelB \
  --similarity_matrix /path/to/S.npy \
  --num_classes 150 \
  --ignore_index 255 \
  --output_dir outputs/eval_run
```

Outputs include image-level metrics, class-level metrics, overall summaries, rankings, bootstrap confidence intervals, and optional case-study assets.

## Structural Metric Tools

Run a direct structural evaluation:

```bash
python metrics/structural_metrics.py \
  --mode dataset \
  --gt-dir /path/to/ground_truth \
  --pred-dir /path/to/predictions \
  --num-classes 150 \
  --ignore-index 255 \
  --max-radius 5 \
  --boundary-width 2
```

Calibrate or sweep parameters with:

```bash
python tools/structural/calibrate_csp_params.py --help
python tools/structural/select_csp_non_degenerate_params.py --help
python tools/structural/sweep_csp_radius.py --help
```

## Semantic Experiment Tools

Build a semantic similarity matrix from a class-to-WordNet mapping:

```bash
python tools/semantic/build_similarity_matrix.py --help
```

Run semantic metric perturbation, comparison, and validation experiments:

```bash
python tools/semantic/run_metric_experiments_v2.py --help
python tools/semantic/run_validations.py --help
python tools/semantic/run_context_validation.py --help
python tools/semantic/run_comparison_experiments.py --help
```

Helper modules for these runs are in the same folder: `corrupt.py`, `bootstrap.py`, `context_metrics.py`, and `analyze_corruption.py`.

## Human Evaluation

The bundled questionnaire package is under `survey_materials/human_evaluation`.

Analyze collected responses:

```bash
python tools/human_eval/analyze_human_evaluation.py \
  --package-dir survey_materials/human_evaluation \
  --response-csv /path/to/filled_response_template.csv \
  --out-dir outputs/human_eval
```

Generate synthetic responses for pipeline testing only:

```bash
python tools/human_eval/simulate_human_responses.py \
  --package-dir survey_materials/human_evaluation \
  --n 200
```

## Six-Dataset mIoU Check

`scripts/evaluate_six_dataset_miou.py` uses environment variables for dataset roots and prediction roots:

```bash
export PREDICTION_ROOT=/path/to/all_datasets_raw
export ADE150_GT_DIR=/path/to/ADE150/validation
export ADE847_GT_DIR=/path/to/ADE847/validation
export PC59_GT_DIR=/path/to/PC59/val
export PC459_GT_DIR=/path/to/PC459/val
export VOC20_GT_DIR=/path/to/VOC20/val
export VOC20B_GT_DIR=/path/to/VOC20b/val

python scripts/evaluate_six_dataset_miou.py --output-dir outputs/miou_6datasets
```

## Notes For Anonymous Release

- No author names, absolute local paths, model weights, or private datasets are included.
- The questionnaire answer key is sanitized: it keeps only fields needed for analysis and removes model identities.
- Add the final project license before making the GitHub repository public.
