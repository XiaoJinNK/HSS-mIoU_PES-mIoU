# Human Evaluation Materials

This folder contains a two-part preference questionnaire for validating semantic and structural segmentation metrics.

- `semantic/`: semantic-preference questions.
- `structural/`: structural-preference questions.
- `response_template.csv`: wide-format response sheet.
- `answer_key.csv`: sanitized analysis key with metric preferences, item roles, repeat links, and metric gaps.
- `protocol.md`: suggested participant and analysis protocol.

Each task contains 2 practice items, 12 core items, and 2 hidden repeat items for consistency checks.

Analyze responses from the repository root with:

```bash
python tools/human_eval/analyze_human_evaluation.py \
  --package-dir survey_materials/human_evaluation \
  --response-csv /path/to/filled_response_template.csv \
  --out-dir outputs/human_eval
```
