# CACR-SP: Context-Anchored Contrastive Ranking with Shortcut Penalty for Visual Commonsense Reasoning

**Status**: Research infrastructure (foundation phase)

## Task Description
Visual Commonsense Reasoning (VCR) is a task that requires an AI to not only select the correct answer to a question about an image (Q → A) but also provide a rationale explaining *why* that answer is correct (QA → R). It consists of images, detected objects with bounding boxes, questions, 4 answer choices, and 4 rationale choices.

## Architecture Overview
The system employs a two-stage approach using a VLM backbone.
- **Stage 1 (Answer Scoring)**: VLM evaluates the 4 answer choices given the image and question.
- **Stage 2 (Rationale Ranking)**: A Contrastive Ranking framework anchors rationales to the selected context.
- **Shortcut Penalty**: Mitigates unimodal shortcuts by computing blind scores (without context) and penalizing confident blind predictions.

## Project Structure
- `data/`: Dataset loading, schemas, and preprocessing.
- `models/`: VLM backbone abstracts, projection head, rationale encoder, similarity, CACR-SP model.
- `losses/`: Contrastive loss, shortcut penalty, total loss.
- `evaluation/`: Metrics (Q→A, QA→R, Q→AR).
- `scripts/`: Entry points for training, evaluation, data inspection, sanity checks.
- `tests/`: Pytest suite covering all modules.

## Setup Instructions
```bash
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Quick Start
To verify the complete architecture works end-to-end on CPU with mock data:
```bash
python scripts/sanity_check.py
```

## Running Tests
Run the test suite using pytest:
```bash
pytest tests/ -v
```

## Configuration
Model, data, and training configurations are handled through YAML files in the `configs/` directory.

## Evaluation Metrics
- **Q→A**: Accuracy of selecting the correct answer.
- **QA→R**: Accuracy of selecting the correct rationale given the correct answer.
- **Q→AR**: Joint accuracy (both answer and rationale must be correct).

## Current Status
Currently in the foundation phase, using a mock VLM backbone and synthetic VCR data for architecture validation and CPU sanity checks.

## Open Research Questions
Please refer to `ARCHITECTURE.md` and `RESEARCH_NOTES.md` for a complete list of open research decisions and notes.

## License
TBD

**Note**: CACR-SP is a research hypothesis and has not yet been validated.
