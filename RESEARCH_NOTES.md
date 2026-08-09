# RESEARCH NOTES

## Environment Constraints
- Development and initial testing is CPU-only.
- All code must execute gracefully without CUDA.

## Dataset Details (VCR)
- Answer choices: exactly 4.
- Rationale choices: exactly 4.
- Bounding box annotations and objects list are provided.
- Object references must be robustly mapped.

## Model Details
- Backbone: Qwen2.5-VL (3B parameters).
- Hidden dimension: 2048.
- Layers: 36.

## Testing Coverage
- Mock VLM backbone and components are tested.
- Losses and projection heads are validated on CPU.
- E2E Sanity check pipeline verifies component connectivity.

## Design Decisions
- Separation of concerns: Data layer, Models layer, Losses layer.
- VLM backbone is abstracted to easily swap underlying architectures.
- The two-stage ranking architecture allows contrastive loss formulation naturally.

## Future Experiments
- Implement real Qwen2.5-VL backbone integration.
- Evaluate exact impact of different shortcut penalty formulations.
- Try different negative sampling techniques.

## Changelog
- Initialised infrastructure, project layout, and test suite.
