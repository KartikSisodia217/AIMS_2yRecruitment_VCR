# ARCHITECTURE

## CONFIRMED / IMPLEMENTED
- VCR data pipeline with reference resolution
- VLM abstraction (MockVLMBackbone, Qwen25VLBackbone stub)
- Stage 1: Answer scoring (AnswerScorer interface, MockAnswerScorer, LogLikelihoodScorer)
- Stage 2: CACR rationale ranking (projection head, rationale encoder, cosine similarity)
- Contrastive loss (InfoNCE, margin ranking)
- Shortcut penalty (placeholder confidence penalty)
- Total loss (contrastive + λ * shortcut)
- Evaluation metrics (Q→A, QA→R, Q→AR)
- CPU end-to-end pipeline (synthetic data + mock VLM)
- Full architecture diagram

## OPEN RESEARCH DECISIONS
1. **Reference Resolution Format**
   - *Description*: How to resolve object references (e.g. [0]).
   - *Current default*: "person N"
   - *Alternatives*: "<person_N>", "[person N]", just "person"

2. **Image Encoding Method**
   - *Description*: How image and textual regions are handled.
   - *Current default*: Standard VLM encoding.

3. **Context Representation**
   - *Description*: Should the context be just (Image + Question + Answer) or include more?
   - *Current default*: (I, Q, A)

4. **Rationale Encoding Method**
   - *Description*: How rationales are represented.
   - *Current default*: Encoded directly by VLM.

5. **Similarity Metric**
   - *Description*: How to compute similarity between context and rationale.
   - *Current default*: Cosine similarity.

6. **Contrastive Loss Formulation**
   - *Description*: Which contrastive loss to use.
   - *Current default*: InfoNCE.
   - *Alternatives*: Margin Ranking.

7. **Shortcut Penalty Strategy**
   - *Description*: Mitigating text-only shortcuts.
   - *Current default*: Confidence penalty on blind scores.

8. **Loss Weighting (λ_sp)**
   - *Description*: Weight of the shortcut penalty.
   - *Current default*: 0.1

9. **VLM Backbone Selection**
   - *Description*: Which base model to use.
   - *Current default*: Qwen2.5-VL 3B (abstracted).

10. **Fine-tuning Strategy**
    - *Description*: How to train the model.
    - *Current default*: LoRA/QLoRA on specific modules.

11. **Projection Head Architecture**
    - *Description*: Layers mapping VLM output to embedding space.
    - *Current default*: 2-layer MLP.

12. **Batch Size & Accumulation**
    - *Description*: Training batch size given memory limits.
    - *Current default*: TBD based on GPU profiling.

13. **Data Augmentation**
    - *Description*: Whether to augment images or text.
    - *Current default*: None.

14. **Negative Sampling**
    - *Description*: How to sample negatives for contrastive learning.
    - *Current default*: In-batch negatives or hard negatives from VCR dataset.

15. **Evaluation Prompting**
    - *Description*: Prompt structure for generation/scoring.
    - *Current default*: Likelihood based scoring.

16. **Caching Strategy**
    - *Description*: Pre-computing image embeddings.
    - *Current default*: No caching.

17. **Handling Invalid Samples**
    - *Description*: Filtering criteria for VCR data.
    - *Current default*: Strict validation.

18. **Multi-turn vs Single-turn Encoding**
    - *Description*: Formulating Q→A and QA→R sequentially or independently.
    - *Current default*: Two-stage sequence.
