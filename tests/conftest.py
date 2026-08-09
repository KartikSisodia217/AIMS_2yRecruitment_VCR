import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def sample_vcr_raw():
    """Raw VCR JSON dict matching the real dataset format."""
    return {
        "annot_id": "test-0",
        "img_id": "test-img-0",
        "movie": "test_movie",
        "img_fn": "test_images/test_0.jpg",
        "metadata_fn": "test_images/test_0.json",
        "question_number": 0,
        "objects": ["person", "person", "table"],
        "question": ["Why", "is", [0], "looking", "at", [1], "?"],
        "answer_choices": [
            [[0], "is", "asking", [1], "to", "sit", "down", "."],
            [[0], "is", "ordering", "food", "from", [1], "."],
            [[0], "wants", "to", "show", [1], "something", "on", "the", [2], "."],
            [[0], "is", "angry", "at", [1], "."]
        ],
        "answer_label": 2,
        "rationale_choices": [
            [[0], "is", "pointing", "to", "an", "object", "on", "the", [2], "."],
            [[0], "is", "holding", "a", "menu", "."],
            [[0], "is", "standing", "near", "a", "door", "."],
            [[1], "is", "wearing", "a", "uniform", "."]
        ],
        "rationale_label": 0
    }

@pytest.fixture
def mock_vlm():
    from models.vlm_backbone import MockVLMBackbone
    return MockVLMBackbone(hidden_dim=256, seed=42)

@pytest.fixture
def debug_dataset():
    from data.debug_dataset import DebugVCRDataset
    return DebugVCRDataset(num_samples=5, create_images=True)

@pytest.fixture
def projection_head():
    from models.projection import ProjectionHead
    return ProjectionHead(input_dim=256, embedding_dim=128, intermediate_dim=256)

@pytest.fixture
def mock_rationale_encoder():
    from models.rationale_encoder import MockRationaleEncoder
    return MockRationaleEncoder(embedding_dim=128)

@pytest.fixture
def cosine_similarity():
    from models.similarity import CosineSimilarity
    return CosineSimilarity(temperature=1.0)
