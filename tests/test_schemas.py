"""Test data schemas."""
import pytest
from data.schemas import VCRSample

def test_vcr_sample_creation(sample_vcr_raw):
    """Test VCRSample creation with valid data."""
    sample = VCRSample(
        sample_id=sample_vcr_raw["annot_id"],
        image_path=sample_vcr_raw["img_fn"],
        question=sample_vcr_raw["question"],
        answer_choices=sample_vcr_raw["answer_choices"],
        rationale_choices=sample_vcr_raw["rationale_choices"],
        answer_label=sample_vcr_raw["answer_label"],
        rationale_label=sample_vcr_raw["rationale_label"],
        objects=sample_vcr_raw["objects"],
        bboxes=[],
        metadata={"movie": sample_vcr_raw["movie"]},
        raw_question=sample_vcr_raw["question"],
        raw_answer_choices=sample_vcr_raw["answer_choices"],
        raw_rationale_choices=sample_vcr_raw["rationale_choices"]
    )
    assert sample.sample_id == "test-0"

def test_vcr_sample_validate_valid(sample_vcr_raw):
    """Test VCRSample.validate() returns empty list for valid sample."""
    sample = VCRSample(
        sample_id="test", image_path="img", question=["?"],
        answer_choices=[["1"], ["2"], ["3"], ["4"]],
        rationale_choices=[["1"], ["2"], ["3"], ["4"]],
        answer_label=0, rationale_label=0, objects=[], bboxes=[], metadata={},
        raw_question=[], raw_answer_choices=[], raw_rationale_choices=[]
    )
    errors = sample.validate()
    assert len(errors) == 0

def test_vcr_sample_validate_wrong_answer_count():
    sample = VCRSample(
        sample_id="test", image_path="img", question=["?"],
        answer_choices=[["1"], ["2"], ["3"]],  # 3 instead of 4
        rationale_choices=[["1"], ["2"], ["3"], ["4"]],
        answer_label=0, rationale_label=0, objects=[], bboxes=[], metadata={},
        raw_question=[], raw_answer_choices=[], raw_rationale_choices=[]
    )
    errors = sample.validate()
    assert len(errors) > 0

def test_vcr_sample_validate_wrong_rationale_count():
    sample = VCRSample(
        sample_id="test", image_path="img", question=["?"],
        answer_choices=[["1"], ["2"], ["3"], ["4"]],
        rationale_choices=[["1"], ["2"]],  # 2 instead of 4
        answer_label=0, rationale_label=0, objects=[], bboxes=[], metadata={},
        raw_question=[], raw_answer_choices=[], raw_rationale_choices=[]
    )
    errors = sample.validate()
    assert len(errors) > 0

def test_vcr_sample_validate_invalid_answer_label():
    sample = VCRSample(
        sample_id="test", image_path="img", question=["?"],
        answer_choices=[["1"], ["2"], ["3"], ["4"]],
        rationale_choices=[["1"], ["2"], ["3"], ["4"]],
        answer_label=5, rationale_label=0, objects=[], bboxes=[], metadata={},
        raw_question=[], raw_answer_choices=[], raw_rationale_choices=[]
    )
    errors = sample.validate()
    assert len(errors) > 0

def test_vcr_sample_validate_invalid_rationale_label():
    sample = VCRSample(
        sample_id="test", image_path="img", question=["?"],
        answer_choices=[["1"], ["2"], ["3"], ["4"]],
        rationale_choices=[["1"], ["2"], ["3"], ["4"]],
        answer_label=0, rationale_label=-1, objects=[], bboxes=[], metadata={},
        raw_question=[], raw_answer_choices=[], raw_rationale_choices=[]
    )
    errors = sample.validate()
    assert len(errors) > 0

def test_vcr_sample_validate_missing_sample_id():
    sample = VCRSample(
        sample_id="", image_path="img", question=["?"],
        answer_choices=[["1"], ["2"], ["3"], ["4"]],
        rationale_choices=[["1"], ["2"], ["3"], ["4"]],
        answer_label=0, rationale_label=0, objects=[], bboxes=[], metadata={},
        raw_question=[], raw_answer_choices=[], raw_rationale_choices=[]
    )
    errors = sample.validate()
    assert len(errors) > 0

def test_vcr_sample_is_valid():
    sample = VCRSample(
        sample_id="test", image_path="img", question=["?"],
        answer_choices=[["1"], ["2"], ["3"], ["4"]],
        rationale_choices=[["1"], ["2"], ["3"], ["4"]],
        answer_label=0, rationale_label=0, objects=[], bboxes=[], metadata={},
        raw_question=[], raw_answer_choices=[], raw_rationale_choices=[]
    )
    assert sample.is_valid

def test_vcr_sample_has_labels():
    sample = VCRSample(
        sample_id="test", image_path="img", question=["?"],
        answer_choices=[["1"], ["2"], ["3"], ["4"]],
        rationale_choices=[["1"], ["2"], ["3"], ["4"]],
        answer_label=0, rationale_label=0, objects=[], bboxes=[], metadata={},
        raw_question=[], raw_answer_choices=[], raw_rationale_choices=[]
    )
    assert sample.has_labels

def test_vcr_sample_has_no_labels():
    sample = VCRSample(
        sample_id="test", image_path="img", question=["?"],
        answer_choices=[["1"], ["2"], ["3"], ["4"]],
        rationale_choices=[["1"], ["2"], ["3"], ["4"]],
        answer_label=None, rationale_label=None, objects=[], bboxes=[], metadata={},
        raw_question=[], raw_answer_choices=[], raw_rationale_choices=[]
    )
    assert not sample.has_labels
