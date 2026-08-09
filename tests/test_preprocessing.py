"""Test data preprocessing."""
import pytest
from data.preprocessing import resolve_references, build_object_name_map, validate_vcr_raw, parse_vcr_sample

def test_resolve_references_single():
    tokens = ["Why", "is", [0], "here", "?"]
    objects = ["person"]
    result = resolve_references(tokens, objects, format="person N")
    assert result == "Why is person 1 here?"

def test_resolve_references_multiple_same_type():
    tokens = [[0], "and", [1]]
    objects = ["person", "person"]
    result = resolve_references(tokens, objects, format="person N")
    assert result == "person 1 and person 2"

def test_resolve_references_multi_object():
    tokens = ["Look", "at", [0, 1]]
    objects = ["person", "person"]
    result = resolve_references(tokens, objects, format="person N")
    assert result == "Look at person 1 and person 2"

def test_resolve_references_different_types():
    tokens = [[0], "and", [1]]
    objects = ["person", "dog"]
    result = resolve_references(tokens, objects, format="person N")
    assert result == "person 1 and dog 1"

def test_resolve_references_tagged():
    tokens = [[0]]
    objects = ["person"]
    result = resolve_references(tokens, objects, format="tagged")
    assert result == "<person_1>"

def test_resolve_references_bracketed():
    tokens = [[0]]
    objects = ["person"]
    result = resolve_references(tokens, objects, format="bracketed")
    assert result == "[person 1]"

def test_build_object_name_map():
    objects = ["person", "dog"]
    obj_map = build_object_name_map(objects)
    assert obj_map[0] == "person 1"
    assert obj_map[1] == "dog 1"

def test_build_object_name_map_duplicates():
    objects = ["person", "person", "dog"]
    obj_map = build_object_name_map(objects)
    assert obj_map[0] == "person 1"
    assert obj_map[1] == "person 2"
    assert obj_map[2] == "dog 1"

def test_validate_vcr_raw_valid(sample_vcr_raw):
    errors = validate_vcr_raw(sample_vcr_raw)
    assert len(errors) == 0

def test_validate_vcr_raw_missing_fields(sample_vcr_raw):
    del sample_vcr_raw["question"]
    errors = validate_vcr_raw(sample_vcr_raw)
    assert len(errors) > 0

def test_validate_vcr_raw_wrong_answer_choices(sample_vcr_raw):
    sample_vcr_raw["answer_choices"].pop()
    errors = validate_vcr_raw(sample_vcr_raw)
    assert len(errors) > 0

def test_parse_vcr_sample(sample_vcr_raw):
    sample = parse_vcr_sample(sample_vcr_raw, vcr_dir=".")
    assert sample.is_valid
    assert sample.sample_id == "test-0"
