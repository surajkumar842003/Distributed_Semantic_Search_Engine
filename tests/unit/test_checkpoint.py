import os
import json
import pytest
import tempfile
from src.common.checkpoint import PipelineCheckpoint

@pytest.fixture
def temp_checkpoint_path():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.remove(path)
    if os.path.exists(f"{path}.tmp"):
        os.remove(f"{path}.tmp")

def test_checkpoint_create_and_mark(temp_checkpoint_path):
    cp = PipelineCheckpoint(temp_checkpoint_path)
    assert not cp.is_complete("stage1", "unit1")
    
    cp.mark_complete("stage1", "unit1")
    assert cp.is_complete("stage1", "unit1")
    assert not cp.is_complete("stage1", "unit2")
    
    progress = cp.get_progress("stage1")
    assert progress.stage_name == "stage1"
    assert "unit1" in progress.completed_units

def test_checkpoint_persistence(temp_checkpoint_path):
    cp1 = PipelineCheckpoint(temp_checkpoint_path)
    cp1.mark_complete("stage1", "unitA")
    cp1.mark_complete("stage2", "unitB")
    cp1.save()
    
    cp2 = PipelineCheckpoint(temp_checkpoint_path)
    assert cp2.is_complete("stage1", "unitA")
    assert cp2.is_complete("stage2", "unitB")
    assert not cp2.is_complete("stage1", "unitB")

def test_checkpoint_atomic_write(temp_checkpoint_path):
    cp = PipelineCheckpoint(temp_checkpoint_path)
    cp.mark_complete("stage1", "unit1")
    cp.save()
    
    # Verify file exists and is not a tmp file
    assert os.path.exists(temp_checkpoint_path)
    assert not os.path.exists(f"{temp_checkpoint_path}.tmp")
    
    with open(temp_checkpoint_path, "r") as f:
        data = json.load(f)
    assert "stage1" in data
    assert "unit1" in data["stage1"]["completed_units"]

def test_checkpoint_skip_completed(temp_checkpoint_path):
    cp = PipelineCheckpoint(temp_checkpoint_path)
    cp.mark_complete("stage1", "unit1")
    cp.save()
    
    # Reload and simulate skipping
    cp2 = PipelineCheckpoint(temp_checkpoint_path)
    if cp2.is_complete("stage1", "unit1"):
        skipped = True
    else:
        skipped = False
    
    assert skipped
