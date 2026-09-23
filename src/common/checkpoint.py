"""Pipeline checkpointing to support resumability."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any

from src.common.logging import get_logger

logger = get_logger("common.checkpoint")

@dataclass
class StageProgress:
    """Tracks progress for a single pipeline stage."""
    stage_name: str
    _completed_set: set = field(default_factory=set)
    total_units: int = 0
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def completed_units(self) -> List[str]:
        """Sorted list view for backward compatibility."""
        return sorted(self._completed_set)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes to dict."""
        return {
            "stage_name": self.stage_name,
            "completed_units": sorted(self._completed_set),
            "total_units": self.total_units,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageProgress":
        """Deserializes from dict."""
        return cls(
            stage_name=data["stage_name"],
            _completed_set=set(data.get("completed_units", [])),
            total_units=data.get("total_units", 0),
            last_updated=data.get("last_updated", datetime.now(timezone.utc).isoformat()),
        )


class PipelineCheckpoint:
    """
    Tracks pipeline progress via an atomic JSON file.
    Allows stages to resume from where they left off after a crash.
    """

    def __init__(self, checkpoint_path: str):
        """Initializes the checkpoint manager."""
        self.checkpoint_path = checkpoint_path
        self._stages: Dict[str, StageProgress] = {}
        self._load()

    def _load(self):
        """Loads checkpoint state from disk if it exists."""
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                for stage_name, stage_data in data.items():
                    self._stages[stage_name] = StageProgress.from_dict(stage_data)
                
                logger.info(f"Loaded checkpoint from {self.checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to load checkpoint from {self.checkpoint_path}: {e}")
                self._stages = {}
        else:
            logger.info(f"No existing checkpoint found at {self.checkpoint_path}. Starting fresh.")

    def _get_or_create_stage(self, stage: str) -> StageProgress:
        """Gets existing stage progress or creates a new one."""
        if stage not in self._stages:
            self._stages[stage] = StageProgress(stage_name=stage)
        return self._stages[stage]

    def mark_complete(self, stage: str, unit_id: str):
        """Marks a specific unit as complete for a given stage."""
        progress = self._get_or_create_stage(stage)
        progress._completed_set.add(unit_id)
        progress.last_updated = datetime.now(timezone.utc).isoformat()
            
    def is_complete(self, stage: str, unit_id: str) -> bool:
        """Checks if a specific unit is marked as complete for a given stage."""
        if stage not in self._stages:
            return False
        return unit_id in self._stages[stage]._completed_set

    def get_progress(self, stage: str) -> StageProgress:
        """Gets progress for a specific stage."""
        return self._get_or_create_stage(stage)

    def reset_stage(self, stage: str):
        """Resets progress for a specific stage."""
        if stage in self._stages:
            self._stages[stage] = StageProgress(stage_name=stage)
            
    def save(self):
        """
        Persists checkpoint state to disk atomically.
        Writes to a temporary file first, then renames.
        """
        data = {name: progress.to_dict() for name, progress in self._stages.items()}
        temp_path = f"{self.checkpoint_path}.tmp"
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.checkpoint_path)), exist_ok=True)
            
            # Write to temp file
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                
            # Atomic rename (replace)
            os.replace(temp_path, self.checkpoint_path)
            logger.debug(f"Saved checkpoint to {self.checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
