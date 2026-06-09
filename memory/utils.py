import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

def save_metadata(path: str, metadata: List[Dict[str, Any]], store_name: str) -> None:
    """Save metadata list to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False)
    logger.info(
        "%s metadata saved to %s (%d entries)",
        store_name,
        path,
        len(metadata),
    )

def load_metadata(path: str, store_name: str) -> List[Dict[str, Any]]:
    """Load metadata list from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    logger.info(
        "%s metadata loaded (%d entries)", store_name, len(metadata)
    )
    return metadata
