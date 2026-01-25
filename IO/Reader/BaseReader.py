from typing import Dict, Any
from pathlib import Path

class BaseReader:
    """
    Base interface for all file readers.
    Reader = sensory input (no thinking, no learning)
    """

    file_type: str = "unknown"
    supported_extensions: tuple[str, ...] = ()

    def can_read(self, file_path: str) -> bool:
        """Check whether this reader supports the file"""
        return Path(file_path).suffix.lower() in self.supported_extensions

    def read(self, file_path: str) -> Dict[str, Any]:
        """
        Read file and return normalized content

        Required output format:
        {
            source: "file",
            file_type: str,
            file_path: str,
            content: str,
            structure: list,
            metadata: dict
        }
        """
        raise NotImplementedError
