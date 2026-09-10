"""ingestion/txt_extractor.py

Phase 3.4 — TXT Extraction via standard file handling

Provides :class:`TxtExtractor` — the focused implementation of
TXT-to-text extraction for Milestone 1.

Design decisions
----------------
* Uses standard library file handling with UTF-8 encoding.
* Handles decoding issues safely using ``errors="replace"``.
* Preserves line structure and whitespace.
* Records metadata including ``line_count``, ``file_size_bytes``,
  and ``has_text``.
* Returns a dictionary with ``text`` and ``metadata`` matching the common ingestion interface.
"""

from __future__ import annotations

import os
from typing import Any, Dict


class TxtExtractor:
    """Extract text and metadata from a TXT file.

    Usage
    -----
    ::

        extractor = TxtExtractor()
        result = extractor.extract("data/technical/deployment_guide.txt")
        print(result["text"])
        print(result["metadata"]["line_count"])
    """

    def extract(self, file_path: str) -> Dict[str, Any]:
        """Open *file_path*, read text, return result dict.

        Parameters
        ----------
        file_path:
            Path to a ``.txt`` file. Must exist and be a regular file.

        Returns
        -------
        dict with keys:

        ``text`` : str
            Full text content with preserved line structure.

        ``metadata`` : dict with keys:
            ``source_location`` : str
            ``document_name`` : str
            ``file_size_bytes`` : int
            ``format`` : str (always "txt")
            ``line_count`` : int
            ``has_text`` : bool

        Raises
        ------
        FileNotFoundError
            If *file_path* does not exist.
        RuntimeError
            If reading the file fails.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path!r}")

        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as exc:
            raise RuntimeError(
                f"Could not read text file {file_path!r}: {exc}"
            ) from exc

        line_count = len(content.splitlines())
        has_text = len(content.strip()) > 0

        metadata: Dict[str, Any] = {
            "source_location": file_path,
            "document_name": filename,
            "file_size_bytes": file_size,
            "format": "txt",
            "line_count": line_count,
            "has_text": has_text,
        }

        return {
            "text": content,
            "metadata": metadata,
        }
