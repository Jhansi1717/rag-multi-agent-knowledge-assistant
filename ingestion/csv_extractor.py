"""ingestion/csv_extractor.py

Phase 3.4 — CSV Extraction via pandas

Provides :class:`CsvExtractor` — the focused implementation of
CSV-to-text extraction for Milestone 1.

Design decisions
----------------
* Uses ``pandas`` for reliable CSV parsing and dialect handling.
* Formats each row into a structured, semantic textual representation
  preserving column labels and row provenance.
  Example:
  ``Row 1: Leave_Type: Annual | Days_Allowed: 20 | Carry_Forward_Limit: 5``
* Captures metadata including ``row_count``, ``column_count``,
  ``column_names``, ``file_size_bytes``, and ``has_text``.
* Produces text that is semantically searchable in downstream vector search.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd


class CsvExtractor:
    """Extract text and tabular metadata from a CSV file using pandas.

    Usage
    -----
    ::

        extractor = CsvExtractor()
        result = extractor.extract("data/hr/leave_allowance.csv")
        print(result["text"])
        print(result["metadata"]["row_count"])
    """

    def extract(self, file_path: str) -> Dict[str, Any]:
        """Open *file_path*, parse CSV rows, return structured textual representation.

        Parameters
        ----------
        file_path:
            Path to a ``.csv`` file. Must exist and be a regular file.

        Returns
        -------
        dict with keys:

        ``text`` : str
            Semantic textual representation of all rows.

        ``metadata`` : dict with keys:
            ``source_location`` : str
            ``document_name`` : str
            ``file_size_bytes`` : int
            ``format`` : str (always "csv")
            ``row_count`` : int
            ``column_count`` : int
            ``column_names`` : list[str]
            ``has_text`` : bool

        Raises
        ------
        FileNotFoundError
            If *file_path* does not exist.
        RuntimeError
            If pandas cannot read or parse the CSV.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path!r}")

        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            raise RuntimeError(
                f"pandas could not read CSV {file_path!r}: {exc}"
            ) from exc

        column_names = [str(col) for col in df.columns]
        row_count = len(df)
        column_count = len(column_names)

        row_lines: List[str] = []
        for idx, row in df.iterrows():
            row_num = idx + 1  # 1-based index
            fields = [f"{col}: {row[col]}" for col in df.columns]
            row_lines.append(f"Row {row_num}: " + " | ".join(fields))

        full_text = "\n".join(row_lines)
        has_text = len(full_text.strip()) > 0

        metadata: Dict[str, Any] = {
            "source_location": file_path,
            "document_name": filename,
            "file_size_bytes": file_size,
            "format": "csv",
            "row_count": row_count,
            "column_count": column_count,
            "column_names": column_names,
            "has_text": has_text,
        }

        return {
            "text": full_text,
            "metadata": metadata,
        }
