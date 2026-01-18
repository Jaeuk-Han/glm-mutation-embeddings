"""FASTA utilities for dataset preparation.

This module intentionally stays dependency-light and supports both plain text
FASTA (.fa/.fasta) and gzip-compressed FASTA (.gz).

Design goals:
- Simple and robust parsing.
- Return a {chrom_name: sequence} dict for fast random access.
- Upper-case sequences.

Note: Loading an entire GRCh38 FASTA into memory can take multiple GB.
"""

from __future__ import annotations

from typing import Dict, Iterable, TextIO
import gzip
import io
import os


def _open_text_maybe_gzip(path: str) -> TextIO:
    """Open a text file that may be gzip-compressed."""
    # Heuristic: if filename ends with .gz, open via gzip.
    # (We avoid sniffing magic bytes to keep it simple/fast.)
    if path.endswith(".gz"):
        return gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return open(path, mode="rt", encoding="utf-8", newline="")


def load_fasta_as_dict(path: str) -> Dict[str, str]:
    """Load a FASTA file into a dict: {chrom_name: sequence}.

    Args:
        path: FASTA file path. Supports .fa/.fasta and .gz variants.

    Returns:
        Dict mapping sequence name (first token after '>') to uppercase sequence.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the FASTA is malformed.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"FASTA file not found: {path}")

    chrom_seqs: Dict[str, str] = {}
    chrom_name: str | None = None
    buf: list[str] = []

    with _open_text_maybe_gzip(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # flush previous sequence
                if chrom_name is not None:
                    chrom_seqs[chrom_name] = "".join(buf).upper()

                chrom_name = line[1:].split()[0]
                if not chrom_name:
                    raise ValueError("Malformed FASTA header: empty sequence name")
                buf = []
            else:
                if chrom_name is None:
                    raise ValueError("Malformed FASTA: sequence data before any header")
                buf.append(line)

    if chrom_name is not None:
        chrom_seqs[chrom_name] = "".join(buf).upper()

    return chrom_seqs
