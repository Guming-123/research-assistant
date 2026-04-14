"""Utility modules for the Multi-Agent Literature Review System"""

from .llm import get_llm_client
from .pdf import extract_text_from_pdf, extract_metadata_from_pdf
from .text import chunk_text, normalize_text
from .embedding import get_embeddings, compute_similarity
from .api import SemanticScholarAPI, ArxivAPI

__all__ = [
    "get_llm_client",
    "extract_text_from_pdf",
    "extract_metadata_from_pdf",
    "chunk_text",
    "normalize_text",
    "get_embeddings",
    "compute_similarity",
    "SemanticScholarAPI",
    "ArxivAPI",
]
