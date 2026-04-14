"""
PDF processing utilities
PDF文档解析和文本提取
"""

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import aiofiles

logger = logging.getLogger(__name__)


async def extract_text_from_pdf(
    pdf_path: str,
    include_references: bool = False,
    clean_whitespace: bool = True,
) -> str:
    """
    从PDF提取文本

    Args:
        pdf_path: PDF文件路径
        include_references: 是否包含参考文献部分
        clean_whitespace: 是否清理空白字符

    Returns:
        提取的文本
    """
    try:
        import PyPDF2

        text_parts = []
        async with aiofiles.open(pdf_path, "rb") as f:
            content = await f.read()
            import io
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue

        full_text = "\n\n".join(text_parts)

        # 清理文本
        if clean_whitespace:
            full_text = clean_pdf_text(full_text)

        # 移除参考文献（如果不需要）
        if not include_references:
            full_text = remove_references_section(full_text)

        return full_text

    except ImportError:
        logger.warning("PyPDF2 not available, using fallback method")
        return await _extract_text_fallback(pdf_path)
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        return ""


async def _extract_text_fallback(pdf_path: str) -> str:
    """备用文本提取方法"""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        text_parts = []

        for page in doc:
            text_parts.append(page.get_text())

        doc.close()
        return "\n\n".join(text_parts)

    except ImportError:
        logger.error("Neither PyPDF2 nor PyMuPDF available")
        return ""


def clean_pdf_text(text: str) -> str:
    """
    清理PDF文本

    Args:
        text: 原始文本

    Returns:
        清理后的文本
    """
    # 移除过长的空白行
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 统一引号
    text = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")

    # 移除页码（简单模式）
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

    # 移除页眉页脚（简单模式）
    text = re.sub(r'\n.*?Proceedings.*?\n', '\n', text, flags=re.IGNORECASE)

    return text.strip()


def remove_references_section(text: str) -> str:
    """
    移除参考文献部分

    Args:
        text: 原始文本

    Returns:
        移除参考文献后的文本
    """
    # 常见的参考文献起始模式
    patterns = [
        r'\n\s*References\s*\n',
        r'\n\s*BIBLIOGRAPHY\s*\n',
        r'\n\s*Reference\s*\n',
        r'\n\s*Citations\s*\n',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return text[:match.start()].strip()

    return text


async def extract_metadata_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """
    从PDF提取元数据

    Args:
        pdf_path: PDF文件路径

    Returns:
        元数据字典
    """
    metadata = {
        "title": "",
        "author": "",
        "subject": "",
        "creator": "",
        "producer": "",
        "creation_date": "",
        "page_count": 0,
    }

    try:
        import PyPDF2
        import io

        async with aiofiles.open(pdf_path, "rb") as f:
            content = await f.read()
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            # 获取元数据
            info = pdf_reader.metadata
            if info:
                metadata.update({
                    "title": info.get("/Title", ""),
                    "author": info.get("/Author", ""),
                    "subject": info.get("/Subject", ""),
                    "creator": info.get("/Creator", ""),
                    "producer": info.get("/Producer", ""),
                    "creation_date": info.get("/CreationDate", ""),
                })

            metadata["page_count"] = len(pdf_reader.pages)

    except Exception as e:
        logger.error(f"Failed to extract metadata from PDF: {e}")

    return metadata


def extract_citations(text: str) -> List[str]:
    """
    从文本中提取引用

    Args:
        text: 文本内容

    Returns:
        引用列表
    """
    # 简单的引用提取模式
    # 如 [1], [Smith et al., 2020], (Smith, 2020)
    patterns = [
        r'\[\d+(?:,\s*\d+)*\]',  # [1], [1, 2, 3]
        r'\([A-Z][a-z]+(?:\s+et\s+al.)?(?:,\s*\d{4})?\)',  # (Smith, 2020), (Smith et al., 2020)
    ]

    citations = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        citations.extend(matches)

    return list(set(citations))


def extract_sections(text: str) -> Dict[str, str]:
    """
    从文本中提取各个部分

    Args:
        text: 文本内容

    Returns:
        各部分的字典 {section_name: content}
    """
    sections = {}

    # 常见的章节标题模式
    section_patterns = {
        "abstract": r'(?:Abstract|摘要)\s*[:\n](.*?)(?=\n\s*(?:Introduction|1\.|Keywords|关键词))',
        "introduction": r'(?:Introduction|引言)\s*[:\n](.*?)(?=\n\s*(?:Related Work|2\.|Methodology|方法))',
        "conclusion": r'(?:Conclusion|结论)\s*[:\n](.*?)(?=\n\s*(?:References|参考文献))',
    }

    for section_name, pattern in section_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            sections[section_name] = match.group(1).strip()

    return sections
