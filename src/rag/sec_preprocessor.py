"""
SEC 10-K text preprocessor.

Strips EDGAR SGML wrappers, HTML/iXBRL tags, XBRL inline markup,
CSS/JS blocks, redundant boilerplate phrases, and duplicate paragraphs
before the text reaches the chunker.

Why this matters for cost:
  Raw EDGAR .txt files are 12-55 MB of which ~40-70% is noise
  (SGML headers, HTML tags, XBRL attributes, exhibit indexes).
  Stripping it here shrinks the token count before embedding,
  directly cutting OpenAI API spend.

Usage:
    from src.rag.sec_preprocessor import preprocess_sec_filing
    clean = preprocess_sec_filing(raw_text, filename="AMZN_10K.txt")
"""

import re
from typing import List, Optional

from ..utils.logging import get_logger

logger = get_logger(__name__)


# ── Boilerplate phrases that repeat across all 10-K filings ──────────────────
# Removing these reduces duplicate embeddings and cuts token count.
_BOILERPLATE_PATTERNS: List[str] = [
    r"This (?:Annual|Quarterly) Report (?:on Form \d+-[A-Z])? ?contains forward[- ]looking statements[^\n]*",
    r"Forward[- ]looking statements[^\n]*involve risks and uncertainties[^\n]*",
    r"We caution (?:you )?(?:not )?to (?:place )?(?:undue )?reliance on (?:these|such|any) forward[- ]looking[^\n]*",
    r"Actual results may differ materially from those anticipated[^\n]*",
    r"(?:Please )?[Ss]ee [\"']?Risk Factors[\"']? (?:in )?(?:Part I[,.]? )?Item 1A[^\n]*",
    r"[Tt]able of [Cc]ontents\s*(?:\n|\.{3,}|\s{2,})\s*(?:Page\s*)?\d+",
    r"Index to (?:Financial )?(?:Consolidated )?Statements?\s*F[-–]\d+",
    r"[Ss]ee [Aa]ccompanying [Nn]otes to (?:the )?(?:[Cc]onsolidated )?[Ff]inancial [Ss]tatements?",
    r"The (?:accompanying )?notes (?:to the )?(?:financial|consolidated) statements? are an? integral part",
    r"(?:All )?[Dd]ollar amounts? (?:are )?(?:in )?(?:expressed )?in (?:millions|billions|thousands)[^\n]*",
    r"\(in millions[^\)]*\)",
    r"\(in billions[^\)]*\)",
    r"(?:Incorporated )?[Hh]erein by reference",
    r"Not applicable\.?\s*$",
]
_BOILERPLATE_RE = re.compile(
    "|".join(_BOILERPLATE_PATTERNS), re.MULTILINE | re.DOTALL
)

# ── Exhibit index section ─────────────────────────────────────────────────────
# The exhibit index is a long table of exhibit numbers/descriptions.
# It contributes no useful Q&A content.
_EXHIBIT_SECTION_RE = re.compile(
    r"(?:ITEM\s+15\.?|Exhibit\s+(?:Index|List|No\.|Number))[^\n]*\n"
    r"(?:.*\n){0,5}(?:(?:\d+\.\d+|[A-Z]\d*)\s+[^\n]+\n){3,}",
    re.IGNORECASE | re.MULTILINE,
)

# ── EDGAR SGML structure ──────────────────────────────────────────────────────
_SGML_HEADER_RE = re.compile(
    r"<SEC-DOCUMENT>.*?<TEXT>",
    re.DOTALL | re.IGNORECASE,
)
_SGML_CLOSE_RE = re.compile(
    r"</TEXT>.*",
    re.DOTALL | re.IGNORECASE,
)
_SGML_TAG_RE = re.compile(
    r"<(?:SEC-DOCUMENT|SEC-HEADER|DOCUMENT|SEQUENCE|FILENAME|DESCRIPTION|TYPE|TEXT)"
    r"[^>]*>",
    re.IGNORECASE,
)

# ── HTML / XBRL cleanup ───────────────────────────────────────────────────────
_STYLE_BLOCK_RE = re.compile(r"<style[^>]*>.*?</style>", re.DOTALL | re.IGNORECASE)
_SCRIPT_BLOCK_RE = re.compile(r"<script[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)
_XBRL_CONTEXT_RE = re.compile(r"<(?:xbrli?|ix|us-gaap|dei)[^\s>]*[^>]*>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]{1,500}>")  # length cap avoids catastrophic backtracking
_HTML_ENTITY_RE = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z]{2,8});")

_HTML_ENTITIES = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
    "&apos;": "'", "&nbsp;": " ", "&ndash;": "-", "&mdash;": "-",
    "&ldquo;": '"', "&rdquo;": '"', "&lsquo;": "'", "&rsquo;": "'",
    "&trade;": "™", "&reg;": "®", "&copy;": "©",
}


def _decode_entities(text: str) -> str:
    for entity, replacement in _HTML_ENTITIES.items():
        text = text.replace(entity, replacement)
    # Remaining numeric entities
    text = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), text)
    text = re.sub(r"&#x([0-9a-fA-F]+);", lambda m: chr(int(m.group(1), 16)), text)
    return text


def _strip_html_xbrl(text: str) -> str:
    """Remove all HTML and XBRL markup, keeping only the visible text."""
    text = _STYLE_BLOCK_RE.sub(" ", text)
    text = _SCRIPT_BLOCK_RE.sub(" ", text)
    text = _XBRL_CONTEXT_RE.sub("", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _decode_entities(text)
    return text


# ── Whitespace normalisation ──────────────────────────────────────────────────

def _normalise_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of spaces/tabs on the same line
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Trim trailing spaces from each line
    text = re.sub(r" +\n", "\n", text)
    # Collapse 3+ blank lines to two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Duplicate-paragraph removal ───────────────────────────────────────────────

def _dedup_paragraphs(text: str, min_len: int = 80) -> str:
    """
    Remove exact duplicate paragraphs (common in SEC filings where the
    same disclaimer block repeats across multiple documents in the filing).
    Only paragraphs longer than `min_len` characters are deduplicated to
    avoid removing intentional one-liners (e.g. section headers).
    """
    seen: set = set()
    out: List[str] = []
    for para in text.split("\n\n"):
        stripped = para.strip()
        key = re.sub(r"\s+", " ", stripped).lower()
        if len(key) >= min_len:
            if key in seen:
                continue
            seen.add(key)
        out.append(para)
    return "\n\n".join(out)


# ── Noise-line removal ────────────────────────────────────────────────────────
# Lines that are purely numbers, dots, dashes (page separators, TOC dots)
_NOISE_LINE_RE = re.compile(
    r"^[\s\d\.\-_=\|•·\*]+$"
)

def _remove_noise_lines(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if not _NOISE_LINE_RE.match(line.strip()):
            lines.append(line)
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def is_sec_filing(filename: str) -> bool:
    """Return True if the filename looks like an SEC 10-K filing."""
    name = filename.upper()
    return "_10K" in name or "10-K" in name or name.endswith("_10K.TXT")


def preprocess_sec_filing(raw_text: str, filename: str = "") -> str:
    """
    Full preprocessing pipeline for SEC EDGAR 10-K text files.

    Steps:
      1. Extract the main <TEXT> block (skip SGML metadata headers)
      2. Strip HTML, CSS, JS, and XBRL inline tags
      3. Remove boilerplate phrases that repeat across all filings
      4. Remove exhibit index tables
      5. Remove noise lines (pure numbers/dots/dashes)
      6. Normalise whitespace
      7. Remove exact duplicate paragraphs

    Returns cleaned plain text ready for chunking.
    """
    original_len = len(raw_text)

    # 1. Extract main TEXT block from EDGAR SGML wrapper
    #    The filing may contain multiple <DOCUMENT> blocks; the first 10-K
    #    document is the most useful. We strip everything outside <TEXT>...</TEXT>.
    text_match = re.search(r"<TEXT>(.*?)</TEXT>", raw_text, re.DOTALL | re.IGNORECASE)
    if text_match:
        text = text_match.group(1)
    else:
        # No SGML wrapper – file is already plain text or HTML
        text = raw_text

    # 2. Strip HTML / XBRL
    text = _strip_html_xbrl(text)

    # 3. Remove boilerplate
    text = _BOILERPLATE_RE.sub(" ", text)

    # 4. Remove exhibit index
    text = _EXHIBIT_SECTION_RE.sub("\n", text)

    # 5. Remove noise lines
    text = _remove_noise_lines(text)

    # 6. Normalise whitespace
    text = _normalise_whitespace(text)

    # 7. Dedup paragraphs
    text = _dedup_paragraphs(text)

    cleaned_len = len(text)
    reduction = (1 - cleaned_len / max(original_len, 1)) * 100
    logger.info(
        f"SEC preprocess [{filename}]: "
        f"{original_len:,} -> {cleaned_len:,} chars  "
        f"(-{reduction:.0f}%)"
    )

    return text


def preprocess_generic(raw_text: str) -> str:
    """
    Lightweight preprocessing for non-SEC documents (PDF/TXT that aren't EDGAR files).
    Only strips HTML tags and normalises whitespace.
    """
    text = _strip_html_xbrl(raw_text)
    text = _normalise_whitespace(text)
    return text
