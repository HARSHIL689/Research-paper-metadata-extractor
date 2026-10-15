"""
extract_metadata_smart.py

Header-first scholarly PDF metadata extraction with selective GROBID fulltext fallback.

Design goals:
- Do NOT extract the paper body by default.
- Use GROBID /processHeaderDocument first.
- Inspect only the first few PDF pages locally when header metadata needs help.
- Call /processFulltextDocument only when metadata is still materially incomplete
  and the missing information is reasonably likely to be recoverable from fulltext.
- Never treat ordinary all-caps person names as organizations.
- Remove organization/affiliation blocks that GROBID accidentally emits as authors.
- Recover author emails and numbered affiliations from the first pages when possible.
- Preserve legitimate single-author papers.
- Avoid contaminating abstracts with author footnotes.
- Distinguish an arXiv version date from venue information when possible.
- Never return the full paper text; fulltext is used only as a metadata fallback.

Dependencies:
    pip install requests lxml pymupdf

Usage:
    python extract_metadata_smart_fixed_v2.py paper.pdf
    python extract_metadata_smart_fixed_v2.py paper.pdf --debug
    python extract_metadata_smart_fixed_v2.py --input ./papers --output results.jsonl --workers 4
    python extract_metadata_smart_fixed_v2.py paper.pdf --force-fulltext
"""

import sys
import json
import re
import os
import glob
import time
import traceback
import unicodedata
from typing import Dict, List, Optional, Any, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter, Retry
from lxml import etree

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


# ============================================================================
# Constants
# ============================================================================

GROBID_HEADER_URL = "http://localhost:8070/api/processHeaderDocument"
GROBID_FULLTEXT_URL = "http://localhost:8070/api/processFulltextDocument"
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

# Only inspect a small number of PDF pages locally. This is intentionally NOT
# the whole paper and does not create a fulltext result.
FIRST_PAGES_TO_INSPECT = 3

ARXIV_ID_PATTERN = re.compile(
    r"(?:arXiv\s*:\s*|(?<![\d.]))"
    r"("
    r"(?:\d{4}\.\d{4,5}(?:v\d+)?)"
    r"|"
    r"(?:[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)"
    r")"
    r"(?![\d.])",
    re.IGNORECASE,
)

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b", re.IGNORECASE)

# Conservative organization detection. This is used ONLY for an author field,
# never for affiliation text.
ORG_NAMES: Set[str] = {
    "google", "google brain", "google research", "google ai", "google cloud",
    "microsoft", "microsoft research", "microsoft ai", "facebook", "facebook ai",
    "meta", "meta ai", "amazon", "amazon web services", "aws", "ibm", "ibm research",
    "apple", "apple ai", "nvidia", "nvidia research", "intel", "intel labs",
    "qualcomm", "openai", "deepmind", "anthropic", "cohere", "inflection",
    "google brain research", "google research brain",
    "nips", "neurips", "icml", "iclr", "acl", "emnlp", "naacl", "eacl",
    "cvpr", "iccv", "eccv", "siggraph", "sigcomm", "sigmod", "vldb",
}

ORG_SINGLE_WORDS = {
    "google", "microsoft", "facebook", "meta", "amazon", "aws", "ibm",
    "apple", "nvidia", "intel", "qualcomm", "openai", "deepmind",
    "anthropic", "cohere", "inflection",
}

LEGAL_SUFFIX_RE = re.compile(
    r"\s+(?:Inc|Corp|Corporation|Ltd|LLC|GmbH|SAS|Sarl|BV|PLC|AG|KG|MBH)\.?$",
    re.IGNORECASE,
)

SUSPICIOUS_AUTHOR_PATTERNS = [
    r"^(?:Google|Facebook|Microsoft|Amazon|IBM|Apple|Meta|OpenAI|DeepMind|Anthropic|Cohere|NVIDIA|Intel|Qualcomm)(?:\s+(?:Brain|Research|AI|Labs?|Cloud|Web Services))?$",
    r"^[A-Z][A-Za-z&.\- ]+\s+(?:Inc|Corp|Corporation|Ltd|LLC|GmbH|SAS|Sarl|BV|PLC|AG|KG|MBH)\.?$",
    r"^(?:University|Institute|College|School|Laboratory|Lab|Center|Centre|Department|Faculty|Academy|Foundation|Consortium)(?:\s+of)?\s+.+$",
    r"^(?:NIPS|NeurIPS|ICML|ICLR|ACL|EMNLP|NAACL|EACL|CVPR|ICCV|ECCV|SIGGRAPH|SIGCOMM|SIGMOD|VLDB)$",
]

AFFILIATION_HINT_RE = re.compile(
    r"\b(?:university|universit|institute|institut|school|college|department|faculty|"
    r"research|brain|laborator|laboratory|lab|center|centre|hospital|clinic|"
    r"academy|infn|cnao|cnrs|iphc|gsi|technische|inria|mit|stanford|"
    r"google|microsoft|ibm|amazon|openai|deepmind|nvidia|toronto|washington)\b",
    re.IGNORECASE,
)

# 1. Keep your existing keywords and append the new broader ones
VENUE_HINT_RE = re.compile(
    r"\b(?:conference|symposium|workshop|proceedings|journal|transactions|"
    r"neurips|nips|icml|iclr|acl|emnlp|cvpr|iccv|eccv|aaai|ijcai|sigir|"
    r"nature|science|springer|elsevier|arxiv|plos one|ieee|acm|"
    r"advances in|frontiers in|annals of|review of|de gruyter)\b",
    re.IGNORECASE,
)

# New RegEx clues for detecting structure on tricky pages
VOL_ARTICLE_RE = re.compile(r'\bVolume\s+\d+.*?Article\b', re.IGNORECASE)
DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
ISSN_PATTERN = re.compile(r"\bISSN\s*[:\-]?\s*(\d{4}[-–]\d{3}[0-9X])\b", re.IGNORECASE)
EISSN_PATTERN = re.compile(r"\be[- ]?ISSN\s*[:\-]?\s*(\d{4}[-–]\d{3}[0-9X])\b", re.IGNORECASE)
ISBN_PATTERN = re.compile(r"\bISBN(?:-1[03])?\s*[:\-]?\s*((?:97[89][\- ]?)?[0-9][0-9\- ]{8,16}[0-9X])\b", re.IGNORECASE)
ARTICLE_TYPE_PATTERNS = [
    ("Research Article", re.compile(r"^\s*(?:RESEARCH\s+ARTICLE|Research\s+Article)\s*$", re.I)),
    ("Original Research", re.compile(r"^\s*Original\s+Research\s*$", re.I)),
    ("Review Article", re.compile(r"^\s*Review\s+Article\s*$", re.I)),
    ("Systematic Review", re.compile(r"^\s*Systematic\s+Review\s*$", re.I)),
    ("Short Paper", re.compile(r"^\s*SHORT[- ]PAPER\s*$", re.I)),
    ("Conference Paper", re.compile(r"^\s*Conference\s+Paper\s*$", re.I)),
    ("Case Report", re.compile(r"^\s*Case\s+Report\s*$", re.I)),
]

FOOTNOTE_MARKER_RE = re.compile(
    r"\s+(?:[*†‡§]+\s*)?(?:equal contribution|work performed while|deceased|"
    r"corresponding author(?:s)?|correspondence)\b.*$",
    re.IGNORECASE,
)


# ============================================================================
# General utilities
# ============================================================================

def make_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 502, 503, 504],
        allowed_methods=["POST"],
    )
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def _text_or_none(node) -> Optional[str]:
    if node is None:
        return None
    text = " ".join(node.itertext()).strip()
    return " ".join(text.split()) or None


def _get_attr_or_none(node, attr: str) -> Optional[str]:
    if node is None:
        return None
    value = node.get(attr)
    return value.strip() if value else None


def normalize_for_match(text: str) -> str:
    """Normalize Unicode/punctuation enough for fuzzy PDF name matching."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = text.casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def author_name_key(name: str) -> str:
    return normalize_for_match(name)


def clean_author_name(name: str) -> str:
    if not name:
        return ""
    return " ".join(name.strip().split())


def clean_abstract(text: Optional[str]) -> Optional[str]:
    """Remove obvious author-note contamination from an abstract."""
    if not text:
        return None
    text = " ".join(text.split()).strip()
    text = FOOTNOTE_MARKER_RE.sub("", text).strip()
    # GROBID sometimes appends several footnote sentences after asterisks.
    text = re.split(
        r"\s+[†‡]\s*(?:work performed|deceased|corresponding author)",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    return text or None


def clean_arxiv_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip()
    value = re.sub(r"^arXiv:", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\[.*?\]", "", value)
    return value.strip() or None


# ============================================================================
# Organization filtering
# ============================================================================

def is_organization_name(name: str) -> bool:
    """Conservatively identify organization strings accidentally emitted as authors."""
    if not name:
        return True

    original = " ".join(name.strip().split())
    normalized = original.casefold()

    if normalized in ORG_NAMES:
        return True
    if len(normalized.split()) == 1 and normalized in ORG_SINGLE_WORDS:
        return True

    for pattern in SUSPICIOUS_AUTHOR_PATTERNS:
        if re.fullmatch(pattern, original, re.IGNORECASE):
            return True

    if LEGAL_SUFFIX_RE.search(original):
        return True

    return False


def _extract_author_name(author_elem) -> Optional[str]:
    persname = author_elem.find(".//tei:persName", TEI_NS)
    if persname is not None:
        text = _text_or_none(persname)
        if text:
            return text

    forename = author_elem.find(".//tei:forename", TEI_NS)
    surname = author_elem.find(".//tei:surname", TEI_NS)
    fore = _text_or_none(forename)
    sur = _text_or_none(surname)
    if fore and sur:
        return f"{fore} {sur}"
    if fore:
        return fore
    if sur:
        return sur
    return None


def count_filtered_organizations(root) -> int:
    count = 0
    for author_elem in root.findall(".//tei:sourceDesc//tei:author", TEI_NS):
        name = _extract_author_name(author_elem)
        if name and is_organization_name(name):
            count += 1
    return count


# ============================================================================
# TEI affiliation/email parsing
# ============================================================================

def clean_affiliation_text(text: Optional[str]) -> Optional[str]:
    """Remove obvious GROBID author-footnote contamination from affiliations."""
    if not text:
        return None
    text = " ".join(text.split()).strip()
    text = re.sub(
        r"^(?:[*†‡§]+\s*)?(?:work performed while|work performed|deceased|corresponding author).*?(?=(?:University|Google|INFN|Institute|Department|School|College|Laboratory|Lab|Center|Centre|Faculty|CNRS|CNAO|IPHC|GSI)\b|$)",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    text = re.sub(
        r"\s+\d+(?:st|nd|rd|th)?\s+Conference on Neural Information Processing Systems.*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return text.rstrip(" .;,:") or None


def extract_affiliation_from_author(author_elem) -> Optional[str]:
    aff_elems = author_elem.findall(".//tei:affiliation", TEI_NS)
    if not aff_elems:
        return None

    values = []
    for aff in aff_elems:
        text = clean_affiliation_text(_text_or_none(aff))
        if text and text not in values:
            values.append(text)
    return " ; ".join(values) if values else None


def parse_author(author_elem) -> Optional[Dict[str, Any]]:
    name = _extract_author_name(author_elem)
    if not name or is_organization_name(name):
        return None

    name = clean_author_name(name)
    if not name:
        return None

    return {
        "name": name,
        "affiliation": extract_affiliation_from_author(author_elem),
        "email": _text_or_none(author_elem.find(".//tei:email", TEI_NS)),
    }


def parse_authors_from_tei(root, max_authors: Optional[int] = None) -> List[Dict[str, Any]]:
    """Parse real people and merge duplicate GROBID author blocks."""
    authors: List[Dict[str, Any]] = []
    by_name: Dict[str, Dict[str, Any]] = {}

    for elem in root.findall(".//tei:sourceDesc//tei:author", TEI_NS):
        data = parse_author(elem)
        if not data:
            continue

        key = author_name_key(data["name"])
        if key in by_name:
            existing = by_name[key]
            for field in ("affiliation", "email"):
                value = data.get(field)
                if not value:
                    continue
                if not existing.get(field):
                    existing[field] = value
                elif value != existing[field]:
                    vals = [v.strip() for v in existing[field].split(" ; ") if v.strip()]
                    if value not in vals:
                        vals.append(value)
                    existing[field] = " ; ".join(vals)
            continue

        by_name[key] = data
        authors.append(data)
        if max_authors and len(authors) >= max_authors:
            break

    return authors


def parse_language(root) -> str:
    header = root.find(".//tei:teiHeader", TEI_NS)
    return (
        (header.get(XML_LANG) if header is not None else None)
        or root.get(XML_LANG)
        or root.get("lang")
        or "unknown"
    )


def parse_tei_common(tei_xml: bytes) -> etree._Element:
    try:
        return etree.fromstring(tei_xml)
    except (etree.XMLSyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid TEI XML from GROBID: {exc}") from exc


def extract_tei_metadata(root) -> Dict[str, Any]:
    """Extract metadata only. Never returns body/fulltext content."""
    result: Dict[str, Any] = {
        "title": _text_or_none(root.find(".//tei:titleStmt/tei:title", TEI_NS)),
        "authors": parse_authors_from_tei(root),
        "abstract": None,
        "publication_date": None,
        "doi": None,
        "journal_or_venue": None,
        "arxiv_id": None,
        "keywords": [],
        "language": parse_language(root),
    }

    abstract_elem = root.find(".//tei:profileDesc/tei:abstract", TEI_NS)
    if abstract_elem is not None:
        p = abstract_elem.find(".//tei:p", TEI_NS)
        result["abstract"] = clean_abstract(_text_or_none(p if p is not None else abstract_elem))

    date_node = root.find(".//tei:publicationStmt/tei:date", TEI_NS)
    result["publication_date"] = _get_attr_or_none(date_node, "when") or _text_or_none(date_node)

    doi_node = root.find(".//tei:sourceDesc//tei:idno[@type='DOI']", TEI_NS)
    result["doi"] = _text_or_none(doi_node)

    # GROBID often stores the venue in monogr/title. For some conference papers,
    # the venue is only visible in the PDF first page, handled separately below.
    venue_node = root.find(".//tei:sourceDesc//tei:monogr/tei:title", TEI_NS)
    result["journal_or_venue"] = _text_or_none(venue_node)

    arxiv_node = root.find(".//tei:sourceDesc//tei:idno[@type='arXiv']", TEI_NS)
    result["arxiv_id"] = clean_arxiv_id(_text_or_none(arxiv_node))

    result["keywords"] = [
        text for text in (_text_or_none(t) for t in root.findall(
            ".//tei:profileDesc//tei:keywords//tei:term", TEI_NS
        )) if text
    ]

    return result


def parse_tei_header(tei_xml: bytes) -> Dict[str, Any]:
    return extract_tei_metadata(parse_tei_common(tei_xml))


def parse_tei_fulltext(tei_xml: bytes) -> Dict[str, Any]:
    root = parse_tei_common(tei_xml)
    result = extract_tei_metadata(root)

    # Fulltext TEI can place the abstract outside profileDesc.
    if not result.get("abstract"):
        body = root.find(".//tei:text/tei:body", TEI_NS)
        if body is not None:
            for head in body.findall(".//tei:head", TEI_NS):
                if "abstract" not in (_text_or_none(head) or "").lower():
                    continue
                nxt = head.getnext()
                if nxt is not None and nxt.tag == f"{{{TEI_NS['tei']}}}p":
                    result["abstract"] = clean_abstract(_text_or_none(nxt))
                    break

    return result


# ============================================================================
# PDF first-pages metadata rescue
# ============================================================================

def extract_first_pages_text(pdf_path: str, max_pages: int = FIRST_PAGES_TO_INSPECT) -> str:
    """Extract text from only the first N pages. Does not extract/store paper body."""
    if fitz is None:
        return ""

    try:
        doc = fitz.open(pdf_path)
        chunks = []
        for page_index in range(min(max_pages, len(doc))):
            chunks.append(doc[page_index].get_text("text"))
        doc.close()
        return "\n".join(chunks)
    except Exception:
        return ""


def _line_is_known_author(line: str, authors_by_key: Dict[str, Dict[str, Any]]) -> bool:
    key = normalize_for_match(line)
    return bool(key and key in authors_by_key)


def _author_regex(name: str) -> re.Pattern:
    """Create a regex tolerant of punctuation/whitespace differences in names."""
    tokens = re.findall(r"[\w]+", unicodedata.normalize("NFKC", name), flags=re.UNICODE)
    if not tokens:
        return re.compile(r"$a")
    return re.compile(r"\s*[^\w\n]*\s*".join(re.escape(t) for t in tokens), re.IGNORECASE | re.UNICODE)


def _extract_affiliation_blocks(text: str) -> Dict[str, str]:
    """Parse simple numbered affiliation blocks such as '1University of ...'."""
    lines = [line.strip() for line in text.splitlines()]
    blocks: Dict[str, List[str]] = {}
    current_num: Optional[str] = None

    for line in lines:
        if not line:
            continue
        m = re.match(r"^(\d{1,3})\s*(?=[A-Za-zÀ-ÿ])(.+)$", line)
        if m:
            current_num = m.group(1)
            blocks.setdefault(current_num, []).append(m.group(2).strip())
            continue

        # Continuation line for the currently open affiliation.
        if current_num and AFFILIATION_HINT_RE.search(line):
            # Avoid page numbers, arXiv lines and author lines.
            if not re.match(r"^(?:arXiv:|\d{4}\.\d+)", line, re.IGNORECASE):
                blocks[current_num].append(line)

    return {
        num: " ".join(parts).strip()
        for num, parts in blocks.items()
        if " ".join(parts).strip()
    }


def _find_affiliation_numbers_after_author(text: str, author: Dict[str, Any]) -> List[str]:
    """Find numeric affiliation markers immediately after an author name."""
    pattern = _author_regex(author["name"])
    matches = list(pattern.finditer(text))
    if not matches:
        return []

    # Prefer a match followed by digits, because that is the normal PDF form
    # (e.g. 'Silvia Muraro9').
    for match in matches:
        tail = text[match.end():match.end() + 30]
        m = re.match(r"\s*(?:\*+|†|‡|§)?\s*((?:\d{1,3}\s*[,;]\s*)*\d{1,3})", tail)
        if m:
            nums = re.findall(r"\d{1,3}", m.group(1))
            return nums
    return []


def _recover_author_emails(text: str, authors: List[Dict[str, Any]]) -> None:
    """Fill missing author emails using the local author block on first pages."""
    lines = [line.strip() for line in text.splitlines()]
    author_by_key = {author_name_key(a["name"]): a for a in authors}
    author_keys = set(author_by_key)

    # First, use line/block structure. This handles papers where every author
    # has an email directly below the affiliation.
    for i, line in enumerate(lines):
        key = normalize_for_match(line)
        author = author_by_key.get(key)
        if not author or author.get("email"):
            continue

        for nxt in lines[i + 1:i + 5]:
            if not nxt:
                continue
            if normalize_for_match(nxt) in author_keys:
                break
            m = EMAIL_RE.search(nxt)
            if m:
                author["email"] = m.group(0)
                break

    # Second, fall back to a narrow character window only when the author is
    # not represented as a standalone line.
    email_matches = list(EMAIL_RE.finditer(text))
    for author in authors:
        if author.get("email"):
            continue
        pattern = _author_regex(author["name"])
        name_match = pattern.search(text)
        if not name_match:
            continue
        window_start = max(0, name_match.start() - 30)
        window_end = min(len(text), name_match.end() + 180)
        nearby = [m for m in email_matches if window_start <= m.start() <= window_end]
        if len(nearby) == 1:
            author["email"] = nearby[0].group(0)


def _recover_unmapped_affiliations(text: str, authors: List[Dict[str, Any]]) -> int:
    """Recover numbered affiliations from first-page PDF text."""
    aff_map = _extract_affiliation_blocks(text)
    if not aff_map:
        return 0

    recovered = 0
    for author in authors:
        if author.get("affiliation"):
            continue
        nums = _find_affiliation_numbers_after_author(text, author)
        values = [aff_map[n] for n in nums if n in aff_map]
        if values:
            author["affiliation"] = " ; ".join(dict.fromkeys(clean_affiliation_text(v) or v for v in values))
            recovered += 1
    return recovered


def _recover_unmapped_non_numbered_affiliations(text: str, authors: List[Dict[str, Any]]) -> int:
    """Recover simple name -> organization blocks when no numeric superscripts exist."""
    lines = [line.strip() for line in text.splitlines()]
    if not lines:
        return 0

    author_keys = {author_name_key(a["name"]): a for a in authors}
    recovered = 0

    for i, line in enumerate(lines):
        key = normalize_for_match(line)
        author = author_keys.get(key)
        if not author or author.get("affiliation"):
            continue

        candidates: List[str] = []
        for nxt in lines[i + 1:i + 5]:
            if not nxt:
                continue
            if normalize_for_match(nxt) in author_keys:
                break
            if EMAIL_RE.fullmatch(nxt):
                continue
            if re.match(r"^(?:Abstract|Keywords|arXiv:)", nxt, re.IGNORECASE):
                break
            if AFFILIATION_HINT_RE.search(nxt):
                candidates.append(nxt)
            elif candidates:
                # Address continuation is allowed once an affiliation has begun.
                candidates.append(nxt)

        if candidates:
            author["affiliation"] = clean_affiliation_text(" ".join(candidates).strip())
            recovered += 1

    return recovered



def _clean_metadata_value(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.replace("\u00ad", "")
    value = re.sub(r"\s+", " ", value).strip(" \t\r\n.;,:")
    return value or None


def _first_match(patterns: List[re.Pattern], text: str) -> Optional[str]:
    for pattern in patterns:
        m = pattern.search(text)
        if m:
            return _clean_metadata_value(m.group(1) if m.lastindex else m.group(0))
    return None


def _extract_explicit_published_date(text: str) -> Optional[str]:
    """Return a date only when the paper explicitly labels it Published/Published on."""
    patterns = [
        re.compile(r"\bPublished(?:\s+on|\s+online|\s+date)?\s*[:：]?\s*([0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})", re.I),
        re.compile(r"\bPublished(?:\s+on|\s+online|\s+date)?\s*[:：]?\s*([A-Za-z]{3,9}\s+[0-9]{1,2},\s+[0-9]{4})", re.I),
        re.compile(r"\bPublished(?:\s+on|\s+online|\s+date)?\s*[:：]?\s*((?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2})", re.I),
        re.compile(r"\bPublished\s*:\s*([0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})", re.I),
    ]
    return _first_match(patterns, text)


def _split_keywords(value: str) -> List[str]:
    value = re.sub(r"\s+", " ", value).strip(" \t\r\n:;-–—")
    parts = re.split(r"\s*[;,]\s*|\s+•\s+", value)
    cleaned = []
    for part in parts:
        part = _clean_metadata_value(part)
        if part and len(part) <= 180:
            cleaned.append(part)
    return list(dict.fromkeys(cleaned))


def _extract_labeled_value(lines: List[str], labels: str) -> Optional[str]:
    pat = re.compile(rf"^\s*(?:{labels})\s*(?:[:：]|[—–-])?\s*(.*)$", re.I)
    for i, line in enumerate(lines):
        m = pat.match(line.strip())
        if not m:
            continue
        value = m.group(1).strip()
        if value:
            return _clean_metadata_value(value)
        for nxt in lines[i + 1:i + 3]:
            nxt = nxt.strip()
            if nxt and not re.match(r"^(?:abstract|keywords?|received|accepted|published|doi|references)\b", nxt, re.I):
                return _clean_metadata_value(nxt)
    return None


def _extract_explicit_arxiv_id(text: str) -> Optional[str]:
    """Return an arXiv identifier only when the PDF explicitly identifies it as arXiv."""
    if not text:
        return None
    explicit = re.search(
        r"\barXiv\s*:\s*(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)\b",
        text,
        re.I,
    )
    if explicit:
        return explicit.group(1)
    explicit_url = re.search(
        r"arxiv\.org/(?:abs|pdf)/((?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?)",
        text,
        re.I,
    )
    return explicit_url.group(1) if explicit_url else None


def _normalize_venue(value: Optional[str]) -> Optional[str]:
    """Normalize journal/proceedings names and reject obvious prose contamination."""
    if not value:
        return None
    v = _clean_metadata_value(value)
    if not v:
        return None

    # Known journal names/abbreviations appearing in the supplied papers.
    if re.search(r"\bJ\.?\s*Intell\.?\s*Syst\.?\b", v, re.I):
        return "Journal of Intelligent Systems"
    if re.search(r"\bIntelligent Decision Technologies\b", v, re.I):
        return "Intelligent Decision Technologies"
    if re.search(r"\bAdvances in Artificial Intelligence\b", v, re.I):
        return "Advances in Artificial Intelligence"
    if re.search(r"\bPLOS\s+ONE\b", v, re.I):
        return "PLOS ONE"

    # Explicit journal label: keep the journal only, not the publisher.
    m = re.match(r"^Journal(?:\s+Name)?\s*[:：-]\s*(.+)$", v, re.I)
    if m:
        v = m.group(1).strip()
        v = re.split(r"\s*[,;]\s*(?:IOS\s+Press|IEEE|ACM|Springer(?:\s+Nature)?|Elsevier|Hindawi(?:\s+Publishing\s+Corporation)?|De\s+Gruyter|PLOS\s+ONE)\s*$", v, flags=re.I)[0]
        return v.strip() or None

    # Proceedings/conference citation: remove publication wrapper and trailing date/location.
    v = re.sub(r"^Published\s+in\s+", "", v, flags=re.I).strip()
    if re.search(r"\bProceedings\s+of\b", v, re.I):
        v = re.split(r"\s*,\s*(?:\d{1,2}\s*[-–]\s*\d{1,2}\s+[A-Za-z]+\s*,?\s*)?(?:19|20)\d{2}\b", v, maxsplit=1, flags=re.I)[0]
        v = re.split(r"\s*,\s*(?:19|20)\d{2}\b", v, maxsplit=1)[0]
        return v.strip(" .,") or None

    # Reject ordinary prose that happened to contain a venue keyword.
    if re.search(r"^(?:This section|This paper|The section|We |Our |It |The proposed|The present)\b", v, re.I):
        return None
    if len(v.split()) > 25:
        return None
    return v.strip(" .,") or None


def _repair_title(title: Optional[str]) -> Optional[str]:
    if not title:
        return None
    t = _clean_metadata_value(title)
    # Copyright/venue text can get appended to a GROBID title on some IEEE PDFs.
    t = re.split(r"\s+Published\s+in\s+Proceedings\b", t, maxsplit=1, flags=re.I)[0].strip(" .:-")
    return t or None


def _extract_structural_metadata(text: str, metadata: Dict[str, Any]) -> List[str]:
    """Deterministic first-page rules for the publisher layouts in the supplied papers."""
    actions: List[str] = []
    lines = [re.sub(r"\s+", " ", x).strip() for x in text.splitlines()]
    lines = [x for x in lines if x]
    joined = "\n".join(lines)

    repaired_title = _repair_title(metadata.get("title"))
    if repaired_title and repaired_title != metadata.get("title"):
        metadata["title"] = repaired_title
        actions.append("Cleaned title contamination from publication header")

    if not metadata.get("article_type"):
        for label, pat in ARTICLE_TYPE_PATTERNS:
            if any(pat.fullmatch(line) for line in lines):
                metadata["article_type"] = label
                actions.append(f"Recovered article type: {label}")
                break

    if not metadata.get("keywords"):
        kw_labels = r"Keywords?|Key words?|Index Terms?"
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Accept normal labelled forms (Keywords:, Keywords—, Key words:)
            # and an exact standalone KEYWORDS heading. Do NOT match prose such
            # as "keywords, regular expressions..." inside the abstract/body.
            m = re.match(rf"^\s*(?:{kw_labels})\s*(?:[:：]|[—–-])\s*(.*)$", stripped, re.I)
            standalone = re.fullmatch(r"KEYWORDS", stripped, re.I)
            if not m and not standalone:
                continue

            value = m.group(1).strip() if m else ""
            if not value and i + 1 < len(lines):
                value = lines[i + 1].strip()
                start = i + 2
            else:
                start = i + 1

            # ACM-style keyword blocks can wrap, but stop before citation
            # boilerplate such as ACM Reference Format.
            extra = []
            for nxt in lines[start:start + 4]:
                if re.match(r"^(?:ACM Reference Format|Abstract|Introduction|Received|Accepted|Published|DOI|References|I\.?\s+INTRODUCTION|1\.?\s+Introduction)\b", nxt, re.I):
                    break
                if len(nxt) > 250:
                    break
                extra.append(nxt)
            if extra:
                value = " ".join([value] + extra)

            kws = _split_keywords(value)
            if kws:
                metadata["keywords"] = kws
                actions.append("Recovered keywords/index terms from first pages")
                break

    if not metadata.get("doi"):
        m = DOI_PATTERN.search(joined)
        if m:
            metadata["doi"] = m.group(0).rstrip(".,;)")
            actions.append("Recovered DOI from first pages")
    explicit_arxiv = _extract_explicit_arxiv_id(joined)
    if explicit_arxiv:
        if metadata.get("arxiv_id") != explicit_arxiv:
            metadata["arxiv_id"] = explicit_arxiv
            actions.append("Recovered arXiv ID from explicit arXiv marker")
    elif metadata.get("arxiv_id"):
        # Never accept a DOI suffix or another numeric token as an arXiv ID.
        metadata["arxiv_id"] = None
    if not metadata.get("issn"):
        m = ISSN_PATTERN.search(joined)
        if m:
            metadata["issn"] = m.group(1).replace("–", "-")
            actions.append("Recovered ISSN from first pages")
    if not metadata.get("eissn"):
        m = EISSN_PATTERN.search(joined)
        if m:
            metadata["eissn"] = m.group(1).replace("–", "-")
            actions.append("Recovered eISSN from first pages")
    if not metadata.get("isbn"):
        m = ISBN_PATTERN.search(joined)
        if m:
            metadata["isbn"] = re.sub(r"\s+", "-", m.group(1)).strip("-")
            actions.append("Recovered ISBN from first pages")


    # Prefer an explicitly printed Published date over a generic GROBID date
    # (e.g. Rice: Published on 29 August 2017 vs. earlier review metadata).
    explicit_published = _extract_explicit_published_date(joined)
    if explicit_published and metadata.get("publication_date") != explicit_published:
        metadata["publication_date"] = explicit_published
        actions.append("Corrected publication date from explicit Published label")

    if not metadata.get("publication_date"):
        published_patterns = [
            re.compile(r"\bPublished(?:\s+on|\s+online|\s+date)?\s*[:：]?\s*([0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})", re.I),
            re.compile(r"\bPublished(?:\s+on|\s+online|\s+date)?\s*[:：]?\s*([A-Za-z]{3,9}\s+[0-9]{1,2},\s+[0-9]{4})", re.I),
            re.compile(r"\bPublished(?:\s+on|\s+online|\s+date)?\s*[:：]?\s*((?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2})", re.I),
        ]
        published = _first_match(published_patterns, joined)
        if published:
            metadata["publication_date"] = published
            actions.append("Recovered published date from labeled metadata")

    if not metadata.get("corresponding_author"):
        corr = re.search(r"(?:Correspondence\s+(?:should\s+be\s+addressed\s+to|to)|Corresponding\s+author(?:\s+is)?)\s*([^;\n]+)", joined, re.I)
        if corr:
            value = _clean_metadata_value(corr.group(1))
            if value:
                metadata["corresponding_author"] = value
                actions.append("Recovered corresponding author")

    # Always validate/repair the venue using the first-page text. This can
    # override a bad GROBID venue such as an abstract sentence.
    venue_candidate = None
    labeled = _extract_labeled_value(lines, r"Journal(?:\s+Name)?|Venue")
    if labeled:
        venue_candidate = labeled

    # Journal citation formats, e.g. "J. Intell. Syst. 2017; 26(3): 457–469".
    if not venue_candidate:
        for line in lines[:40]:
            if re.search(r"\bJ\.?\s*Intell\.?\s*Syst\.?\s+(?:19|20)\d{2}\b", line, re.I):
                venue_candidate = "Journal of Intelligent Systems"
                break

    # Known journal names in isolation.
    if not venue_candidate:
        for line in lines[:80]:
            if re.fullmatch(r"(?:Advances in Artificial Intelligence|Intelligent Decision Technologies|PLOS ONE)", line.strip(), re.I):
                venue_candidate = line.strip()
                break

    # IEEE/ACM proceedings printed before the title.
    if not venue_candidate:
        for line in lines[:35]:
            if re.search(r"\bProceedings\s+of\b", line, re.I):
                venue_candidate = line
                break

    normalized_candidate = _normalize_venue(venue_candidate) if venue_candidate else None
    if normalized_candidate:
        if metadata.get("journal_or_venue") != normalized_candidate:
            metadata["journal_or_venue"] = normalized_candidate
            actions.append("Normalized journal/venue from first-page structure")
    else:
        normalized_existing = _normalize_venue(metadata.get("journal_or_venue"))
        if normalized_existing:
            metadata["journal_or_venue"] = normalized_existing

    # Conference/event is separate from the journal/venue column.
    if not metadata.get("conference_event"):
        for line in lines[:35]:
            if re.search(r"\bProceedings\s+of\b", line, re.I):
                event = _normalize_venue(line)
                if event:
                    metadata["conference_event"] = event
                    actions.append("Recovered conference/event")
                    break

    if not metadata.get("publisher"):
        publisher_patterns = [
            r"(?:Published\s+by|Publisher)\s*[:：]?\s*([^\n]+)",
            r"\b(Hindawi\s+Publishing\s+Corporation)\b",
            r"\b(Association\s+for\s+Computational\s+Linguistics)\b",
            r"\b(IEEE)\b", r"\b(ACM)\b", r"\b(Springer(?:\s+Nature)?)\b",
            r"\b(Elsevier)\b", r"\b(De\s+Gruyter)\b", r"\b(PLOS\s+ONE)\b",
        ]
        for pat in publisher_patterns:
            m = re.search(pat, joined, re.I)
            if m:
                metadata["publisher"] = _clean_metadata_value(m.group(1) if m.lastindex else m.group(0))
                actions.append("Recovered publisher")
                break

    if not metadata.get("conference_event"):
        event_patterns = [
            r"((?:Published\s+in\s+)?Proceedings\s+of\s+[^\n]+)",
            r"((?:The\s+)?\d+(?:st|nd|rd|th)\s+[^\n]*?(?:Conference|Annual\s+Meeting)[^\n]*)",
            r"((?:SIGIR\s*['’]\d{2}):\s*Proceedings\s+of\s+[^\n]+)",
            r"(ICACEA[^\n]*)",
        ]
        for pat in event_patterns:
            m = re.search(pat, joined, re.I)
            if m:
                value = _clean_metadata_value(m.group(1))
                if value and len(value) <= 300:
                    metadata["conference_event"] = value
                    actions.append("Recovered conference/event")
                    break

    if metadata.get("conference_event") and metadata.get("journal_or_venue"):
        if re.match(r"^\([^)]{2,80}\)", str(metadata["journal_or_venue"])):
            metadata["journal_or_venue"] = metadata["conference_event"]
            actions.append("Replaced parenthetical venue fragment with conference/event title")

    if not metadata.get("volume") or not metadata.get("issue") or not metadata.get("pages"):
        structural_patterns = [
            # Hindawi: Volume 2014, Article ID 717803, 11 pages
            re.compile(r"\bVolume\s+(\d{4})\s*,\s*Article\s+(?:ID|No\.?|Number)\s+[A-Z0-9._-]+\s*,\s*([0-9]+)\s+pages?", re.I),
            # ACL: Volume 1: Long Papers, pages 14229–14253
            re.compile(r"\bVolume\s+(\d+)\s*:\s*[^\n]*?\bpages?\s+([0-9]+\s*[–-]\s*[0-9]+)", re.I),
            # Rice: 2017, Volume 11, No 3, pp 357–373
            re.compile(r"\b(?:19|20)\d{2}\s*[,;]?\s*Volume\s+(\d+)\s*,\s*(?:No\.?|Number)\s+(\d+)\s*,\s*pp?\.?\s*([0-9]+\s*[–-]\s*[0-9]+)", re.I),
            # Generic Volume 11, No 3, pp 357–373
            re.compile(r"\bVolume\s+(\d+)\s*[,;]?\s*(?:No\.?|Number|Issue)\s+(\d+)\s*[,;]?\s*(?:pp?\.?|pages?)\s*([0-9]+\s*[–-]\s*[0-9]+)", re.I),
            # JISYS / PLOS ONE: 26(3): 457–469 or 18(6): e0287611
            re.compile(r"\b(\d+)\s*\((\d+)\)\s*[:;,]\s*([A-Za-z]?\d+[A-Za-z0-9]*\s*(?:[–-]\s*[A-Za-z]?\d+[A-Za-z0-9]*)?)", re.I),
        ]
        for pat in structural_patterns:
            m = pat.search(joined)
            if m:
                if m.lastindex == 2:
                    if not metadata.get("volume"): metadata["volume"] = m.group(1)
                    if not metadata.get("pages"): metadata["pages"] = m.group(2)
                else:
                    if not metadata.get("volume"): metadata["volume"] = m.group(1)
                    if not metadata.get("issue"): metadata["issue"] = m.group(2)
                    if not metadata.get("pages"): metadata["pages"] = m.group(3).replace("–", "-")
                actions.append("Recovered volume/issue/pages from structural citation format")
                break

    if not metadata.get("volume"):
        metadata["volume"] = _extract_labeled_value(lines, r"Volume|Vol\.")
        if metadata.get("volume"): actions.append("Recovered volume from label")
    if not metadata.get("issue"):
        m = re.search(r"\bIssue\s*[:：-]\s*(\d+)\b", joined, re.I)
        if m:
            metadata["issue"] = m.group(1)
            actions.append("Recovered issue from label")
    if not metadata.get("pages"):
        metadata["pages"] = _extract_labeled_value(lines, r"Pages|pp\.")
        if metadata.get("pages"): actions.append("Recovered pages from label")

    return actions


def recover_from_first_pages(pdf_path: str, metadata: Dict[str, Any], debug: bool = False) -> Tuple[Dict[str, Any], List[str]]:
    """Use only the first few PDF pages to repair metadata without fulltext extraction."""
    text = extract_first_pages_text(pdf_path)
    if not text:
        return metadata, []

    actions: List[str] = []
    authors = metadata.get("authors", [])

    # Emails are often visible on page 1 even when GROBID misses some.
    before = sum(1 for a in authors if a.get("email"))
    _recover_author_emails(text, authors)
    after = sum(1 for a in authors if a.get("email"))
    if after > before:
        actions.append(f"Recovered {after - before} author email(s) from first pages")

    # Numbered affiliation systems are common in multi-author papers.
    before_aff = sum(1 for a in authors if a.get("affiliation"))
    recovered_num = _recover_unmapped_affiliations(text, authors)
    recovered_plain = _recover_unmapped_non_numbered_affiliations(text, authors)
    after_aff = sum(1 for a in authors if a.get("affiliation"))
    if after_aff > before_aff:
        actions.append(
            f"Recovered {after_aff - before_aff} author affiliation(s) from first pages"
        )

    # Recover arXiv ID only from an explicit arXiv marker/URL.
    explicit_arxiv = _extract_explicit_arxiv_id(text)
    if explicit_arxiv:
        metadata["arxiv_id"] = explicit_arxiv
        actions.append("Recovered arXiv ID from explicit arXiv marker")
    elif metadata.get("arxiv_id"):
        metadata["arxiv_id"] = None


    # Recover a venue/conference line only when it is explicitly visible.
    if not metadata.get("journal_or_venue"):
        lines = [line.strip() for line in text.splitlines()]
        for i, line in enumerate(lines):
            if not line or len(line) > 220:
                continue
                
            # A. Check for structural clue (e.g., "Volume 2014, Article ID 717803")
            if VOL_ARTICLE_RE.search(line):
                if i > 0:
                    metadata["journal_or_venue"] = lines[i-1]
                    actions.append("Recovered venue from first pages via volume/article structure")
                    break

            # B. Fallback to keyword matching
            if VENUE_HINT_RE.search(line) and not line.lower().startswith("arxiv"):
                # Avoid treating a generic prose sentence as a venue.
                if re.search(r"\b(?:conference|symposium|workshop|proceedings|journal|transactions|advances in|frontiers in|annals of|review of|plos one|sigir)\b", line, re.IGNORECASE):
                    metadata["journal_or_venue"] = line
                    actions.append("Recovered venue/conference from first pages")
                    break

    # Recover a visible date only when header date is absent. Prefer Published over Received/Accepted.
    if not metadata.get("publication_date"):
        published_patterns = [
            re.compile(r"\bPublished(?:\s+on|\s+online|\s+date)?\s*[:：]?\s*([0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{4})", re.I),
            re.compile(r"\bPublished(?:\s+on|\s+online|\s+date)?\s*[:：]?\s*([A-Za-z]{3,9}\s+[0-9]{1,2},\s+[0-9]{4})", re.I),
            re.compile(r"\bPublished(?:\s+on|\s+online|\s+date)?\s*[:：]?\s*((?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2})", re.I),
        ]
        published = _first_match(published_patterns, text)
        if published:
            metadata["publication_date"] = published
            actions.append("Recovered published date from labeled metadata")
        else:
            date_patterns = [
                re.compile(r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b", re.I),
                re.compile(r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4})\b", re.I),
            ]
            for pat in date_patterns:
                m = pat.search(text)
                if m:
                    metadata["publication_date"] = m.group(1)
                    actions.append("Recovered publication date from first pages")
                    break

    # Deterministic keyword/label/structural rescue tuned to the supplied paper layouts.
    actions.extend(_extract_structural_metadata(text, metadata))

    if debug and actions:
        print(f"First-page metadata rescue: {actions}", file=sys.stderr)

    return metadata, actions


# ============================================================================
# Selective fulltext decision
# ============================================================================

def metadata_quality(metadata: Dict[str, Any]) -> Dict[str, Any]:
    authors = metadata.get("authors", [])
    author_count = len(authors)
    with_aff = sum(bool(a.get("affiliation")) for a in authors)
    with_email = sum(bool(a.get("email")) for a in authors)

    return {
        "author_count": author_count,
        "authors_with_affiliation": with_aff,
        "affiliation_ratio": (with_aff / author_count) if author_count else 0.0,
        "authors_with_email": with_email,
        "has_title": bool(metadata.get("title")),
        "has_abstract": bool(metadata.get("abstract")),
        "has_identifier": bool(metadata.get("doi") or metadata.get("arxiv_id")),
        "has_venue": bool(metadata.get("journal_or_venue")),
    }


def should_use_fulltext(metadata: Dict[str, Any], debug: bool = False) -> Tuple[bool, List[str]]:
    """Decide whether /fulltext is justified for metadata extraction.

    Important: low quality by itself does NOT trigger fulltext. We trigger only
    for concrete metadata deficiencies that fulltext has a reasonable chance to
    repair. Keywords, DOI, venue, and missing optional emails do not automatically
    trigger fulltext because they may legitimately be absent.
    """
    q = metadata_quality(metadata)
    reasons: List[str] = []

    if not q["has_title"]:
        reasons.append("Missing title")

    if q["author_count"] == 0:
        reasons.append("No authors found")

    if not q["has_abstract"]:
        reasons.append("No abstract found")

    # Affiliation completeness matters when there are multiple authors. We only
    # request fulltext when the header has very little affiliation information.
    # First-page PDF rescue happens before this decision.
    if q["author_count"] >= 2 and q["affiliation_ratio"] < 0.50:
        reasons.append(
            f"Only {q['authors_with_affiliation']}/{q['author_count']} authors have affiliations"
        )

    lang = metadata.get("language", "unknown")
    if lang == "unknown":
        reasons.append("Language not detected")

    # Do not trigger fulltext simply because DOI/venue/keywords are missing:
    # those fields may genuinely be absent, and the first pages were already checked.

    use = bool(reasons)
    if debug and reasons:
        print(f"Fulltext decision: {reasons}", file=sys.stderr)
    return use, reasons


def calculate_quality_score(metadata: Dict[str, Any]) -> int:
    """Score metadata completeness independently of the fulltext decision."""
    q = metadata_quality(metadata)
    score = 0

    # Title 10
    if q["has_title"]:
        score += 10

    # Authors 25
    if q["author_count"] >= 3:
        score += 25
    elif q["author_count"] == 2:
        score += 20
    elif q["author_count"] == 1:
        score += 12

    # Abstract 20
    abstract_words = len((metadata.get("abstract") or "").split())
    if abstract_words >= 100:
        score += 20
    elif abstract_words >= 50:
        score += 16
    elif abstract_words >= 20:
        score += 10
    elif abstract_words > 0:
        score += 5

    # Affiliations 15
    score += round(q["affiliation_ratio"] * 15)

    # Identifier 10
    if q["has_identifier"]:
        score += 10

    # Date 5
    if metadata.get("publication_date"):
        score += 5

    # Venue 5, but absence is not necessarily an extraction failure.
    if q["has_venue"]:
        score += 5

    # Keywords 5
    if metadata.get("keywords"):
        score += 5

    # Language 5
    if metadata.get("language") and metadata.get("language") != "unknown":
        score += 5

    return max(0, min(100, score))


# ============================================================================
# GROBID integration
# ============================================================================

def call_grobid(
    session: requests.Session,
    pdf_path: str,
    url: str,
    timeout: int = 60,
    form_data: Optional[Dict[str, str]] = None,
) -> bytes:
    with open(pdf_path, "rb") as f:
        files = {"input": (os.path.basename(pdf_path), f, "application/pdf")}
        response = session.post(
            url,
            files=files,
            data=form_data or {},
            timeout=timeout,
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"GROBID returned {response.status_code}: {response.text[:300]}"
        )
    return response.content


def merge_metadata(base: Dict[str, Any], extra: Dict[str, Any]) -> None:
    """Merge metadata without throwing away richer header information."""
    scalar_keys = [
        "title", "abstract", "publication_date", "doi", "journal_or_venue",
        "arxiv_id", "article_type", "publisher", "conference_event",
        "volume", "issue", "pages", "issn", "eissn", "isbn",
        "corresponding_author", "language",
    ]
    for key in scalar_keys:
        if extra.get(key):
            # Prefer non-empty existing values except for abstract: fulltext can
            # repair a missing/short header abstract.
            if not base.get(key) or (key == "abstract" and len(extra[key].split()) > len((base.get(key) or "").split())):
                base[key] = extra[key]

    if extra.get("keywords") and not base.get("keywords"):
        base["keywords"] = extra["keywords"]

    # Merge authors by normalized name. Never create Name (2).
    extra_by_key = {author_name_key(a["name"]): a for a in extra.get("authors", [])}
    base_by_key = {author_name_key(a["name"]): a for a in base.get("authors", [])}

    for key, extra_author in extra_by_key.items():
        if key not in base_by_key:
            base["authors"].append(extra_author)
            base_by_key[key] = extra_author
            continue

        current = base_by_key[key]
        for field in ("affiliation", "email"):
            value = extra_author.get(field)
            if value and not current.get(field):
                current[field] = value

    # Keep deterministic author order from the header where possible.


def extract_metadata_smart(
    pdf_path: str,
    session: Optional[requests.Session] = None,
    timeout: int = 60,
    debug: bool = False,
    force_fulltext: bool = False,
) -> Dict[str, Any]:
    """Header-first metadata extraction with selective fulltext fallback."""
    own_session = session is None
    session = session or make_session()

    result: Dict[str, Any] = {
        "title": None,
        "authors": [],
        "abstract": None,
        "publication_date": None,
        "doi": None,
        "journal_or_venue": None,
        "arxiv_id": None,
        "keywords": [],
        "article_type": None,
        "publisher": None,
        "conference_event": None,
        "volume": None,
        "issue": None,
        "pages": None,
        "issn": None,
        "eissn": None,
        "isbn": None,
        "corresponding_author": None,
        "language": None,
        "source_file": os.path.basename(pdf_path),
        "_endpoints_used": [],
        "_used_fulltext": False,
        "_reasons_for_fulltext": [],
        "_first_page_repairs": [],
        "_filtered_organizations": 0,
        "_quality_score": 0,
    }

    try:
        # --------------------------------------------------------------
        # 1. Header first (default fast path)
        # --------------------------------------------------------------
        if not force_fulltext:
            try:
                header_xml = call_grobid(session, pdf_path, GROBID_HEADER_URL, timeout)
                result["_endpoints_used"].append("header")

                if debug:
                    stem = os.path.splitext(os.path.basename(pdf_path))[0]
                    path = f"debug_header__{stem}.xml"
                    with open(path, "wb") as f:
                        f.write(header_xml)
                    print(f"Saved header TEI to {path}", file=sys.stderr)

                header_root = parse_tei_common(header_xml)
                result.update(extract_tei_metadata(header_root))
                result["_filtered_organizations"] = count_filtered_organizations(header_root)

                # ------------------------------------------------------
                # 2. Inspect only first pages before deciding on fulltext
                # ------------------------------------------------------
                result, repairs = recover_from_first_pages(pdf_path, result, debug)
                result["_first_page_repairs"].extend(repairs)

                use_fulltext, reasons = should_use_fulltext(result, debug)
                result["_reasons_for_fulltext"] = reasons

                if not use_fulltext:
                    result["_quality_score"] = calculate_quality_score(result)
                    return result

            except Exception as exc:
                result["_reasons_for_fulltext"] = [f"Header failed: {exc}"]
                if debug:
                    print(f"Header failed; attempting fulltext: {exc}", file=sys.stderr)

        else:
            result["_reasons_for_fulltext"] = ["Forced by --force-fulltext"]

        # --------------------------------------------------------------
        # 3. Fulltext only when justified
        # --------------------------------------------------------------
        fulltext_xml = call_grobid(
            session,
            pdf_path,
            GROBID_FULLTEXT_URL,
            timeout,
        )
        result["_endpoints_used"].append("fulltext")
        result["_used_fulltext"] = True

        if debug:
            stem = os.path.splitext(os.path.basename(pdf_path))[0]
            path = f"debug_fulltext__{stem}.xml"
            with open(path, "wb") as f:
                f.write(fulltext_xml)
            print(f"Saved fulltext TEI to {path}", file=sys.stderr)

        fulltext_root = parse_tei_common(fulltext_xml)
        fulltext_data = extract_tei_metadata(fulltext_root)
        merge_metadata(result, fulltext_data)

        # The first-page rescue is still useful after fulltext because some
        # GROBID fulltext author blocks can lack affiliations/emails.
        result, repairs = recover_from_first_pages(pdf_path, result, debug)
        result["_first_page_repairs"].extend(repairs)

        result["_filtered_organizations"] = max(
            result.get("_filtered_organizations", 0),
            count_filtered_organizations(fulltext_root),
        )

        if not result.get("arxiv_id"):
            result["arxiv_id"] = _extract_arxiv_from_filename(os.path.basename(pdf_path))

        result["quality_score"] = calculate_quality_score(result)
        result["_quality_score"] = result.pop("quality_score")
        return result

    finally:
        if own_session:
            session.close()


def _extract_arxiv_from_filename(filename: str) -> Optional[str]:
    match = ARXIV_ID_PATTERN.search(filename)
    return match.group(1) if match else None


# ============================================================================
# Processing / batch
# ============================================================================

def process_one_paper(
    pdf_path: str,
    session: Optional[requests.Session] = None,
    timeout: int = 60,
    debug: bool = False,
    force_fulltext: bool = False,
) -> Dict[str, Any]:
    try:
        metadata = extract_metadata_smart(
            pdf_path=pdf_path,
            session=session,
            timeout=timeout,
            debug=debug,
            force_fulltext=force_fulltext,
        )
        return {
            "ok": True,
            "source_file": os.path.basename(pdf_path),
            "metadata": metadata,
            "pdf_path": pdf_path,
        }
    except Exception as exc:
        return {
            "ok": False,
            "source_file": os.path.basename(pdf_path),
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "pdf_path": pdf_path,
        }


def process_batch(
    pdf_paths: List[str],
    timeout: int = 60,
    debug: bool = False,
    workers: int = 1,
    force_fulltext: bool = False,
):
    if workers <= 1:
        session = make_session()
        try:
            for pdf_path in pdf_paths:
                yield process_one_paper(
                    pdf_path, session, timeout, debug, force_fulltext
                )
        finally:
            session.close()
        return

    def worker(pdf_path: str):
        thread_session = make_session()
        try:
            return process_one_paper(
                pdf_path, thread_session, timeout, debug, force_fulltext
            )
        finally:
            thread_session.close()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, p): p for p in pdf_paths}
        for future in as_completed(futures):
            yield future.result()


def resolve_pdf_paths(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        return sorted(glob.glob(os.path.join(input_path, "**", "*.pdf"), recursive=True))
    if any(ch in input_path for ch in "*?[]"):
        return sorted(glob.glob(input_path))
    if os.path.isfile(input_path):
        return [input_path]
    return []


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Header-first scholarly metadata extraction with selective GROBID fulltext fallback."
    )
    parser.add_argument("pdf_path", nargs="?", help="Single PDF")
    parser.add_argument("--input", help="A PDF, directory, or glob pattern")
    parser.add_argument("--output", help="JSON Lines output path")
    parser.add_argument("--debug", action="store_true", help="Save raw GROBID XML")
    parser.add_argument("--timeout", type=int, default=60, help="GROBID timeout in seconds")
    parser.add_argument("--workers", type=int, default=1, help="Batch concurrency")
    parser.add_argument(
        "--force-fulltext",
        action="store_true",
        help="Skip the normal decision and call /processFulltextDocument",
    )
    parser.add_argument("--summary", action="store_true", help="Show processing summary")
    args = parser.parse_args()

    target = args.input or args.pdf_path
    if not target:
        print("Error: provide a PDF path or --input <file|dir|glob>", file=sys.stderr)
        sys.exit(1)

    pdf_paths = resolve_pdf_paths(target)
    if not pdf_paths:
        print(f"Error: no PDF files found for: {target}", file=sys.stderr)
        sys.exit(1)

    if len(pdf_paths) == 1 and not args.output:
        result = process_one_paper(
            pdf_paths[0],
            timeout=args.timeout,
            debug=args.debug,
            force_fulltext=args.force_fulltext,
        )
        if not result["ok"]:
            print(f"Error: {result['error']}", file=sys.stderr)
            sys.exit(1)

        print(json.dumps(result["metadata"], indent=2, ensure_ascii=False))

        if args.summary:
            m = result["metadata"]
            print("\n" + "=" * 60, file=sys.stderr)
            print("SUMMARY", file=sys.stderr)
            print("=" * 60, file=sys.stderr)
            print(f"Quality score: {m.get('_quality_score', 0)}/100", file=sys.stderr)
            print(f"Used fulltext: {m.get('_used_fulltext', False)}", file=sys.stderr)
            print(f"Endpoints used: {m.get('_endpoints_used', [])}", file=sys.stderr)
            print(f"Organizations filtered: {m.get('_filtered_organizations', 0)}", file=sys.stderr)
            if m.get("_first_page_repairs"):
                print("First-page repairs:", file=sys.stderr)
                for item in m["_first_page_repairs"]:
                    print(f"  - {item}", file=sys.stderr)
            if m.get("_reasons_for_fulltext"):
                print("Fulltext decision/reason:", file=sys.stderr)
                for item in m["_reasons_for_fulltext"]:
                    print(f"  - {item}", file=sys.stderr)
            print("=" * 60, file=sys.stderr)
        return

    out_fh = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    n_ok = n_fail = n_fulltext = 0
    total_quality = 0
    total_filtered = 0
    t0 = time.time()

    try:
        for result in process_batch(
            pdf_paths,
            timeout=args.timeout,
            debug=args.debug,
            workers=args.workers,
            force_fulltext=args.force_fulltext,
        ):
            out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_fh.flush()

            if result["ok"]:
                n_ok += 1
                m = result["metadata"]
                if m.get("_used_fulltext"):
                    n_fulltext += 1
                total_quality += m.get("_quality_score", 0)
                total_filtered += m.get("_filtered_organizations", 0)
            else:
                n_fail += 1
                print(
                    f"FAILED: {result['source_file']}: {result['error']}",
                    file=sys.stderr,
                )
    finally:
        if args.output:
            out_fh.close()

    elapsed = time.time() - t0

    if args.summary:
        print("\n" + "=" * 60, file=sys.stderr)
        print("SUMMARY", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"Total papers: {len(pdf_paths)}", file=sys.stderr)
        print(f"Successful: {n_ok}", file=sys.stderr)
        print(f"Failed: {n_fail}", file=sys.stderr)
        if n_ok:
            print(f"Used fulltext: {n_fulltext} ({n_fulltext / n_ok * 100:.1f}%)", file=sys.stderr)
            print(f"Average quality: {total_quality / n_ok:.1f}/100", file=sys.stderr)
            print(f"Organizations filtered: {total_filtered}", file=sys.stderr)
        print(f"Time: {elapsed:.1f}s", file=sys.stderr)
        print("=" * 60, file=sys.stderr)

    print(f"Done: {n_ok} ok, {n_fail} failed, {elapsed:.1f}s total", file=sys.stderr)


if __name__ == "__main__":
    main()
