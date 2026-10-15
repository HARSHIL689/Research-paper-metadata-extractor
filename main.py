import os
import re
import shutil
from urllib.parse import quote
from xml.sax.saxutils import unescape

import requests
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.core.extractor import process_one_paper
from app.core.excel_manager import (
    append_metadata_to_excel,
    EXCEL_FILE_PATH,
)

app = FastAPI(title="ScholarMeta Extractor")

UPLOAD_DIR = os.path.join("data", "uploads")
API_EXPORT_DIR = os.path.join("data", "api_exports")
API_EXCEL_FILE_PATH = os.path.join(API_EXPORT_DIR, "papers_metadata_api.xlsx")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(API_EXPORT_DIR, exist_ok=True)

templates = Jinja2Templates(directory=os.path.join("app", "templates"))

CROSSREF_API_URL = "https://api.crossref.org/works/{}"


def clean_doi(doi: str) -> str:
    """Normalize DOI input supplied by the user."""
    doi = (doi or "").strip()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.I)
    doi = re.sub(r"^doi\s*:\s*", "", doi, flags=re.I)
    return doi.strip().rstrip(".,;)")


def crossref_date(message: dict):
    """Use the most specific Crossref publication date available."""
    for key in ("published-print", "published-online", "published", "issued"):
        value = message.get(key) or {}
        parts = value.get("date-parts") or []
        if not parts or not parts[0]:
            continue
        p = parts[0]
        try:
            if len(p) >= 3:
                return f"{p[0]:04d}-{p[1]:02d}-{p[2]:02d}"
            if len(p) == 2:
                return f"{p[0]:04d}-{p[1]:02d}"
            return f"{p[0]:04d}"
        except (TypeError, ValueError):
            continue
    return None


def clean_crossref_abstract(value):
    if not value:
        return None
    # Crossref may return JATS/XML markup in the abstract.
    text = re.sub(r"<[^>]+>", " ", str(value))
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def crossref_to_metadata(doi: str, message: dict) -> dict:
    authors = []
    seen = set()
    for author in message.get("author") or []:
        given = " ".join(str(author.get("given") or "").split())
        family = " ".join(str(author.get("family") or "").split())
        name = " ".join(x for x in (given, family) if x).strip()
        if not name:
            name = " ".join(str(author.get("name") or "").split()).strip()
        key = name.casefold()
        if name and key not in seen:
            seen.add(key)
            authors.append({"name": name, "affiliation": None, "email": None})

    container = message.get("container-title") or []
    venue = container[0] if container else None

    subjects = []
    for subject in message.get("subject") or []:
        value = " ".join(str(subject).split()).strip()
        if value and value not in subjects:
            subjects.append(value)

    language = message.get("language") or "N/A"
    title_list = message.get("title") or []
    title = " ".join(str(title_list[0]).split()) if title_list else None

    return {
        "source_file": f"Crossref API - DOI {doi}",
        "title": title,
        "authors": authors,
        "publication_date": crossref_date(message),
        "journal_or_venue": venue,
        "volume": message.get("volume"),
        "doi": message.get("DOI") or doi,
        "arxiv_id": None,
        "keywords": subjects,
        "language": language,
        "_quality_score": "N/A",
        "abstract": clean_crossref_abstract(message.get("abstract")),
    }


@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    """Serve the dashboard and start a fresh extracted-data workbook on refresh."""
    # A refresh starts a fresh session for both extraction methods.
    for path in (EXCEL_FILE_PATH, API_EXCEL_FILE_PATH):
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


@app.post("/extract")
async def extract_and_save(file: UploadFile = File(...)):
    """Extract metadata from a PDF and append it to the extracted-data workbook."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    safe_filename = os.path.basename(file.filename)
    saved_path = os.path.join(UPLOAD_DIR, safe_filename)

    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = process_one_paper(saved_path, debug=False)
        if not result.get("ok"):
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": result.get("error")},
            )

        metadata = result["metadata"]
        append_metadata_to_excel(metadata)

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "filename": safe_filename,
                "metadata": metadata,
            },
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})
    finally:
        if os.path.exists(saved_path):
            try:
                os.remove(saved_path)
            except OSError:
                pass


@app.get("/download-excel")
async def download_excel():
    """Download the Excel containing data extracted from uploaded papers."""
    if not os.path.exists(EXCEL_FILE_PATH):
        raise HTTPException(status_code=404, detail="No extracted paper data exists yet.")
    return FileResponse(
        path=EXCEL_FILE_PATH,
        filename="papers_metadata_extracted.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.post("/fetch-api-data")
async def fetch_api_data(doi: str):
    """Fetch DOI metadata from Crossref and append it to the API workbook."""
    doi = clean_doi(doi)
    if not doi:
        raise HTTPException(status_code=400, detail="Please enter a DOI.")

    url = CROSSREF_API_URL.format(quote(doi, safe=""))
    headers = {
        "Accept": "application/json",
        "User-Agent": "ScholarMeta/1.0 DOI metadata exporter",
    }

    try:
        response = requests.get(url, headers=headers, timeout=20)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Could not reach Crossref API: {exc}") from exc

    if response.status_code == 404:
        raise HTTPException(status_code=404, detail=f"No Crossref record was found for DOI: {doi}")
    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Crossref API returned HTTP {response.status_code}.",
        )

    try:
        payload = response.json()
        message = payload["message"]
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=502, detail="Crossref returned an invalid response.") from exc

    metadata = crossref_to_metadata(doi, message)

    try:
        append_metadata_to_excel(metadata, API_EXCEL_FILE_PATH)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not save API data to Excel: {exc}") from exc

    # Count saved papers so the UI can confirm that multiple DOI records are accumulating.
    from openpyxl import load_workbook
    try:
        wb = load_workbook(API_EXCEL_FILE_PATH, read_only=True)
        saved_count = max(wb.active.max_row - 1, 0)
        wb.close()
    except Exception:
        saved_count = None

    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "message": "Data has been saved successfully. Enter another DOI.",
            "doi": doi,
            "saved_count": saved_count,
            "metadata": metadata,
        },
    )


@app.get("/download-api-excel")
async def download_api_excel():
    """Download the aggregate Excel containing all DOI/API records saved since refresh."""
    if not os.path.exists(API_EXCEL_FILE_PATH):
        raise HTTPException(status_code=404, detail="No API paper data has been saved yet.")

    return FileResponse(
        path=API_EXCEL_FILE_PATH,
        filename="papers_metadata_from_api.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
