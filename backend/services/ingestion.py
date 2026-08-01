import csv
import io
import uuid
import hashlib
from typing import List, Dict, Any, Tuple
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import HTTPException, status

from backend.config import settings


def compute_content_hash(file_bytes: bytes) -> str:
    """Computes SHA-256 hash of raw document bytes."""
    return hashlib.sha256(file_bytes).hexdigest().strip()


def validate_file(file_bytes: bytes, filename: str) -> str:
    """
    Validates file size and extension against settings in config.py.
    Returns the file extension without a leading dot.
    """
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size exceeds maximum allowed limit of {settings.MAX_FILE_SIZE_MB} MB. "
                f"Current size: {len(file_bytes) / (1024 * 1024):.2f} MB"
            )
        )

    ext = f".{filename.split('.')[-1].lower()}" if "." in filename else ""
    if ext not in settings.ALLOWED_EXTENSIONS:
        allowed_str = ", ".join(sorted(settings.ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Supported formats are: {allowed_str}"
        )

    return ext.lstrip(".")


def process_csv_file(file_bytes: bytes) -> List[Tuple[int, str]]:
    """
    Parses CSV content into structured row-based text snippets paired with row numbers.
    Handles multiple text encodings.
    """
    text_content = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            text_content = file_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if text_content is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode CSV file. File must be valid UTF-8 or Latin-1."
        )

    csv_file = io.StringIO(text_content)
    reader = csv.DictReader(csv_file)

    rows = []
    for row_idx, row in enumerate(reader, start=1):
        row_pairs = [f"{col.strip()}: {val.strip()}" for col, val in row.items() if col and val]
        if row_pairs:
            row_text = f"Row {row_idx}: " + " | ".join(row_pairs)
            rows.append((row_idx, row_text))

    return rows


def extract_pages_from_pdf(file_bytes: bytes) -> List[Tuple[int, str]]:
    """
    Extracts text page-by-page from PDF bytes using PyPDF.
    Returns a list of tuples: (page_number, page_text).
    """
    try:
        pdf_file = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_file)

        if len(reader.pages) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded PDF file contains no pages."
            )

        pages = []
        for index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((index + 1, text))

        if not pages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not extract readable text from the provided PDF file."
            )
        return pages
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse corrupt or invalid PDF document: {str(e)}"
        )


def extract_text_from_plain_file(file_bytes: bytes) -> List[Tuple[int, str]]:
    """
    Extracts text from UTF-8/Latin-1 encoded plain text (.txt) or markdown (.md) files.
    Assigns page number = 1.
    """
    text = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = file_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode file. Ensure file is UTF-8 or Latin-1 encoded."
        )

    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded text file is empty."
        )

    return [(1, text)]


def process_and_chunk_document(
    file_bytes: bytes,
    filename: str
) -> Tuple[str, List[str], List[Dict[str, Any]], int]:
    """
    Main ingestion pipeline: validates, computes hash, parses format, and chunks content.
    """
    ext = validate_file(file_bytes, filename)
    doc_id = str(uuid.uuid4())
    content_hash = compute_content_hash(file_bytes)

    chunks: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    # Format-specific extraction
    if ext == "pdf":
        pages = extract_pages_from_pdf(file_bytes)
    elif ext == "csv":
        csv_rows = process_csv_file(file_bytes)
        total_rows = len(csv_rows)

        for row_num, row_text in csv_rows:
            chunk_id = f"{doc_id}_r{row_num}"
            chunks.append(row_text)
            metadatas.append({
                "doc_id": doc_id,
                "content_hash": content_hash,
                "file_name": filename,
                "row_number": row_num,
                "page_number": row_num,
                "chunk_id": chunk_id,
                "total_pages": total_rows,
                "file_type": "csv"
            })

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CSV processing yielded zero text chunks."
            )

        return doc_id, chunks, metadatas, total_rows
    else:
        pages = extract_text_from_plain_file(file_bytes)

    total_pages = len(pages)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False
    )

    for page_num, page_text in pages:
        page_chunks = text_splitter.split_text(page_text)
        for idx, chunk in enumerate(page_chunks):
            chunk_id = f"{doc_id}_p{page_num}_c{idx}"
            chunks.append(chunk)
            metadatas.append({
                "doc_id": doc_id,
                "content_hash": content_hash,
                "file_name": filename,
                "page_number": page_num,
                "chunk_id": chunk_id,
                "total_pages": total_pages,
                "file_type": ext
            })

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document processing yielded zero text chunks."
        )

    return doc_id, chunks, metadatas, total_pages