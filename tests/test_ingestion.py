import pytest
import io
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.vector_store import vector_store_service

client = TestClient(app)


@pytest.fixture(autouse=True)
def cleanup_vector_store():
    """Ensure vector store is clean before each test execution."""
    indexed_docs = vector_store_service.list_indexed_documents()
    for doc in indexed_docs:
        vector_store_service.delete_document_by_id(doc["doc_id"])
    yield
    indexed_docs = vector_store_service.list_indexed_documents()
    for doc in indexed_docs:
        vector_store_service.delete_document_by_id(doc["doc_id"])


def test_upload_identical_file_replaces_existing_document():
    """
    Uploading identical file bytes twice should replace the document
    and leave exactly ONE document entry / set of chunks in ChromaDB.
    """
    file_content = b"This is a test document content for SHA-256 duplicate detection verification."
    file_name = "test_duplicate.txt"

    # First upload
    res1 = client.post(
        "/api/v1/documents/upload",
        files={"file": (file_name, io.BytesIO(file_content), "text/plain")}
    )
    assert res1.status_code == 200
    data1 = res1.json()
    first_doc_id = data1["document"]["doc_id"]
    assert data1["replaced_doc_id"] is None

    # Check indexed count
    docs_after_first = vector_store_service.list_indexed_documents()
    assert len(docs_after_first) == 1
    assert docs_after_first[0]["doc_id"] == first_doc_id

    # Second upload with IDENTICAL bytes
    res2 = client.post(
        "/api/v1/documents/upload",
        files={"file": (file_name, io.BytesIO(file_content), "text/plain")}
    )
    assert res2.status_code == 200
    data2 = res2.json()
    second_doc_id = data2["document"]["doc_id"]

    # Verify replace behavior
    assert second_doc_id != first_doc_id
    assert data2["replaced_doc_id"] == first_doc_id

    # Check that ONLY ONE document remains indexed in ChromaDB
    docs_after_second = vector_store_service.list_indexed_documents()
    assert len(docs_after_second) == 1
    assert docs_after_second[0]["doc_id"] == second_doc_id


def test_upload_same_filename_different_content_keeps_both_documents():
    """
    Uploading different file contents with the same filename should keep BOTH
    documents indexed in ChromaDB with distinct doc_ids.
    """
    content1 = b"Version 1 of project specification document."
    content2 = b"Version 2 updated project specification document with changes."
    file_name = "project_spec.txt"

    # Upload version 1
    res1 = client.post(
        "/api/v1/documents/upload",
        files={"file": (file_name, io.BytesIO(content1), "text/plain")}
    )
    assert res1.status_code == 200
    doc_id_1 = res1.json()["document"]["doc_id"]

    # Upload version 2 (same filename, different content bytes)
    res2 = client.post(
        "/api/v1/documents/upload",
        files={"file": (file_name, io.BytesIO(content2), "text/plain")}
    )
    assert res2.status_code == 200
    data2 = res2.json()
    doc_id_2 = data2["document"]["doc_id"]

    # Verify NO document was replaced
    assert data2["replaced_doc_id"] is None
    assert doc_id_1 != doc_id_2

    # Check that BOTH documents exist in ChromaDB index
    docs = vector_store_service.list_indexed_documents()
    assert len(docs) == 2
    indexed_doc_ids = {doc["doc_id"] for doc in docs}
    assert doc_id_1 in indexed_doc_ids
    assert doc_id_2 in indexed_doc_ids