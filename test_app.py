import os
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    print("Health check passed.")

def test_upload_supported_file():
    test_file_path = "test_upload.txt"
    with open(test_file_path, "w") as f:
        f.write("This is a test document for ingestion pipeline validation. " * 50)
        
    with open(test_file_path, "rb") as f:
        response = client.post("/upload", files={"file": ("test_upload.txt", f, "text/plain")})
        
    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert data["filename"] == "test_upload.txt"
    assert data["file_type"] == ".txt"
    assert data["chunk_count"] > 0
    assert "metadata" in data
    print("Supported file upload passed. Metadata:", data)
    
    os.remove(test_file_path)

def test_upload_unsupported_file():
    test_file_path = "test_upload.md"
    with open(test_file_path, "w") as f:
        f.write("# Hello")
        
    with open(test_file_path, "rb") as f:
        response = client.post("/upload", files={"file": ("test_upload.md", f, "text/markdown")})
        
    assert response.status_code == 400
    data = response.json()
    assert "Unsupported file format" in data["detail"]
    print("Unsupported file validation passed.")
    
    os.remove(test_file_path)
    
if __name__ == "__main__":
    test_health()
    test_upload_supported_file()
    test_upload_unsupported_file()
    print("All tests passed successfully!")
