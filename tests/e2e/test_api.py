import pytest
from fastapi.testclient import TestClient
from api.server import app                 

client = TestClient(app)

@pytest.mark.parametrize("k", [1, 3])
def test_search_endpoint(tmp_path, tiny_png_bytes, k):
    img = tmp_path / "q.png"
    img.write_bytes(tiny_png_bytes)

    resp = client.post(
        "/search/",
        params={"model_name": "facebook/dinov2-base", "top_k": k},
        files={"file": ("q.png", img.read_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == k
