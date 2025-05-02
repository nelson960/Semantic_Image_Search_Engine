from indexer.search import search_similar_images

def test_search_returns_top_k(image_dir, tmp_path, tiny_png_bytes):
    query = tmp_path / "q.png"
    query.write_bytes(tiny_png_bytes)

    results = search_similar_images(
        query_image_path=str(query),
        model_name="facebook/dinov2-base",
        index_path="ignored.faiss",
        image_dir=image_dir,
        top_k=2,
    )
    assert len(results) == 2
    # distances should be non‑decreasing
    assert [d for _, d in results] == sorted(d for _, d in results)
