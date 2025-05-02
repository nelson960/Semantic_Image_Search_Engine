import torch


def test_shape(tmp_path, tiny_png_bytes):
    from extractor.extract import extract_single_image_embedding
    img_path = tmp_path / "tiny.png"
    img_path.write_bytes(tiny_png_bytes)

    emb = extract_single_image_embedding(str(img_path))
    assert emb.shape == torch.Size([1024])
    assert emb.dtype == torch.float32

