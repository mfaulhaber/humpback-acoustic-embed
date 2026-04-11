"""Smoke test confirming PyTorch is installed and usable.

PyTorch is bundled into the tf-* extras alongside TensorFlow so that Pass 2
and Pass 3 of the call parsing pipeline can train models. This test fails
fast if the dependency is missing from the environment.
"""


def test_pytorch_importable_and_basic_ops_work() -> None:
    import torch

    assert torch.__version__, "torch.__version__ should be non-empty"

    a = torch.zeros(2, 3)
    assert a.shape == (2, 3)

    b = torch.ones(3, 2)
    product = a @ b
    assert product.shape == (2, 2)
    assert torch.all(product == 0)
