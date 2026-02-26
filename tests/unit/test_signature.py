from humpback.processing.signature import compute_encoding_signature


def test_idempotent():
    sig1 = compute_encoding_signature("v1", 5.0, 32000, {"type": "logmel"})
    sig2 = compute_encoding_signature("v1", 5.0, 32000, {"type": "logmel"})
    assert sig1 == sig2


def test_different_params_different_signature():
    sig1 = compute_encoding_signature("v1", 5.0, 32000)
    sig2 = compute_encoding_signature("v1", 10.0, 32000)
    assert sig1 != sig2


def test_dict_key_order_irrelevant():
    sig1 = compute_encoding_signature("v1", 5.0, 32000, {"a": 1, "b": 2})
    sig2 = compute_encoding_signature("v1", 5.0, 32000, {"b": 2, "a": 1})
    assert sig1 == sig2


def test_none_feature_config():
    sig = compute_encoding_signature("v1", 5.0, 32000, None)
    assert isinstance(sig, str)
    assert len(sig) == 64  # SHA-256 hex
