import numpy as np

from humpback.processing.audio_io import decode_audio, resample


def test_decode_wav(test_wav):
    audio, sr = decode_audio(test_wav)
    assert sr == 16000
    assert audio.dtype == np.float32
    assert len(audio) == 16000 * 10  # 10 seconds
    assert np.abs(audio).max() <= 1.0


def test_decode_flac(test_flac):
    audio, sr = decode_audio(test_flac)
    assert sr == 16000
    assert audio.dtype == np.float32
    assert len(audio) == 16000 * 10  # 10 seconds
    assert np.abs(audio).max() <= 1.0


def test_resample_noop():
    audio = np.random.randn(16000).astype(np.float32)
    result = resample(audio, 16000, 16000)
    np.testing.assert_array_equal(result, audio)


def test_resample_changes_length():
    audio = np.random.randn(16000).astype(np.float32)
    result = resample(audio, 16000, 32000)
    assert len(result) == 32000
