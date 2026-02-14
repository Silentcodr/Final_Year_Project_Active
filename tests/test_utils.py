from main import calculate_hash, hash_difference_percentage
from unittest.mock import mock_open, patch

def test_hash_difference_percentage():
    hash1 = "abcdef"
    hash2 = "abcde1"
    # 1 char diff out of 6 -> 1/6 * 100 = 16.666...
    diff = hash_difference_percentage(hash1, hash2)
    assert 16.0 < diff < 17.0

def test_calculate_hash():
    with patch("builtins.open", mock_open(read_data=b"data")) as mock_file:
        h = calculate_hash("dummy_path")
        # sha256 of "data"
        # echo -n "data" | sha256sum -> 3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7
        assert h == "3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7"
