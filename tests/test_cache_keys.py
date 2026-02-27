from backend.main import _cache_key


def test_cache_key_stable() -> None:
    key = _cache_key("ds_123", "facts", "hashabc", schema_hash="schema123", params="sample")
    assert key == "cache:v1:ds_123:facts:hashabc:schema123:sample"
