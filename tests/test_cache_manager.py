from backend.cache import CacheManager


def test_cache_hit_miss_and_invalidate() -> None:
    cache = CacheManager()
    key = "cache:v1:test:artifact:hash:schema:params"
    value = {"ok": True}

    cache.set(key, value)
    assert cache.get(key) == value

    cache.invalidate_prefix("cache:v1:test:")
    assert cache.get(key) is None
