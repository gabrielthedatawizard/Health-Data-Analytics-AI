import pandas as pd

from backend.main import _deterministic_sample


def test_deterministic_sampling_reproducible() -> None:
    df = pd.DataFrame({"a": range(1000)})
    s1 = _deterministic_sample(df, max_rows=100, seed=42)
    s2 = _deterministic_sample(df, max_rows=100, seed=42)
    assert s1.index.tolist() == s2.index.tolist()
