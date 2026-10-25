"""
High-Performance Compute Engine for Health Data Analytics AI
Phase 5.1: Advanced Data Processing

Adaptive compute engine leveraging DuckDB columnar execution and vectorized operations:
- Auto-selects DuckDB vs Pandas based on dataset size and operation complexity
- High-speed columnar SQL queries directly over Pandas DataFrames
- Streaming ingestion for memory-constrained large dataset handling
- Vectorized summary aggregation and parallel profiling
"""

from __future__ import annotations
import math
import os
import time
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


class ComputeEngine:
    """
    Adaptive high-performance compute engine.
    Switches between DuckDB columnar SQL engine and Pandas based on workload size.
    """

    def __init__(self, duckdb_memory_limit: str = "2GB", threads: int = 4):
        self.duckdb_memory_limit = duckdb_memory_limit
        self.threads = threads
        self._con: Optional[Any] = None

    def _get_connection(self) -> Any:
        if not DUCKDB_AVAILABLE:
            raise RuntimeError("DuckDB is not installed or available.")
        if self._con is None:
            self._con = duckdb.connect(database=":memory:")
            try:
                self._con.execute(f"SET memory_limit='{self.duckdb_memory_limit}'")
                self._con.execute(f"SET threads={self.threads}")
            except Exception:
                pass
        return self._con

    # ─────────────────────────────────────────────────────────────
    # 1. Backend Auto-Selection
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def auto_select_backend(df_or_rows: Union[pd.DataFrame, int], operation: str = "aggregate") -> str:
        """
        Selects optimal execution backend based on row count and operation type.
        - Rows >= 10,000 or SQL aggregations: DuckDB (10-50x faster)
        - Rows < 10,000 simple operations: Pandas (zero setup overhead)
        """
        row_count = len(df_or_rows) if isinstance(df_or_rows, pd.DataFrame) else int(df_or_rows)

        if not DUCKDB_AVAILABLE:
            return "pandas"

        if operation in ("sql", "join", "distinct_count", "complex_filter"):
            return "duckdb"

        if row_count >= 10_000:
            return "duckdb"

        return "pandas"

    # ─────────────────────────────────────────────────────────────
    # 2. Vectorized SQL Query Execution (DuckDB over DataFrames)
    # ─────────────────────────────────────────────────────────────

    def execute_sql(
        self,
        query: str,
        tables: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Executes vectorized SQL directly against in-memory DataFrames using DuckDB.
        Tables dictionary maps SQL table aliases to Pandas DataFrames.
        """
        if not DUCKDB_AVAILABLE:
            raise RuntimeError("DuckDB backend is required for execute_sql.")

        con = self._get_connection()

        # Register input DataFrames into DuckDB virtual catalog
        registered_names = []
        if tables:
            for name, df in tables.items():
                con.register(name, df)
                registered_names.append(name)

        try:
            rel = con.execute(query)
            res_df = rel.df()
            return res_df
        finally:
            # Unregister to free memory
            for name in registered_names:
                try:
                    con.unregister(name)
                except Exception:
                    pass

    # ─────────────────────────────────────────────────────────────
    # 3. Fast Columnar Aggregations (Segments & KPI Summaries)
    # ─────────────────────────────────────────────────────────────

    def compute_aggregations(
        self,
        df: pd.DataFrame,
        group_by: List[str],
        metrics: List[str],
        operations: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        High-performance multi-metric aggregation.
        Uses DuckDB when available for columnar speedup, falls back to Pandas.
        """
        if df.empty or not metrics:
            return pd.DataFrame()

        ops = operations or ["sum", "mean", "min", "max", "count"]
        backend = self.auto_select_backend(df, operation="aggregate")

        if backend == "duckdb" and DUCKDB_AVAILABLE:
            select_clauses = [f'"{col}"' for col in group_by]
            for m in metrics:
                for op in ops:
                    op_upper = op.upper()
                    if op_upper == "MEAN":
                        op_upper = "AVG"
                    alias = f"{m}_{op}"
                    select_clauses.append(f'{op_upper}("{m}") AS "{alias}"')

            group_clause = ", ".join(f'"{col}"' for col in group_by) if group_by else "1"
            select_str = ", ".join(select_clauses)
            sql = f"SELECT {select_str} FROM input_df GROUP BY {group_clause}"

            return self.execute_sql(sql, tables={"input_df": df})

        # Pandas fallback
        agg_dict = {m: ops for m in metrics}
        if group_by:
            res = df.groupby(group_by, as_index=False).agg(agg_dict)
            # Flatten multi-index columns if present
            if isinstance(res.columns, pd.MultiIndex):
                res.columns = [
                    f"{col[0]}_{col[1]}" if col[1] else col[0]
                    for col in res.columns
                ]
            return res
        else:
            return df[metrics].agg(ops)

    # ─────────────────────────────────────────────────────────────
    # 4. Fast Parallel / Columnar Profiling
    # ─────────────────────────────────────────────────────────────

    def fast_profile_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes high-speed statistical profiling across all columns.
        Uses single-pass DuckDB query if rows >= 5000.
        """
        n_rows = len(df)
        n_cols = len(df.columns)
        if n_rows == 0:
            return {"row_count": 0, "column_count": n_cols, "columns": {}}

        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not str(c).startswith("_")]

        if DUCKDB_AVAILABLE and n_rows >= 5000 and numeric_cols:
            try:
                clauses = []
                for c in numeric_cols:
                    clauses.append(f'SUM(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) AS "{c}_nulls"')
                    clauses.append(f'AVG("{c}") AS "{c}_mean"')
                    clauses.append(f'MIN("{c}") AS "{c}_min"')
                    clauses.append(f'MAX("{c}") AS "{c}_max"')
                    clauses.append(f'STDDEV_SAMP("{c}") AS "{c}_std"')

                sql = f"SELECT {', '.join(clauses)} FROM df_profile"
                stat_row = self.execute_sql(sql, tables={"df_profile": df}).iloc[0].to_dict()

                col_summaries = {}
                for c in numeric_cols:
                    nulls = int(stat_row.get(f"{c}_nulls", 0))
                    col_summaries[c] = {
                        "missing_count": nulls,
                        "missing_pct": round(nulls / n_rows * 100, 2),
                        "mean": round(float(stat_row.get(f"{c}_mean") or 0.0), 3),
                        "min": round(float(stat_row.get(f"{c}_min") or 0.0), 3),
                        "max": round(float(stat_row.get(f"{c}_max") or 0.0), 3),
                        "std": round(float(stat_row.get(f"{c}_std") or 0.0), 3),
                    }
                return {
                    "backend_used": "duckdb",
                    "row_count": n_rows,
                    "column_count": n_cols,
                    "columns": col_summaries,
                }
            except Exception:
                pass

        # Standard Pandas profiling
        col_summaries = {}
        for c in numeric_cols:
            s = df[c].dropna()
            nulls = int(df[c].isna().sum())
            col_summaries[c] = {
                "missing_count": nulls,
                "missing_pct": round(nulls / max(n_rows, 1) * 100, 2),
                "mean": round(float(s.mean() if len(s) else 0.0), 3),
                "min": round(float(s.min() if len(s) else 0.0), 3),
                "max": round(float(s.max() if len(s) else 0.0), 3),
                "std": round(float(s.std() if len(s) > 1 else 0.0), 3),
            }

        return {
            "backend_used": "pandas",
            "row_count": n_rows,
            "column_count": n_cols,
            "columns": col_summaries,
        }

    # ─────────────────────────────────────────────────────────────
    # 5. Memory-Constrained Streaming Ingestion
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def stream_csv_chunks(
        file_path: str, chunk_size: int = 50_000
    ) -> Generator[pd.DataFrame, None, None]:
        """
        Yields DataFrames in fixed-size chunks to support arbitrarily large files
        without exceeding server memory boundaries.
        """
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
            yield chunk

    def process_large_csv_summary(
        self,
        file_path: str,
        numeric_cols: List[str],
        chunk_size: int = 50_000,
    ) -> Dict[str, Any]:
        """
        Calculates cumulative statistics across a multi-million-row CSV without loading it all into memory.
        """
        totals: Dict[str, float] = {c: 0.0 for c in numeric_cols}
        counts: Dict[str, int] = {c: 0 for c in numeric_cols}
        mins: Dict[str, float] = {c: float("inf") for c in numeric_cols}
        maxs: Dict[str, float] = {c: float("-inf") for c in numeric_cols}
        total_rows = 0

        for chunk in self.stream_csv_chunks(file_path, chunk_size=chunk_size):
            total_rows += len(chunk)
            for c in numeric_cols:
                if c in chunk.columns:
                    s = pd.to_numeric(chunk[c], errors="coerce").dropna()
                    if not s.empty:
                        totals[c] += float(s.sum())
                        counts[c] += int(s.count())
                        mins[c] = min(mins[c], float(s.min()))
                        maxs[c] = max(maxs[c], float(s.max()))

        kpis = {}
        for c in numeric_cols:
            cnt = counts[c]
            kpis[c] = {
                "total": round(totals[c], 2),
                "count": cnt,
                "mean": round(totals[c] / cnt, 2) if cnt > 0 else 0.0,
                "min": mins[c] if mins[c] != float("inf") else 0.0,
                "max": maxs[c] if maxs[c] != float("-inf") else 0.0,
            }

        return {"total_rows_streamed": total_rows, "kpis": kpis}
