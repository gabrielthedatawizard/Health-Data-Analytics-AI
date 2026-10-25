"""
Base Connector Interface for Health Data Analytics AI
Phase 5.2: Multi-Source Data Connectors
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd


class BaseConnector(ABC):
    """
    Abstract interface for enterprise and external health data connectors.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.is_connected = False

    @abstractmethod
    def connect(self) -> bool:
        """Establishes connection to the data source."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Releases active connections or session resources."""
        pass

    @abstractmethod
    def test_connection(self) -> Dict[str, Any]:
        """Validates connection health, credentials, and reachability."""
        pass

    @abstractmethod
    def fetch_schema(self) -> List[Dict[str, Any]]:
        """Retrieves list of available tables, collections, or indicators."""
        pass

    @abstractmethod
    def fetch_data(self, query_or_params: Any, limit: Optional[int] = None) -> pd.DataFrame:
        """Fetches data into a structured Pandas DataFrame."""
        pass
