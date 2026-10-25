"""
Multi-Source Data Connectors Package for Health Data Analytics AI
Phase 5.2: Multi-Source Data Connectors
"""

from backend.connectors.base import BaseConnector
from backend.connectors.dhis2_connector import DHIS2Connector
from backend.connectors.sql_connector import SQLConnector
from backend.connectors.rest_connector import RESTAPIConnector

__all__ = [
    "BaseConnector",
    "DHIS2Connector",
    "SQLConnector",
    "RESTAPIConnector",
    "get_connector",
]


def get_connector(connector_type: str, **kwargs) -> BaseConnector:
    """
    Factory function to instantiate connectors by type name.
    Supported types: 'dhis2', 'sql', 'rest'.
    """
    c_type = connector_type.lower().strip()
    if c_type == "dhis2":
        return DHIS2Connector(**kwargs)
    elif c_type in ("sql", "postgres", "mysql", "sqlite"):
        return SQLConnector(**kwargs)
    elif c_type in ("rest", "api", "http"):
        return RESTAPIConnector(**kwargs)
    else:
        raise ValueError(f"Unknown connector type: {connector_type}. Supported: 'dhis2', 'sql', 'rest'.")
