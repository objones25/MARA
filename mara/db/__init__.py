"""MARA leaf database package.

Public exports:
    LeafRepository          — Protocol (structural interface)
    SQLiteLeafRepository    — Concrete SQLite implementation
"""

from mara.db.repository import LeafRepository
from mara.db.sqlite_repository import SQLiteLeafRepository

__all__ = ["LeafRepository", "SQLiteLeafRepository"]
