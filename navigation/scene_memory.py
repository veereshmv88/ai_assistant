"""
navigation/scene_memory.py — Persistent Scene Memory
=====================================================
SQLite-backed memory store. Records:
  • Scene descriptions with GPS location + timestamp
  • Recognised faces with location + time
  • OCR text (signs, documents) with location
  • Visited landmarks

Enables queries like:
  • "What did I pass 5 minutes ago?"
  • "Have I been here before?"
  • "Where did I last see Mom?"

Provides:
  • async store_scene(description, gps_fix)
  • async store_face(name, gps_fix)
  • async store_text(text, gps_fix)
  • async recall_recent(n)                → list[MemoryEntry]
  • async recall_near(lat, lon, radius_m) → list[MemoryEntry]
  • async initialise() / cleanup()
"""

import asyncio
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import Config
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class MemoryEntry:
    id: int
    entry_type: str          # "scene" | "face" | "text" | "landmark"
    content: str
    latitude: Optional[float]
    longitude: Optional[float]
    timestamp: float

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    @property
    def age_human(self) -> str:
        secs = int(self.age_seconds)
        if secs < 60:
            return f"{secs} seconds ago"
        elif secs < 3600:
            return f"{secs // 60} minutes ago"
        else:
            return f"{secs // 3600} hours ago"

    def __str__(self):
        loc = f" @ ({self.latitude:.4f},{self.longitude:.4f})" if self.latitude else ""
        return f"[{self.entry_type.upper()}]{loc} {self.age_human}: {self.content[:80]}"


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_type  TEXT NOT NULL,
    content     TEXT NOT NULL,
    latitude    REAL,
    longitude   REAL,
    timestamp   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_timestamp ON memories (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_type ON memories (entry_type);
"""


class SceneMemory:
    """
    Async SQLite-backed scene memory.
    All DB operations run in an executor thread (sqlite3 is synchronous).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._db_path = cfg.navigation.SCENE_MEMORY_DB
        self._conn: Optional[sqlite3.Connection] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    async def initialise(self):
        if not self.cfg.system.ENABLE_SCENE_MEMORY:
            log.info("Scene memory: disabled.")
            return
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._init_db)
        log.info(f"Scene memory initialised: {self._db_path}")

    def _init_db(self):
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_CREATE_TABLE_SQL)
        self._conn.commit()

    async def cleanup(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Storage ───────────────────────────────────────────────────────────────
    async def store_scene(self, description: str, gps_fix=None):
        await self._insert(
            entry_type="scene",
            content=description,
            gps_fix=gps_fix,
        )

    async def store_face(self, name: str, gps_fix=None):
        await self._insert(
            entry_type="face",
            content=f"Saw {name}",
            gps_fix=gps_fix,
        )

    async def store_text(self, text: str, gps_fix=None):
        await self._insert(
            entry_type="text",
            content=text,
            gps_fix=gps_fix,
        )

    async def store_landmark(self, name: str, gps_fix=None):
        await self._insert(
            entry_type="landmark",
            content=name,
            gps_fix=gps_fix,
        )

    async def _insert(self, entry_type: str, content: str, gps_fix=None):
        if not self._conn:
            return
        lat = gps_fix.latitude if gps_fix else None
        lon = gps_fix.longitude if gps_fix else None
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._insert_sync(entry_type, content, lat, lon),
        )

    def _insert_sync(self, entry_type, content, lat, lon):
        self._conn.execute(
            "INSERT INTO memories (entry_type, content, latitude, longitude, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (entry_type, content, lat, lon, time.time()),
        )
        self._conn.commit()

    # ── Recall ────────────────────────────────────────────────────────────────
    async def recall_recent(self, n: int = 10) -> list[MemoryEntry]:
        """Return the N most recent memory entries."""
        if not self._conn:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._query_recent(n)
        )

    def _query_recent(self, n: int) -> list[MemoryEntry]:
        rows = self._conn.execute(
            "SELECT * FROM memories ORDER BY timestamp DESC LIMIT ?", (n,)
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    async def recall_near(
        self,
        lat: float,
        lon: float,
        radius_m: float = 50.0,
    ) -> list[MemoryEntry]:
        """Return memory entries recorded near a given GPS position."""
        if not self._conn:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._query_near(lat, lon, radius_m)
        )

    def _query_near(self, lat: float, lon: float, radius_m: float) -> list[MemoryEntry]:
        """Simple bounding-box query (good enough for small radius)."""
        import math
        # 1 degree latitude ≈ 111 km
        lat_delta = radius_m / 111_000
        lon_delta = radius_m / (111_000 * math.cos(math.radians(lat)))
        rows = self._conn.execute(
            "SELECT * FROM memories "
            "WHERE latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ? "
            "ORDER BY timestamp DESC LIMIT 20",
            (lat - lat_delta, lat + lat_delta, lon - lon_delta, lon + lon_delta),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    async def recall_by_type(self, entry_type: str, n: int = 5) -> list[MemoryEntry]:
        """Return recent entries of a specific type."""
        if not self._conn:
            return []
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: [
                self._row_to_entry(r)
                for r in self._conn.execute(
                    "SELECT * FROM memories WHERE entry_type=? ORDER BY timestamp DESC LIMIT ?",
                    (entry_type, n),
                ).fetchall()
            ],
        )

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        return MemoryEntry(
            id=row["id"],
            entry_type=row["entry_type"],
            content=row["content"],
            latitude=row["latitude"],
            longitude=row["longitude"],
            timestamp=row["timestamp"],
        )

    def is_ready(self) -> bool:
        return self._conn is not None or not self.cfg.system.ENABLE_SCENE_MEMORY
