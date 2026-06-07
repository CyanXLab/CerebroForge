"""
CerebroForge (铸脑) - 3-Layer Memory System
=============================================
Three-layer memory architecture with SQLite + ChromaDB:
  L1 — Working / Episodic (short-term, high-fidelity)
  L2 — Semantic / Compressed (patterns extracted from L1)
  L3 — Wisdom / Deep Abstractions (cross-pattern insights)

Features:
  - Hybrid search: keyword (SQLite FTS) + vector (ChromaDB)
  - Brain-like compression: QUAD extraction (Goal|Action|Result|Error)
  - Forgetting rule: L3 items not accessed in 30 days → frozen (weight=0.1)
  - Weight-based retrieval: only fragments with weight > 0.6
  - Index key mechanism: compressed memories include reversible lookup key
  - Tool registry with usage statistics
  - Evolution logging with EGL tracking
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from backend.config import (
        CHROMA_DIR,
        DATA_DIR,
        DB_PATH,
        L1_MAX_ENTRIES,
        L1_MAX_TOKENS,
        L3_FREEZE_DAYS,
        L3_FROZEN_WEIGHT,
        RETRIEVAL_MIN_WEIGHT,
    )
except ImportError:
    from config import (
        CHROMA_DIR,
        DATA_DIR,
        DB_PATH,
        L1_MAX_ENTRIES,
        L1_MAX_TOKENS,
        L3_FREEZE_DAYS,
        L3_FROZEN_WEIGHT,
        RETRIEVAL_MIN_WEIGHT,
    )

try:
    from backend.schemas import MemoryLevel, MemoryOp
except ImportError:
    from schemas import MemoryLevel, MemoryOp

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Embedding Helper (sentence-transformers or TF-IDF fallback)
# ────────────────────────────────────────────────────────────────────────────

class EmbeddingProvider:
    """Provides vector embeddings for text. Falls back to TF-IDF if
    sentence-transformers is unavailable."""

    def __init__(self) -> None:
        self._model = None
        self._tfidf_fallback = False
        self._vocab: Dict[str, int] = {}
        self._dim = 384

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"Loaded sentence-transformers model (dim={self._dim})")
        except Exception as exc:
            logger.warning(
                f"sentence-transformers unavailable ({exc}), using TF-IDF fallback"
            )
            self._tfidf_fallback = True
            self._dim = 256

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> List[float]:
        """Generate an embedding for the given text."""
        if self._model is not None:
            return self._model.encode(text).tolist()
        return self._tfidf_embed(text)

    def _tfidf_embed(self, text: str) -> List[float]:
        """Simple TF-IDF–like embedding as a fallback."""
        import math
        from collections import Counter

        words = text.lower().split()
        word_counts = Counter(words)
        total = len(words) or 1

        # Build vocabulary on the fly
        for w in words:
            if w not in self._vocab:
                self._vocab[w] = len(self._vocab)

        # Create sparse vector (hash to fixed dimension)
        vec = [0.0] * self._dim
        for word, count in word_counts.items():
            tf = count / total
            idx = hash(word) % self._dim
            vec[idx] += tf

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vec = [v / norm for v in vec]
        return vec


# ────────────────────────────────────────────────────────────────────────────
# Memory System
# ────────────────────────────────────────────────────────────────────────────

class MemorySystem:
    """3-layer memory with SQLite + ChromaDB hybrid search."""

    def __init__(self, db_path: Optional[Path] = None, chroma_dir: Optional[Path] = None) -> None:
        self.db_path = db_path or DB_PATH
        self.chroma_dir = chroma_dir or CHROMA_DIR

        # Initialize embedding provider
        self._embedder = EmbeddingProvider()

        # Initialize SQLite
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

        # Initialize ChromaDB
        self._chroma = self._init_chroma()

    # ── SQLite Table Creation ──────────────────────────────────────────────

    def _create_tables(self) -> None:
        cursor = self._conn.cursor()

        # Memories table (L1/L2/L3)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                level TEXT NOT NULL DEFAULT 'L1',
                key TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL DEFAULT '',
                metadata TEXT NOT NULL DEFAULT '{}',
                weight REAL NOT NULL DEFAULT 1.0,
                token_count INTEGER NOT NULL DEFAULT 0,
                index_key TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                accessed_at REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0
            )
        """)

        # FTS5 virtual table for keyword search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING FTS5(key, content, content=memories, content_rowid=rowid)
        """)

        # Tools table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tools (
                name TEXT PRIMARY KEY,
                description TEXT NOT NULL DEFAULT '',
                code TEXT NOT NULL DEFAULT '',
                tool_type TEXT NOT NULL DEFAULT 'base',
                usage_count INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                avg_execution_time REAL NOT NULL DEFAULT 0.0,
                is_validated INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
        """)

        # Interactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL DEFAULT '',
                response TEXT NOT NULL DEFAULT '',
                tools_used TEXT NOT NULL DEFAULT '[]',
                system_mode TEXT NOT NULL DEFAULT '1',
                prediction_error REAL NOT NULL DEFAULT 0.0,
                duration_ms REAL NOT NULL DEFAULT 0.0,
                session_id TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL
            )
        """)

        # Evolution log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evolution_log (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                before_state TEXT NOT NULL DEFAULT '{}',
                after_state TEXT NOT NULL DEFAULT '{}',
                egl_delta REAL NOT NULL DEFAULT 0.0,
                created_at REAL NOT NULL
            )
        """)

        # Skills table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                name TEXT PRIMARY KEY,
                description TEXT NOT NULL DEFAULT '',
                steps TEXT NOT NULL DEFAULT '[]',
                tools_required TEXT NOT NULL DEFAULT '[]',
                success_rate REAL NOT NULL DEFAULT 0.0,
                usage_count INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

        self._conn.commit()

    # ── ChromaDB Initialization ────────────────────────────────────────────

    def _init_chroma(self):
        """Initialize ChromaDB with PersistentClient."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(self.chroma_dir))

            # Create or get collections for each memory level
            collections = {}
            for level in ("L1", "L2", "L3"):
                collections[level] = client.get_or_create_collection(
                    name=f"memory_{level}",
                    metadata={"hnsw:space": "cosine"},
                )

            return {"client": client, "collections": collections}
        except Exception as exc:
            logger.warning(f"ChromaDB initialization failed: {exc}, vector search disabled")
            return None

    # ── Memory Store ───────────────────────────────────────────────────────

    def store_memory(
        self,
        level: str,
        content: str,
        key: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        weight: float = 1.0,
    ) -> str:
        """Store a memory item at the specified level."""
        mem_id = str(uuid.uuid4())
        now = time.time()
        token_count = len(content.split())  # Approximate token count
        meta_json = json.dumps(metadata or {})
        index_key = self._generate_index_key(key, content)

        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO memories (id, level, key, content, metadata, weight,
                                  token_count, index_key, created_at, updated_at, accessed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (mem_id, level, key, content, meta_json, weight, token_count,
             index_key, now, now, now),
        )

        # Update FTS
        rowid = cursor.lastrowid
        cursor.execute(
            "INSERT INTO memories_fts(rowid, key, content) VALUES (?, ?, ?)",
            (rowid, key, content),
        )

        self._conn.commit()

        # Add to ChromaDB
        self._add_to_chroma(mem_id, content, level, metadata)

        # Check if L1 compression is needed
        if level == "L1" and self.should_compress_l1():
            self.compress_l1()

        return mem_id

    def _add_to_chroma(
        self,
        mem_id: str,
        content: str,
        level: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a memory item to ChromaDB for vector search."""
        if self._chroma is None:
            return

        try:
            embedding = self._embedder.embed(content)
            collection = self._chroma["collections"].get(level)
            if collection is None:
                return

            chroma_metadata = {"level": level, "key": metadata.get("key", "") if metadata else ""}
            if metadata:
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        chroma_metadata[k] = v

            collection.add(
                ids=[mem_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[chroma_metadata],
            )
        except Exception as exc:
            logger.warning(f"Failed to add to ChromaDB: {exc}")

    # ── Memory Retrieval (Hybrid) ─────────────────────────────────────────

    def retrieve_relevant(
        self,
        query: str,
        top_k: int = 5,
        levels: Optional[List[str]] = None,
        min_weight: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: keyword (SQLite FTS) + vector (ChromaDB).

        Results are merged and re-ranked by combined score.
        Only memories with weight > min_weight are returned.
        """
        levels = levels or ["L1", "L2", "L3"]
        min_weight = min_weight or RETRIEVAL_MIN_WEIGHT

        # Apply forgetting rule: update frozen L3 weights
        self._apply_forgetting_rule()

        # 1. Keyword search via FTS
        keyword_results = self._keyword_search(query, top_k * 2, levels, min_weight)

        # 2. Vector search via ChromaDB
        vector_results = self._vector_search(query, top_k * 2, levels, min_weight)

        # 3. Merge and re-rank
        merged = self._merge_search_results(keyword_results, vector_results, top_k)

        # Update access timestamps
        for item in merged:
            self._touch_memory(item["id"])

        return merged

    def _keyword_search(
        self,
        query: str,
        limit: int,
        levels: List[str],
        min_weight: float,
    ) -> List[Dict[str, Any]]:
        """FTS5 keyword search."""
        placeholders = ",".join("?" for _ in levels)
        try:
            rows = self._conn.execute(
                f"""
                SELECT m.* FROM memories m
                JOIN memories_fts fts ON m.rowid = fts.rowid
                WHERE memories_fts MATCH ?
                  AND m.level IN ({placeholders})
                  AND m.weight >= ?
                ORDER BY m.weight DESC, m.accessed_at DESC
                LIMIT ?
                """,
                [query] + levels + [min_weight, limit],
            ).fetchall()
        except sqlite3.OperationalError:
            # FTS might fail on special characters; fall back to LIKE
            rows = self._conn.execute(
                f"""
                SELECT * FROM memories
                WHERE (content LIKE ? OR key LIKE ?)
                  AND level IN ({placeholders})
                  AND weight >= ?
                ORDER BY weight DESC, accessed_at DESC
                LIMIT ?
                """,
                [f"%{query}%", f"%{query}%"] + levels + [min_weight, limit],
            ).fetchall()

        return [self._row_to_dict(row) for row in rows]

    def _vector_search(
        self,
        query: str,
        limit: int,
        levels: List[str],
        min_weight: float,
    ) -> List[Dict[str, Any]]:
        """ChromaDB vector similarity search."""
        if self._chroma is None:
            return []

        results: List[Dict[str, Any]] = []
        query_embedding = self._embedder.embed(query)

        for level in levels:
            try:
                collection = self._chroma["collections"].get(level)
                if collection is None:
                    continue

                chroma_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(limit, 20),
                    where={"level": level},
                )

                if chroma_results and chroma_results["ids"] and chroma_results["ids"][0]:
                    ids = chroma_results["ids"][0]
                    distances = chroma_results["distances"][0] if chroma_results["distances"] else [0.0] * len(ids)

                    for mem_id, distance in zip(ids, distances):
                        # Fetch from SQLite for full data
                        row = self._conn.execute(
                            "SELECT * FROM memories WHERE id = ? AND weight >= ?",
                            (mem_id, min_weight),
                        ).fetchone()
                        if row:
                            item = self._row_to_dict(row)
                            item["vector_score"] = 1.0 - distance  # Convert distance to similarity
                            results.append(item)

            except Exception as exc:
                logger.warning(f"Vector search failed for level {level}: {exc}")

        return results

    def _merge_search_results(
        self,
        keyword_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Merge and deduplicate keyword and vector search results."""
        combined: Dict[str, Dict[str, Any]] = {}

        for item in keyword_results:
            mid = item["id"]
            combined[mid] = {**item, "keyword_score": 1.0, "vector_score": 0.0}

        for item in vector_results:
            mid = item["id"]
            vs = item.pop("vector_score", 0.0)
            if mid in combined:
                combined[mid]["vector_score"] = vs
            else:
                combined[mid] = {**item, "keyword_score": 0.0, "vector_score": vs}

        # Compute combined score
        for item in combined.values():
            kw = item.get("keyword_score", 0.0)
            vs = item.get("vector_score", 0.0)
            weight = item.get("weight", 1.0)
            item["combined_score"] = (0.4 * kw + 0.6 * vs) * weight

        # Sort by combined score, descending
        sorted_items = sorted(
            combined.values(),
            key=lambda x: x.get("combined_score", 0.0),
            reverse=True,
        )

        return sorted_items[:top_k]

    # ── L1 Compression ────────────────────────────────────────────────────

    def should_compress_l1(self) -> bool:
        """Check if L1 compression should be triggered."""
        # Trigger when L1 > 100 entries or > 50K tokens
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt, COALESCE(SUM(token_count), 0) as total_tokens "
            "FROM memories WHERE level = 'L1'"
        ).fetchone()

        count = row["cnt"]
        total_tokens = row["total_tokens"]
        return count > L1_MAX_ENTRIES or total_tokens > L1_MAX_TOKENS

    def compress_l1(self) -> int:
        """
        Brain-like compression of L1 memories.

        For each batch of L1 memories, extract QUAD (Goal|Action|Result|Error)
        and store the compressed version in L2. Preserve index keys for
        reversible recall.
        """
        # Get L1 memories ordered by age (oldest first)
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE level = 'L1' ORDER BY created_at ASC"
        ).fetchall()

        if not rows:
            return 0

        compressed_count = 0
        batch_size = 5  # Compress in batches of 5

        for i in range(0, len(rows), batch_size):
            batch = rows[i: i + batch_size]

            # Build QUAD summary
            goals = []
            actions = []
            results = []
            errors = []

            for row in batch:
                item = self._row_to_dict(row)
                meta = item.get("metadata", {})
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except json.JSONDecodeError:
                        meta = {}

                content = item["content"]
                key = item["key"]

                # Extract QUAD components from content
                goal = meta.get("goal", content[:100])
                action = meta.get("action", "")
                result = meta.get("result", "")
                error = meta.get("error", "")

                goals.append(f"[{key}] {goal}" if key else goal)
                if action:
                    actions.append(action)
                if result:
                    results.append(result)
                if error:
                    errors.append(error)

            # Build compressed content
            compressed_parts = [
                f"GOAL: {'; '.join(goals)}",
                f"ACTION: {'; '.join(actions)}" if actions else "ACTION: N/A",
                f"RESULT: {'; '.join(results)}" if results else "RESULT: N/A",
                f"ERROR: {'; '.join(errors)}" if errors else "ERROR: None",
            ]
            compressed_content = "\n".join(compressed_parts)

            # Generate reversible index key
            index_key = self._generate_index_key(
                "", "_".join(item["key"] for item in [self._row_to_dict(r) for r in batch] if item["key"])
            )

            # Store compressed version in L2
            avg_weight = sum(self._row_to_dict(r)["weight"] for r in batch) / len(batch)
            self.store_memory(
                level="L2",
                content=compressed_content,
                key=f"compressed_l1_{i // batch_size}",
                metadata={
                    "type": "compressed_l1",
                    "source_ids": [self._row_to_dict(r)["id"] for r in batch],
                    "index_key": index_key,
                    "compression_ratio": len(batch),
                },
                weight=avg_weight,
            )

            # Archive original L1 memories (reduce weight, don't delete)
            for row in batch:
                item = self._row_to_dict(row)
                self._conn.execute(
                    "UPDATE memories SET level = 'L1_archived', weight = 0.3, updated_at = ? WHERE id = ?",
                    (time.time(), item["id"]),
                )

            compressed_count += len(batch)

        self._conn.commit()
        logger.info(f"Compressed {compressed_count} L1 memories into L2")
        return compressed_count

    # ── L2 → L3 Compression ───────────────────────────────────────────────

    def compress_l2(self) -> int:
        """
        Merge similar L2 patterns into L3 wisdom.

        Groups L2 memories by similarity of content and merges them into
        higher-level abstractions.
        """
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE level = 'L2' ORDER BY updated_at DESC"
        ).fetchall()

        if len(rows) < 3:
            return 0

        items = [self._row_to_dict(r) for r in rows]
        compressed_count = 0

        # Simple clustering by key prefix
        clusters: Dict[str, List[Dict[str, Any]]] = {}
        for item in items:
            key_prefix = item.get("key", "unknown").rsplit("_", 1)[0]
            clusters.setdefault(key_prefix, []).append(item)

        for prefix, cluster in clusters.items():
            if len(cluster) < 2:
                continue

            # Merge cluster into L3 wisdom
            merged_content = self._merge_l2_cluster(cluster)
            avg_weight = sum(it["weight"] for it in cluster) / len(cluster)
            index_key = self._generate_index_key(prefix, merged_content)

            self.store_memory(
                level="L3",
                content=merged_content,
                key=f"wisdom_{prefix}",
                metadata={
                    "type": "compressed_l2",
                    "source_ids": [it["id"] for it in cluster],
                    "index_key": index_key,
                    "cluster_size": len(cluster),
                },
                weight=min(avg_weight * 1.1, 1.0),  # Slight weight boost for wisdom
            )

            # Archive L2 items
            for item in cluster:
                self._conn.execute(
                    "UPDATE memories SET level = 'L2_archived', weight = 0.3, updated_at = ? WHERE id = ?",
                    (time.time(), item["id"]),
                )

            compressed_count += len(cluster)

        self._conn.commit()
        logger.info(f"Compressed {compressed_count} L2 memories into L3 wisdom")
        return compressed_count

    @staticmethod
    def _merge_l2_cluster(cluster: List[Dict[str, Any]]) -> str:
        """Merge a cluster of L2 memories into a single L3 wisdom entry."""
        all_goals = set()
        all_actions = set()
        all_results = set()
        all_errors = set()

        for item in cluster:
            content = item.get("content", "")
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("GOAL:"):
                    all_goals.add(line[5:].strip())
                elif line.startswith("ACTION:"):
                    all_actions.add(line[7:].strip())
                elif line.startswith("RESULT:"):
                    all_results.add(line[7:].strip())
                elif line.startswith("ERROR:"):
                    err = line[6:].strip()
                    if err and err != "None" and err != "N/A":
                        all_errors.add(err)

        parts = [
            f"WISDOM GOAL: {'; '.join(list(all_goals)[:5])}",
            f"PATTERN: {'; '.join(list(all_actions)[:5])}" if all_actions else "",
            f"OUTCOME: {'; '.join(list(all_results)[:5])}" if all_results else "",
            f"PITFALL: {'; '.join(list(all_errors)[:5])}" if all_errors else "",
        ]
        return "\n".join(p for p in parts if p)

    # ── Forgetting Rule ────────────────────────────────────────────────────

    def _apply_forgetting_rule(self) -> int:
        """
        Apply forgetting rule: L3 items not accessed in 30 days get
        weight dropped to 0.1 → 'frozen' (not deleted, lowest retrieval priority).
        """
        cutoff = time.time() - (L3_FREEZE_DAYS * 86400)

        cursor = self._conn.execute(
            """
            UPDATE memories
            SET weight = ?, updated_at = ?
            WHERE level = 'L3'
              AND accessed_at < ?
              AND weight > ?
            """,
            (L3_FROZEN_WEIGHT, time.time(), cutoff, L3_FROZEN_WEIGHT),
        )

        affected = cursor.rowcount
        if affected > 0:
            self._conn.commit()
            logger.info(f"Frozen {affected} L3 memories (not accessed in {L3_FREEZE_DAYS} days)")

        return affected

    # ── Interaction Recording ──────────────────────────────────────────────

    def record_interaction(
        self,
        query: str,
        response: str,
        tools_used: Optional[List[str]] = None,
        system_mode: str = "1",
        prediction_error: float = 0.0,
        duration_ms: float = 0.0,
        session_id: str = "",
    ) -> str:
        """Record an interaction with real timestamps."""
        interaction_id = str(uuid.uuid4())
        now = time.time()

        self._conn.execute(
            """
            INSERT INTO interactions (id, query, response, tools_used, system_mode,
                                       prediction_error, duration_ms, session_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (interaction_id, query, response, json.dumps(tools_used or []),
             system_mode, prediction_error, duration_ms, session_id, now),
        )
        self._conn.commit()
        return interaction_id

    # ── Tool Registry ──────────────────────────────────────────────────────

    def register_tool(
        self,
        name: str,
        description: str = "",
        code: str = "",
        tool_type: str = "base",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a new tool in the registry."""
        now = time.time()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO tools (name, description, code, tool_type,
                                           created_at, updated_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (name, description, code, tool_type, now, now, json.dumps(metadata or {})),
        )
        self._conn.commit()

    def update_tool_stats(
        self,
        name: str,
        success: bool,
        execution_time: float = 0.0,
    ) -> None:
        """Update tool usage statistics after an invocation."""
        tool = self._conn.execute(
            "SELECT * FROM tools WHERE name = ?", (name,)
        ).fetchone()

        if tool is None:
            return

        current_usage = tool["usage_count"]
        current_success = tool["success_count"]
        current_failure = tool["failure_count"]
        current_avg_time = tool["avg_execution_time"]

        new_usage = current_usage + 1
        new_success = current_success + (1 if success else 0)
        new_failure = current_failure + (0 if success else 1)
        new_avg_time = (current_avg_time * current_usage + execution_time) / new_usage if new_usage else 0.0

        self._conn.execute(
            """
            UPDATE tools SET usage_count = ?, success_count = ?, failure_count = ?,
                             avg_execution_time = ?, updated_at = ?
            WHERE name = ?
            """,
            (new_usage, new_success, new_failure, new_avg_time, time.time(), name),
        )
        self._conn.commit()

    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a tool by name."""
        row = self._conn.execute(
            "SELECT * FROM tools WHERE name = ?", (name,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all registered tools."""
        rows = self._conn.execute(
            "SELECT * FROM tools ORDER BY usage_count DESC"
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_high_freq_tools(self, threshold: Optional[int] = None) -> List[str]:
        """Get tool names that qualify as high-frequency based on usage count."""
        threshold = threshold or L1_MAX_ENTRIES  # Use HIGH_FREQ_THRESHOLD from config
        try:
            from backend.config import HIGH_FREQ_THRESHOLD as _hft
        except ImportError:
            from config import HIGH_FREQ_THRESHOLD as _hft
        threshold = _hft

        rows = self._conn.execute(
            """
            SELECT name FROM tools
            WHERE usage_count >= ? AND is_validated = 1
            ORDER BY usage_count DESC
            """,
            (threshold,),
        ).fetchall()
        return [row["name"] for row in rows]

    # ── Evolution Logging ──────────────────────────────────────────────────

    def log_evolution(
        self,
        event_type: str,
        description: str,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
        egl_delta: float = 0.0,
    ) -> str:
        """Log an evolution event with EGL (Evolutionary Growth Level) tracking."""
        event_id = str(uuid.uuid4())
        now = time.time()

        self._conn.execute(
            """
            INSERT INTO evolution_log (id, event_type, description, before_state,
                                        after_state, egl_delta, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (event_id, event_type, description,
             json.dumps(before_state or {}), json.dumps(after_state or {}),
             egl_delta, now),
        )
        self._conn.commit()
        return event_id

    def get_evolution_stats(self) -> Dict[str, Any]:
        """Get evolution statistics including EGL tracking."""
        total_events = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM evolution_log"
        ).fetchone()["cnt"]

        total_egl = self._conn.execute(
            "SELECT COALESCE(SUM(egl_delta), 0.0) as total FROM evolution_log"
        ).fetchone()["total"]

        by_type_rows = self._conn.execute(
            "SELECT event_type, COUNT(*) as cnt, SUM(egl_delta) as egl_sum "
            "FROM evolution_log GROUP BY event_type"
        ).fetchall()

        by_type = {
            row["event_type"]: {"count": row["cnt"], "egl": row["egl_sum"]}
            for row in by_type_rows
        }

        recent = self._conn.execute(
            "SELECT * FROM evolution_log ORDER BY created_at DESC LIMIT 10"
        ).fetchall()

        return {
            "total_events": total_events,
            "total_egl": total_egl,
            "by_type": by_type,
            "recent_events": [self._row_to_dict(r) for r in recent],
        }

    # ── Memory Stats ───────────────────────────────────────────────────────

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        stats: Dict[str, Any] = {}

        # Per-level stats
        for level in ("L1", "L2", "L3"):
            row = self._conn.execute(
                f"SELECT COUNT(*) as cnt, COALESCE(SUM(token_count), 0) as tokens, "
                f"COALESCE(AVG(weight), 0) as avg_weight "
                f"FROM memories WHERE level = ?",
                (level,),
            ).fetchone()
            stats[f"{level}_count"] = row["cnt"]
            stats[f"{level}_tokens"] = row["tokens"]
            stats[f"{level}_avg_weight"] = round(row["avg_weight"], 3)

        # Frozen items
        frozen = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE level = 'L3' AND weight <= ?",
            (L3_FROZEN_WEIGHT,),
        ).fetchone()["cnt"]
        stats["frozen_l3_count"] = frozen

        # Tools
        tool_count = self._conn.execute("SELECT COUNT(*) as cnt FROM tools").fetchone()["cnt"]
        stats["tool_count"] = tool_count

        # Interactions
        interaction_count = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM interactions"
        ).fetchone()["cnt"]
        stats["interaction_count"] = interaction_count

        # Evolution
        evo = self.get_evolution_stats()
        stats["evolution_total_events"] = evo["total_events"]
        stats["evolution_total_egl"] = evo["total_egl"]

        return stats

    # ── Helpers ────────────────────────────────────────────────────────────

    def _touch_memory(self, mem_id: str) -> None:
        """Update access timestamp and increment access count."""
        now = time.time()
        self._conn.execute(
            "UPDATE memories SET accessed_at = ?, access_count = access_count + 1 "
            "WHERE id = ?",
            (now, mem_id),
        )
        self._conn.commit()

    @staticmethod
    def _generate_index_key(key: str, content: str) -> str:
        """Generate a reversible index key for compressed memories."""
        raw = f"{key}:{content[:200]}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a sqlite3.Row to a dictionary."""
        d = dict(row)
        # Parse JSON fields
        for json_field in ("metadata", "before_state", "after_state", "tools_required", "steps"):
            if json_field in d and isinstance(d[json_field], str):
                try:
                    d[json_field] = json.loads(d[json_field])
                except json.JSONDecodeError:
                    pass
        return d

    def close(self) -> None:
        """Close database connections."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()
