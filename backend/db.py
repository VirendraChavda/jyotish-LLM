# backend/db.py
import os
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv
from auth import hash_password

import sys
from pathlib import Path
from datetime import datetime
from typing import Iterable, Dict

from datetime import datetime, date, timezone
from decimal import Decimal
import numpy as np

load_dotenv()
mongo_uri = os.getenv("mongo_uri")
uri = mongo_uri

# Create a new client and connect to the server
client = MongoClient(uri)

auth_db = client["auth_db"]
astro_db = client["astro_db"]

users_col = auth_db["users"]                 # credentials only
profiles_col = astro_db["users_profile"]     # birth-dependent identity
sessions_col = astro_db["sessions"]          # current time/location (no dynamic charts)

# indexes
users_col.create_index([("username", ASCENDING)], unique=True)
profiles_col.create_index([("username", ASCENDING)], unique=True)
sessions_col.create_index([("username", ASCENDING)], unique=True)

def mongo_sanitize(obj):
    """Recursively make data BSON-safe for MongoDB."""
    if isinstance(obj, dict):
        return {str(k): mongo_sanitize(v) for k, v in obj.items()}  # keys -> str
    if isinstance(obj, (list, tuple, set)):
        return [mongo_sanitize(v) for v in obj]

    # --- dates & datetimes ---
    if isinstance(obj, date) and not isinstance(obj, datetime):
        # store as UTC midnight so it's queryable with $gte/$lte
        return datetime(obj.year, obj.month, obj.day, tzinfo=timezone.utc)
    if isinstance(obj, datetime):
        # ensure tz-aware UTC (Mongo stores UTC)
        return obj if obj.tzinfo else obj.replace(tzinfo=timezone.utc)

    # --- numerics & arrays often coming from numpy/Decimal ---
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return obj

# ----------------------------------ProsgreBD functions--------------------------------------
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine

from pgvector.psycopg2 import register_vector
from pgvector import Vector as PgVector

DATABASE_URL = os.environ.get("DATABASE_URL")
engine: Engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)

# Register pgvector BEFORE opening any connections
@event.listens_for(engine, "connect")
def _on_connect(dbapi_conn, conn_record):
    register_vector(dbapi_conn)  # enables pgvector param adaptation
    try:
        cur = dbapi_conn.cursor()
        # Tune ANN search; change to "SET hnsw.ef_search = 40;" if you use HNSW
        cur.execute("SET ivfflat.probes = 10;")
        cur.close()
    except Exception:
        pass

# ----------------- Ensure Schema ----------
def ensure_schema():
    """Optional helper to create episodic table & topic column if missing."""
    with engine.begin() as conn:
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        except Exception:
            pass
        conn.execute(text("""
            ALTER TABLE IF EXISTS memory_entities
            ADD COLUMN IF NOT EXISTS topic TEXT;
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS memory_episodes (
              id BIGSERIAL PRIMARY KEY,
              user_id TEXT NOT NULL,
              asked_at TIMESTAMPTZ DEFAULT now(),
              question TEXT NOT NULL,
              answer_digest TEXT NOT NULL,
              topic TEXT,
              embedding vector(1536)
            );
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_mem_episodes_user_time
              ON memory_episodes (user_id, asked_at DESC);
        """))

from sqlalchemy import text
def ensure_full_memory_schema():
    ensure_schema()  # creates memory_episodes + adds topic on entities if needed
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS memory_entities (
              user_id TEXT NOT NULL,
              type TEXT NOT NULL,
              key TEXT NOT NULL,
              value TEXT,
              topic TEXT,
              last_seen_at TIMESTAMPTZ DEFAULT now(),
              PRIMARY KEY (user_id, type, key)
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS memory_notes (
              user_id TEXT NOT NULL,
              note TEXT NOT NULL,
              embedding vector(1536),
              last_seen_at TIMESTAMPTZ DEFAULT now(),
              PRIMARY KEY (user_id, note)
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS memory_summary (
              user_id TEXT PRIMARY KEY,
              text TEXT NOT NULL,
              updated_at TIMESTAMPTZ DEFAULT now()
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_details (
              user_id TEXT PRIMARY KEY,
              birth_date DATE,
              birth_time TIME,
              birth_place TEXT,
              current_place TEXT,
              created_at TIMESTAMPTZ DEFAULT now()
            );
        """))

# ------------------------------------User details------------------------------------------
from typing import Optional, Any
def query_user_details(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Load all details for a user from the user_details table.
    Returns a dict with columns or None if not found.
    """
    with engine.begin() as conn:
        row = conn.execute(
            text("""
                SELECT user_id, birth_date, birth_time, birth_place, current_place, created_at
                FROM user_details
                WHERE user_id = :uid
                LIMIT 1
            """),
            {"uid": user_id},
        ).fetchone()
    return dict(row._mapping) if row else None

def update_user_details(user_id: str, birth_date_iso: str | None = None, birth_time_str: str | None = None,
                        birth_place: str | None = None, current_place: str | None = None):
    sets = []
    params = {"uid": user_id}
    if birth_date_iso is not None:
        sets.append("birth_date = CAST(:bd AS DATE)")
        params["bd"] = birth_date_iso
    if birth_time_str is not None:
        sets.append("birth_time = CAST(:bt AS TIME)")
        params["bt"] = birth_time_str
    if birth_place is not None:
        sets.append("birth_place = :bp")
        params["bp"] = birth_place
    if current_place is not None:
        sets.append("current_place = :cp")
        params["cp"] = current_place
    if not sets:
        return
    sql = f"UPDATE user_details SET {', '.join(sets)}, created_at = now() WHERE user_id = :uid"
    with engine.begin() as conn:
        conn.execute(text(sql), params)
        
def upsert_user_details(user_id: str, birth_date_iso: str, birth_time_str: str,
                        birth_place: str, current_place: str):
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO user_details
                    (user_id, birth_date, birth_time, birth_place, current_place, created_at)
                VALUES
                    (:uid, CAST(:bd AS DATE), CAST(:bt AS TIME), :bp, :cp, now())
                ON CONFLICT (user_id)
                DO UPDATE SET
                    birth_date    = EXCLUDED.birth_date,
                    birth_time    = EXCLUDED.birth_time,
                    birth_place   = EXCLUDED.birth_place,
                    current_place = EXCLUDED.current_place,
                    created_at    = now()
            """),
            {"uid": user_id, "bd": birth_date_iso, "bt": birth_time_str, "bp": birth_place, "cp": current_place},
        )
                
# ------------------------------------Summarization------------------------------------------
def load_summary(user_id: str) -> str:
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT text FROM memory_summary WHERE user_id = :uid"),
            {"uid": user_id},
        ).fetchone()
    return row[0] if row else ""

def save_summary(user_id: str, text_value: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO memory_summary (user_id, text, updated_at)
                VALUES (:uid, :text, now())
                ON CONFLICT (user_id)
                DO UPDATE SET text = EXCLUDED.text, updated_at = now()
            """),
            {"uid": user_id, "text": text_value},
        )

# ------------------------------------Entity extraction------------------------------------------
from typing import List, Dict, Any, Tuple, Optional, TypedDict
def select_relevant_entities(user_id: str, topic: Optional[str] = None, limit: int = 4) -> List[str]:
    params = {"uid": user_id, "lim": limit}
    where = "user_id = :uid"
    if topic:
        params["topic"] = f"%{topic.lower()}%"
        where += " AND (lower(type) = :t OR lower(key) LIKE :topic OR lower(value) LIKE :topic)"
        params["t"] = topic.lower()

    sql = f"""
        SELECT type, key, value
        FROM memory_entities
        WHERE {where}
        ORDER BY last_seen_at DESC
        LIMIT :lim
    """
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).fetchall()
    return [f"{k}: {v}" if k and v else f"{t}={v or k}" for t, k, v in rows]

def upsert_entities(user_id: str, ents: List[dict]) -> None:
    with engine.begin() as conn:
        for e in ents:
            conn.execute(text("""
              INSERT INTO memory_entities (user_id, type, key, value, topic, last_seen_at)
              VALUES (:uid, :type, :key, :value, :topic, now())
              ON CONFLICT (user_id, type, key)
              DO UPDATE SET value=EXCLUDED.value, topic=COALESCE(EXCLUDED.topic, memory_entities.topic), last_seen_at=now()
            """), {
              "uid": user_id,
              "type": (e.get("type") or "").strip()[:50],
              "key":  (e.get("key") or "").strip()[:100],
              "value": (e.get("value") or "").strip()[:500],
              "topic": (e.get("topic") or "").strip()[:100]
            })

# ----------------- Memory: notes (pgvector) --
def upsert_semantic_notes(memory_embedder, user_id: str, notes: List[str]) -> None:
    if not notes:
        return
    vecs = memory_embedder.embed_documents(notes)  # -> List[List[float]]
    with engine.begin() as conn:
        for note, vec in zip(notes, vecs):
            conn.execute(
                text("""
                    INSERT INTO memory_notes (user_id, note, embedding, last_seen_at)
                    VALUES (:uid, :note, :emb, now())
                    ON CONFLICT (user_id, note)
                    DO UPDATE SET embedding = EXCLUDED.embedding, last_seen_at = now()
                """),
                {"uid": user_id, "note": note, "emb": PgVector(vec)},
            )

def search_semantic_notes(memory_embedder, user_id: str, query: str, k: int = 4) -> List[str]:
    qvec = PgVector(memory_embedder.embed_query(query))
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT note
                FROM memory_notes
                WHERE user_id = :uid
                ORDER BY embedding <=> :qvec
                LIMIT :k
            """),
            {"uid": user_id, "qvec": qvec, "k": k},
        ).fetchall()
    return [r[0] for r in rows]

# ----------------- Memory: episodes ----------
import re
def make_digest(answer: str, notes: List[str]) -> str:
    if notes:
        return " | ".join(notes)[:400]
    return re.sub(r"\s+", " ", answer).strip()[:400]

def upsert_episode(memory_embedder, user_id: str, question: str, answer: str, topic: Optional[str], notes: List[str]) -> None:
    digest = make_digest(answer, notes)
    embed_text = f"Q: {question}\nA: {digest}"
    vec = PgVector(memory_embedder.embed_query(embed_text))
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO memory_episodes (user_id, question, answer_digest, topic, embedding)
            VALUES (:uid, :q, :d, :topic, :emb)
        """), {"uid": user_id, "q": question, "d": digest, "topic": topic, "emb": vec})

def episode_query_vector(memory_embedder, user_id: str, question: str) -> PgVector:
    ctx = (load_summary(user_id) or "")[:200]
    q_aug = f"{question}\nContext: {ctx}"
    return PgVector(memory_embedder.embed_query(q_aug))

def recall_episodes(memory_embedder, user_id: str, question: str, k: int = 3, dist_cutoff: float = 0.35) -> List[dict]:
    qvec = episode_query_vector(memory_embedder, user_id, question)
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT asked_at, question, answer_digest, (embedding <=> :qvec) AS dist
            FROM memory_episodes
            WHERE user_id = :uid
            ORDER BY dist
            LIMIT :k
        """), {"uid": user_id, "qvec": qvec, "k": k}).fetchall()
    return [
        {"asked_at": r[0], "question": r[1], "digest": r[2], "dist": float(r[3])}
        for r in rows if r[3] is not None and float(r[3]) <= dist_cutoff
    ]

def recent_episodes(user_id: str, n: int = 3) -> List[dict]:
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT asked_at, question, answer_digest
            FROM memory_episodes
            WHERE user_id = :uid
            ORDER BY asked_at DESC
            LIMIT :n
        """), {"uid": user_id, "n": n}).fetchall()
    return [{"asked_at": r[0], "question": r[1], "digest": r[2]} for r in rows]




# Optional: quick connectivity check
if __name__ == "__main__":
    with engine.connect() as conn:
        print("Connected to:", conn.execute(text("SELECT version()")).scalar())
