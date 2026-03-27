import pymysql
from neo4j import GraphDatabase
from app.config import settings
from app.models import RelationshipTypeSchema
from typing import List, Dict, Any
import logging
import time
from threading import Lock

logger = logging.getLogger("uvicorn.error")


class Neo4jClient:
    def __init__(self):
        self.driver = None

    def connect(self):
        if not self.driver:
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            print("✅ Neo4j connected successfully.")

    def close(self):
        if self.driver:
            self.driver.close()
            print("🛑 Neo4j connection closed.")
            self.driver = None

    def get_session(self):
        if not self.driver:
            self.connect()
        return self.driver.session()


neo4j_client = Neo4jClient()

_cache_lock = Lock()
_rel_types_cache = {"ts": 0.0, "data": None}
_node_types_cache = {"ts": 0.0, "data": None}
_syndrome_candidates_cache = {"ts": 0.0, "data": []}


def _cache_valid(cache: Dict[str, Any], ttl_seconds: int) -> bool:
    if ttl_seconds <= 0:
        return False
    data = cache.get("data")
    if data is None:
        return False
    return (time.time() - float(cache.get("ts", 0.0))) < ttl_seconds


def get_mysql_conn(dict_cursor: bool = True):
    cursorclass = pymysql.cursors.DictCursor if dict_cursor else None
    return pymysql.connect(
        host=settings.MYSQL_HOST,
        port=settings.MYSQL_PORT,
        user=settings.MYSQL_USER,
        password=settings.MYSQL_PASSWORD,
        database=settings.MYSQL_DB,
        cursorclass=cursorclass,
        connect_timeout=5,
    )


def get_relationship_types_from_mysql() -> List[Dict[str, Any]]:
    """Fetch relationship types from MySQL"""
    ttl = settings.REL_TYPES_CACHE_TTL_SECONDS
    if _cache_valid(_rel_types_cache, ttl):
        return _rel_types_cache["data"]

    logger.info("Connecting to MySQL...")
    try:
        conn = get_mysql_conn(dict_cursor=True)
        logger.info("Connected to MySQL.")
    except Exception as e:
        logger.error(f"Failed to connect to MySQL: {e}")
        return []

    try:
        with conn.cursor() as cursor:
            sql = f"""
                SELECT 
                    {settings.REL_TYPE_NAME_COL} as type_name,
                    {settings.REL_TYPE_DISPLAY_COL} as display_name,
                    {settings.REL_TYPE_DESC_COL} as description
                FROM {settings.REL_TYPE_TABLE}
            """
            logger.info(f"Executing query: {sql}")
            cursor.execute(sql)
            results = cursor.fetchall()
            if ttl > 0:
                with _cache_lock:
                    _rel_types_cache["data"] = results
                    _rel_types_cache["ts"] = time.time()
            return results
    except Exception as e:
        logger.error(f"Error fetching relationship types: {e}")
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_node_types_from_mysql() -> List[Dict[str, Any]]:
    """Fetch node types from MySQL"""
    ttl = settings.NODE_TYPES_CACHE_TTL_SECONDS
    if _cache_valid(_node_types_cache, ttl):
        return _node_types_cache["data"]

    logger.info("Connecting to MySQL to fetch node types...")
    try:
        conn = get_mysql_conn(dict_cursor=True)
    except Exception as e:
        logger.error(f"Failed to connect to MySQL: {e}")
        return []

    try:
        with conn.cursor() as cursor:
            sql = f"""
                SELECT 
                    {settings.NODE_TYPE_NAME_COL} as type_name,
                    {settings.NODE_TYPE_DISPLAY_COL} as display_name
                FROM {settings.NODE_TYPE_TABLE}
            """
            logger.info(f"Executing query: {sql}")
            cursor.execute(sql)
            results = cursor.fetchall()
            if ttl > 0:
                with _cache_lock:
                    _node_types_cache["data"] = results
                    _node_types_cache["ts"] = time.time()
            return results
    except Exception as e:
        logger.error(f"Error fetching node types: {e}")
        return []
    finally:
        conn.close()


def get_syndrome_name_candidates(limit: int) -> List[str]:
    """
    Fetch syndrome names from MySQL `originaldata_syndrome` (or configured table).
    Returns a de-duplicated list of non-empty names.
    """
    limit = int(limit or 0)
    if limit <= 0:
        limit = settings.SYNDROME_CANDIDATE_LIMIT_DEFAULT
    limit = max(100, min(limit, settings.SYNDROME_CANDIDATE_LIMIT_MAX))

    ttl = settings.SYNDROME_CANDIDATE_CACHE_TTL_SECONDS
    if ttl > 0:
        with _cache_lock:
            if _cache_valid(_syndrome_candidates_cache, ttl):
                cached = _syndrome_candidates_cache.get("data") or []
                if len(cached) >= limit:
                    return cached[:limit]

    try:
        conn = get_mysql_conn(dict_cursor=True)
    except Exception as e:
        logger.error(f"Failed to connect to MySQL (syndrome candidates): {e}")
        return []

    try:
        with conn.cursor() as cursor:
            sql = f"""
                SELECT {settings.SYNDROME_NAME_COL} AS name
                FROM {settings.SYNDROME_TABLE}
                WHERE {settings.SYNDROME_NAME_COL} IS NOT NULL
                  AND TRIM({settings.SYNDROME_NAME_COL}) <> ''
                LIMIT %s
            """
            cursor.execute(sql, (limit,))
            rows = cursor.fetchall() or []
            out: List[str] = []
            seen = set()
            for r in rows:
                name = (r.get("name") or "").strip()
                if not name or name in seen:
                    continue
                seen.add(name)
                out.append(name)
            if ttl > 0:
                with _cache_lock:
                    _syndrome_candidates_cache["data"] = out
                    _syndrome_candidates_cache["ts"] = time.time()
            return out
    except Exception as e:
        logger.error(f"Error fetching syndrome name candidates: {e}")
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


def get_syndrome_introductions_by_names(names: List[str]) -> List[Dict[str, str]]:
    """
    Batch fetch syndrome introductions by exact names.
    Returns list of {name, introduction}.
    """
    if not names:
        return []
    cleaned: List[str] = []
    seen = set()
    for n in names:
        nn = (str(n or "")).strip()
        if nn and nn not in seen:
            seen.add(nn)
            cleaned.append(nn)
    if not cleaned:
        return []

    try:
        conn = get_mysql_conn(dict_cursor=True)
    except Exception as e:
        logger.error(f"Failed to connect to MySQL (syndrome introductions): {e}")
        return []

    try:
        with conn.cursor() as cursor:
            placeholders = ",".join(["%s"] * len(cleaned))
            sql = f"""
                SELECT
                    {settings.SYNDROME_NAME_COL} AS name,
                    {settings.SYNDROME_INTRO_COL} AS introduction
                FROM {settings.SYNDROME_TABLE}
                WHERE {settings.SYNDROME_NAME_COL} IN ({placeholders})
            """
            cursor.execute(sql, tuple(cleaned))
            rows = cursor.fetchall() or []
            out: List[Dict[str, str]] = []
            for r in rows:
                name = (r.get("name") or "").strip()
                intro = (r.get("introduction") or "").strip()
                if not name:
                    continue
                out.append({"name": name, "introduction": intro})
            return out
    except Exception as e:
        logger.error(f"Error fetching syndrome introductions: {e}")
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass
