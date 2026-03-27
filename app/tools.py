import random
import time
from typing import List, Dict, Any, Callable, Tuple
from threading import Lock
from langchain_core.tools import StructuredTool
from neo4j.exceptions import ServiceUnavailable, SessionExpired, Neo4jError

from app.database import neo4j_client, get_node_types_from_mysql
from app.config import settings

_tools_cache_lock = Lock()
_tools_cache = {"ts": 0.0, "tools": None, "label_tool_map": None}

_nodes_cache_lock = Lock()
_nodes_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}


def get_nodes_by_label(label: str, limit: int = 1000) -> List[Dict[str, Any]]:
    """Generic function to get nodes by label from Neo4j (retry up to 3 times on transient failures)."""
    ttl = settings.NODE_LIST_CACHE_TTL_SECONDS
    cache_key = (label, int(limit or 0))
    if ttl > 0:
        with _nodes_cache_lock:
            cached = _nodes_cache.get(cache_key)
            if cached and (time.time() - cached["ts"] < ttl):
                return cached["data"]

    query = f"MATCH (n:{label}) RETURN n.name as name LIMIT $limit"

    max_attempts = 3
    base_delay = 0.3  # seconds

    def _is_retryable(err: Exception) -> bool:
        # Neo4j driver specific transient cases
        if isinstance(err, (ServiceUnavailable, SessionExpired)):
            return True
        # Neo4jError often contains classification/code
        if isinstance(err, Neo4jError):
            # Many transient errors are classified as "TransientError"
            code = getattr(err, "code", "") or ""
            return "TransientError" in code
        return False

    last_err: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            with neo4j_client.get_session() as session:
                result = session.run(query, limit=limit)
                data = [{"name": record["name"]} for record in result if record.get("name")]
                if ttl > 0:
                    with _nodes_cache_lock:
                        _nodes_cache[cache_key] = {"ts": time.time(), "data": data}
                return data
        except Exception as e:
            last_err = e
            # 不可重试：直接失败
            if not _is_retryable(e) or attempt == max_attempts:
                print(f"Error fetching nodes for label {label} (attempt {attempt}/{max_attempts}): {e}")
                break

            # 可重试：指数退避 + 抖动
            delay = base_delay * (2 ** (attempt - 1))
            delay += random.uniform(0, 0.2)  # jitter
            print(
                f"Retryable error fetching nodes for label {label} "
                f"(attempt {attempt}/{max_attempts}): {e}. Retrying in {delay:.2f}s..."
            )
            time.sleep(delay)
    # 全部尝试失败
    return []


def create_tool_function(label: str, display_name: str) -> Callable:
    """Create a tool function for a specific label"""

    def tool_func(limit: int = 1000) -> Dict[str, Any]:
        nodes = get_nodes_by_label(label, limit)
        return {"nodes": nodes}

    tool_func.__name__ = f"get_all_{label.lower()}_nodes"
    tool_func.__doc__ = f"Get all {display_name} nodes ({label}) from the database."
    return tool_func


def generate_dynamic_tools():
    """Generate tools and label map from database"""
    ttl = settings.TOOLS_CACHE_TTL_SECONDS
    if ttl > 0:
        with _tools_cache_lock:
            cached = _tools_cache
            if cached.get("tools") and (time.time() - float(cached.get("ts", 0.0)) < ttl):
                return cached["tools"], cached["label_tool_map"]

    node_types = get_node_types_from_mysql()

    label_tool_map = []
    tools = []

    for nt in node_types:
        label = nt.get("type_name")
        display_name = nt.get("display_name")

        if not label:
            continue

        tool_func = create_tool_function(label, display_name)
        tool_name = tool_func.__name__

        # Create LangChain StructuredTool
        structured_tool = StructuredTool.from_function(
            func=tool_func,
            name=tool_name,
            description=tool_func.__doc__
        )

        tools.append(structured_tool)
        label_tool_map.append({
            "label": label,
            "nodeType": display_name,
            "tool": tool_name
        })

    if ttl > 0:
        with _tools_cache_lock:
            _tools_cache["tools"] = tools
            _tools_cache["label_tool_map"] = label_tool_map
            _tools_cache["ts"] = time.time()

    return tools, label_tool_map
