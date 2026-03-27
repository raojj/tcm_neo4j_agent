import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")

    CANDIDATE_LIMIT_DEFAULT = int(os.getenv("CANDIDATE_LIMIT_DEFAULT", "120"))
    CANDIDATE_LIMIT_MAX = int(os.getenv("CANDIDATE_LIMIT_MAX", "200"))

    MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DB = os.getenv("MYSQL_DB", "sci_manage_system")

    REL_TYPE_TABLE = os.getenv("REL_TYPE_TABLE", "neo4j_relationship_type")
    REL_TYPE_NAME_COL = os.getenv("REL_TYPE_NAME_COL", "type_name")
    REL_TYPE_DISPLAY_COL = os.getenv("REL_TYPE_DISPLAY_COL", "display_name")
    REL_TYPE_DESC_COL = os.getenv("REL_TYPE_DESC_COL", "description")

    NODE_TYPE_TABLE = os.getenv("NODE_TYPE_TABLE", "neo4j_node_type")
    NODE_TYPE_NAME_COL = os.getenv("NODE_TYPE_NAME_COL", "type_name")
    NODE_TYPE_DISPLAY_COL = os.getenv("NODE_TYPE_DISPLAY_COL", "display_name")

    # Syndrome recommendation (MySQL)
    SYNDROME_TABLE = os.getenv("SYNDROME_TABLE", "originaldata_syndrome")
    SYNDROME_NAME_COL = os.getenv("SYNDROME_NAME_COL", "name")
    SYNDROME_INTRO_COL = os.getenv("SYNDROME_INTRO_COL", "introduction")
    SYNDROME_CANDIDATE_LIMIT_DEFAULT = int(os.getenv("SYNDROME_CANDIDATE_LIMIT_DEFAULT", "2000"))
    SYNDROME_CANDIDATE_LIMIT_MAX = int(os.getenv("SYNDROME_CANDIDATE_LIMIT_MAX", "5000"))

    # Symptom recommendation (Neo4j)
    SYMPTOM_LABEL = os.getenv("SYMPTOM_LABEL", "SYMPTOM")
    SYMPTOM_CANDIDATE_LIMIT_DEFAULT = int(os.getenv("SYMPTOM_CANDIDATE_LIMIT_DEFAULT", "2000"))
    SYMPTOM_CANDIDATE_LIMIT_MAX = int(os.getenv("SYMPTOM_CANDIDATE_LIMIT_MAX", "5000"))
    SYMPTOM_MATCH_MAX_RETURN = int(os.getenv("SYMPTOM_MATCH_MAX_RETURN", "50"))

    # Caching (seconds). 0 disables cache.
    REL_TYPES_CACHE_TTL_SECONDS = int(os.getenv("REL_TYPES_CACHE_TTL_SECONDS", "300"))
    NODE_TYPES_CACHE_TTL_SECONDS = int(os.getenv("NODE_TYPES_CACHE_TTL_SECONDS", "300"))
    TOOLS_CACHE_TTL_SECONDS = int(os.getenv("TOOLS_CACHE_TTL_SECONDS", "300"))
    NODE_LIST_CACHE_TTL_SECONDS = int(os.getenv("NODE_LIST_CACHE_TTL_SECONDS", "300"))
    SYNDROME_CANDIDATE_CACHE_TTL_SECONDS = int(os.getenv("SYNDROME_CANDIDATE_CACHE_TTL_SECONDS", "300"))
    AGENT_CACHE_TTL_SECONDS = int(os.getenv("AGENT_CACHE_TTL_SECONDS", "300"))

    ENABLE_EXTRA_RELATIONS = os.getenv("ENABLE_EXTRA_RELATIONS", "true").strip().lower() in ("1", "true", "yes")

settings = Settings()
