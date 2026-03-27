from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.database import neo4j_client
from app.models import ParseReq, ParseResp, RecommendSyndromeReq, RecommendSyndromeResp
from app.service import parse_text_with_langchain, recommend_syndromes_from_mysql
from app.config import settings
import logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        neo4j_client.connect()
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")

    yield

    # Shutdown
    neo4j_client.close()


app = FastAPI(title="TCM Neo4j Agent", lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "TCM Neo4j Agent is running"}


logger = logging.getLogger("uvicorn.error")


@app.post("/test-post")
def test_post(data: dict):
    logger.info(f"Received test-post: {data}")
    return {"received": data}


@app.post("/nlp/parse", response_model=ParseResp)
def parse(req: ParseReq):
    logger.info(f"Received parse request: {req}")
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text不能为空")

    limit = req.limit or settings.CANDIDATE_LIMIT_DEFAULT
    limit = max(10, min(int(limit), settings.CANDIDATE_LIMIT_MAX))

    try:
        return parse_text_with_langchain(text, limit=limit)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"parse failed: {e}")


@app.post("/tcm/syndrome/recommend", response_model=RecommendSyndromeResp)
def recommend_syndrome(req: RecommendSyndromeReq):
    logger.info(f"Received syndrome recommend request: {req}")
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text不能为空")

    try:
        names = recommend_syndromes_from_mysql(text, candidate_limit=req.candidateLimit)
        print(names)
        return RecommendSyndromeResp(syndromeNames=names)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"recommend failed: {e}")
