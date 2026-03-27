from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict


class ParseReq(BaseModel):
    text: str = Field(..., description="User input text to parse")
    limit: Optional[int] = Field(None, description="Limit for candidate nodes fetching")


class NodeItem(BaseModel):
    name: str
    nodeType: str


class RelationshipItem(BaseModel):
    sourceNodeName: str
    targetNodeName: str
    relationshipType: str


class ParseResp(BaseModel):
    newNodes: List[NodeItem]
    nodes: List[NodeItem]
    relationships: List[RelationshipItem]


class RelationshipTypeSchema(BaseModel):
    type_name: str
    display_name: str
    description: Optional[str] = None


class RecommendSyndromeReq(BaseModel):
    text: str = Field(..., description="用户症状描述")
    candidateLimit: Optional[int] = Field(None, description="从MySQL拉取的证型名称候选数量上限")


class RecommendSyndromeResp(BaseModel):
    syndromeNames: List[str] = Field(..., description="最相关的5个证型名称列表")
