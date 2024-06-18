from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field
from typing_extensions import TypedDict


# GRADERS
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents"""
    binary_score: str = Field(
        description = "Are documents relevant to the question? 'yes' or 'no'"
    )
    
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer"""
    binary_score: str = Field(
        description = "Is answer grounded in the facts? 'yes' or 'no'"
    )
    
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question"""
    binary_score: str = Field(
        description = "Does answer address the question? 'yes' or 'no'"
    )
    
# GRAPH STATE
class GraphState(TypedDict):
    """
    Represents the graph's state.
    
    Attributes:
        question: Question
        generation: LLM's generation
        documents: List of documents
    """
    question: str
    generation: str
    documents: List[str]
