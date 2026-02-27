from typing import Dict, List, Union

from pydantic import BaseModel, Field, field_validator


class ErrorResponse(BaseModel):
    error: str = Field(..., json_schema_extra={"example": "INTERNAL_SERVER_ERROR"})
    detail: str = Field(
        ..., json_schema_extra={"example": "A detailed description of the error."}
    )
    status_code: int = Field(..., json_schema_extra={"example": 500})


class AnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        description="Unstructured text to analyze",
        json_schema_extra={"example": "The revenue grew by 20% in Q3 2023..."},
    )
    categories: List[str] = Field(
        ...,
        description="List of categories for zero-shot classification",
        json_schema_extra={"example": ["Finance", "Strategy", "Operations"]},
    )

    @field_validator("text")
    @classmethod
    def validate_text_length(cls, v: str) -> str:
        if not (100 <= len(v) <= 50000):
            raise ValueError("Text length must be between 100 and 50,000 characters")
        return v


class CategoryResult(BaseModel):
    label: str = Field(..., json_schema_extra={"example": "Finance"})
    score: float = Field(..., json_schema_extra={"example": 95.5})
    evidence: str = Field(
        ...,
        json_schema_extra={
            "example": "Revenue growth indicates strong financial performance."
        },
    )
    confidence_level: str = "Medium"


class NumericalInsight(BaseModel):
    value: str = Field(..., json_schema_extra={"example": "20%"})
    context: str = Field(..., json_schema_extra={"example": "Revenue growth"})
    confidence: float = Field(..., json_schema_extra={"example": 0.98})


class AnalyzeResponse(BaseModel):
    intent: str = Field(..., json_schema_extra={"example": "CLASSIFICATION"})
    primary_domain: str = Field(..., json_schema_extra={"example": "Finance"})
    secondary_domains: List[str] = Field(
        ..., json_schema_extra={"example": ["Strategy"]}
    )
    excluded_domains: List[str] = Field(..., json_schema_extra={"example": ["Sports"]})
    reasoning: str = Field(
        ..., json_schema_extra={"example": "Text contains strong financial markers."}
    )
    answer: str = Field(
        None,
        json_schema_extra={
            "example": "The document is primarily classified as Finance."
        },
    )
    evidence_quotes: List[str] = Field(
        ..., json_schema_extra={"example": ["...revenue grew by 20%..."]}
    )
    numerical_insights: List[NumericalInsight] = []
    xai_attributions: List[List[Union[str, float]]] = Field(
        None,
        description="Optional token-level attributions for low-confidence results.",
    )


class ChatRequest(BaseModel):
    context: str = Field(..., description="The document text to chat with")
    question: str = Field(..., description="The user's question about the document")


class ChatResponse(BaseModel):
    intent: str = Field(..., json_schema_extra={"example": "FACT"})
    answer: str = Field(None, json_schema_extra={"example": "The revenue grew by 20%."})
    evidence: Union[str, List[str]] = Field(
        None, json_schema_extra={"example": '"...revenue grew by 20%..."'}
    )
    function: str = None
    reasoning: str = None
    conclusion: str = None
    limitations: str = None


class ExplainRequest(BaseModel):
    text: str = Field(..., description="Text to explain")
    target_label: str = Field(
        None, description="Optional category label to explain (for Zero-Shot)"
    )


class ExplainResponse(BaseModel):
    predicted_label: str = Field(..., json_schema_extra={"example": "POSITIVE"})
    attributions: List[List[Union[str, float]]]  # Tuple [token, score]


class FeedbackRequest(BaseModel):
    document_hash: str = Field(..., json_schema_extra={"example": "a1b2c3d4..."})
    original_categories: List[Dict[str, Union[str, float]]]
    edited_categories: List[str]
    timestamp: str
