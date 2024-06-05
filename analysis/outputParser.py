from typing import List, Dict

from pydantic import BaseModel, Field


class StatementParser(BaseModel):
    opening_statements: List[str] = Field(description="list of different Opening statements")
    questioning_statements: List[str] = Field(description="list of different Questioning statements")
    presenting_statements: List[str] = Field(description="list of different Presenting statements")
    closing_outcome_sentences: List[Dict[str, str]] = Field(
        description="list of dictionaries containing closing statement of REP ('closing_statement') and outcome statements OF HCP ('outcome_statement')")


class StatementParserWithoutOpening(BaseModel):
    questioning_statements: List[str] = Field(description="list of different Questioning statements")
    presenting_statements: List[str] = Field(description="list of different Presenting statements")
    closing_outcome_sentences: List[Dict[str, str]] = Field(
        description="list of dictionaries containing closing and outcome statements. Example: {'closing_statement': closing_dailouge (REP), 'outcome_statement': outcome_dailouge(HCP)}")


class StatementParserWithoutClosing(BaseModel):
    opening_statements: List[str] = Field(description="list of different Opening statements")
    questioning_statements: List[str] = Field(description="list of different Questioning statements")
    presenting_statements: List[str] = Field(description="list of different Presenting statements")


class StatementParserWithoutOpeningAndClosing(BaseModel):
    questioning_statements: List[str] = Field(description="list of different Questioning statements")
    presenting_statements: List[str] = Field(description="list of different Presenting statements")


class StatementLevelModel(BaseModel):
    id: int = Field(description="this field is the ID of the statement")
    level: int = Field(description="this field is the assigned level (1-4)")
    confidence_score: int = Field(description="this field is your confidence score, indicating how certain you are about your evaluation.")
    reason: int = Field(description="reason for you evaluation in max 1 line")


class EvaluationResult(BaseModel):
    statements: List[StatementLevelModel] = Field(..., description="A list of statement level Model")
