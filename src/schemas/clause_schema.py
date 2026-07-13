from pydantic import BaseModel


class ClauseExtraction(BaseModel):
    """
    Structured output returned by the LLM.
    """

    termination_clause: str = ""
    confidentiality_clause: str = ""
    liability_clause: str = ""