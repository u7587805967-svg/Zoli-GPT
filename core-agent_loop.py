from pydantic import BaseModel, Field
from typing import Optional

class AgentThought(BaseModel):
    thought: str = Field(description="A probléma lépésről lépésre történő elemzése")
    action: str = Field(description="Választott eszköz vagy válaszadási útvonal")
    response: str = Field(description="A felhasználónak szánt végleges válasz")

class CritiqueResult(BaseModel):
    is_valid: bool = Field(description="Megfelel-e a válasz a logikai és ténybeli követelményeknek?")
    feedback: Optional[str] = Field(description="Hiányosságok vagy kijavítandó pontok")

def run_reasoning_loop(user_input: str, llm_client) -> str:
    structured_llm = llm_client.with_structured_output(AgentThought)
    critique_llm = llm_client.with_structured_output(CritiqueResult)
    
    draft = structured_llm.invoke(f"Feladat: {user_input}")
    
    critique = critique_llm.invoke(f"Ellenőrizd a logikát és tényeket: {draft.response}")
    if not critique.is_valid:
        draft = structured_llm.invoke(
            f"Javítsd a következők alapján: {critique.feedback}. Eredeti válasz: {draft.response}"
        )
        
    return draft.response