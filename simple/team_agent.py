from pydantic import BaseModel
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.team import Team
from agno.tools.yfinance import YFinanceTools
import uuid
import hashlib

def create_uuid_from_string(val: str):
    hex_string = hashlib.md5(val.encode("UTF-8")).hexdigest()
    return str(uuid.UUID(hex=hex_string))

class StockAnalysis(BaseModel):
    symbol: str
    company_name: str
    analysis: str

class CompanyAnalysis(BaseModel):
    company_name: str
    analysis: str

stock_searcher = Agent(
    name="Stock Searcher",
    model=Ollama(id="phi4-mini"),
    response_model=StockAnalysis,
    role="Searches for information on stocks and provides price analysis.",
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
        )
    ],
)

company_info_agent = Agent(
    name="Company Info Searcher",
    model=Ollama(id="phi4-mini"),
    role="Searches for information about companies and recent news.",
    response_model=CompanyAnalysis,
    tools=[
        YFinanceTools(
            stock_price=False,
            company_info=True,
            company_news=True,
        )
    ],
)

stock_research_team = Team(
    name="Stock Research Team",
    mode="route",
    model=Ollama(id="phi4-mini"),
    members=[stock_searcher, company_info_agent],
    markdown=True,
    team_id=create_uuid_from_string("Stock Research Team"),
)