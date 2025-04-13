from agno.agent import Agent
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.tools.yfinance import YFinanceTools
from agno.playground import Playground, serve_playground_app
from agno.models.ollama import Ollama
from dotenv import load_dotenv
from team_agent import stock_research_team
from rag_agent import internal_document_agent

load_dotenv()

finance_agent = Agent(
    name="Finance Agent",
    model=Ollama(id="phi4-mini"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=["Use tables to display data"],
    storage=SqliteAgentStorage(table_name="finance_agent", db_file="agents.db"),
    add_history_to_messages=True,
    markdown=True,
)

app = Playground(
    agents=[finance_agent, internal_document_agent],
    teams=[stock_research_team],
    ).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)