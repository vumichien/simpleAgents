from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.model.ollama import Ollama
from phi.storage.sqlite import SqliteStorage # noqa: F401

web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=Ollama(id="qwen2.5:7b"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=Ollama(id="qwen2.5:7b"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
    instructions=["Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    team=[web_agent, finance_agent],
    model=Ollama(id="qwen2.5:7b"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
)

agent_team.print_response("Summarize analyst recommendations and share the latest news for TESLA", stream=True)