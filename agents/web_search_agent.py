from phi.agent import Agent
from phi.model.ollama import Ollama
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import dotenv_values

config = dotenv_values(".env")

web_agent = Agent(
    name="Web Agent",
    # model=OpenRouter(id="mistralai/mistral-small-3.1-24b-instruct:free", api_key=config["OPENROUTER_API_KEY"]),
    model=Ollama(id="qwen2.5:7b"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)
web_agent.print_response("Tell me about OpenAI Sora?", stream=True)