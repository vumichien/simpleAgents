from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools       
from dotenv import load_dotenv
import os
load_dotenv(override=True)

web_search_agent = Agent(
    name="Web Search Agent",
    model=(
        OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        if os.getenv("LOCAL_MODEL") == "false"
        else Ollama(id="llama3.2:latest")
    ),
    # model=Ollama(id="qwen2.5:7b"),
    tools=[DuckDuckGoTools()],
    instructions=["常に情報源を含めてください"],
    show_tool_calls=True,
    markdown=True,
)
# web_agent.print_response("Tell me about OpenAI Sora?", stream=True)
