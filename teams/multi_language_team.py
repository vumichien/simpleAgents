from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from agno.team.team import Team
import os
from dotenv import load_dotenv

load_dotenv(override=True)

english_agent = Agent(
    name="English Agent",
    role="You only answer in English",
    model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("LOCAL_MODEL") == "false" else Ollama(id="llama3.2:latest"),
)
japanese_agent = Agent(
    name="Japanese Agent",
    role="You only answer in Japanese",
    model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("LOCAL_MODEL") == "false" else Ollama(id="llama3.2:latest"),
)
chinese_agent = Agent(
    name="Chinese Agent",
    role="You only answer in Chinese",
    model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("LOCAL_MODEL") == "false" else Ollama(id="llama3.2:latest"),
)
spanish_agent = Agent(
    name="Spanish Agent",
    role="You can only answer in Spanish",
    model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("LOCAL_MODEL") == "false" else Ollama(id="llama3.2:latest"),
)
french_agent = Agent(
    name="French Agent",
    role="You can only answer in French",
    model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("LOCAL_MODEL") == "false" else Ollama(id="llama3.2:latest"),
)
german_agent = Agent(
    name="German Agent",
    role="You can only answer in German",
    model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("LOCAL_MODEL") == "false" else Ollama(id="llama3.2:latest"),
)

multi_language_team = Team(
    name="Multi Language Team",
    mode="route",
    model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("LOCAL_MODEL") == "false" else Ollama(id="llama3.2:latest"),
    members=[
        english_agent,
        spanish_agent,
        japanese_agent,
        french_agent,
        german_agent,
        chinese_agent,
    ],
    show_tool_calls=True,
    markdown=True,
    instructions=[
        "You are a language router that directs questions to the appropriate language agent.",
        "If the user asks in a language whose agent is not a team member, respond in English with:",
        "'I can only answer in the following languages: English, Spanish, Japanese, French and German. Please ask your question in one of these languages.'",
        "Always check the language of the user's input before routing to an agent.",
        "For unsupported languages like Italian, respond in English with the above message.",
    ],
    show_members_responses=True,
)