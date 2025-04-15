from typing import List

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from agno.run.team import TeamRunResponse  # type: ignore
from agno.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.hackernews import HackerNewsTools
from agno.tools.newspaper4k import Newspaper4kTools
import os
from dotenv import load_dotenv

load_dotenv(override=True)


hn_researcher = Agent(
    name="HackerNews Researcher",
    model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("LOCAL_MODEL") == "false" else Ollama(id="llama3.2:latest"),
    role="Gets top stories from hackernews.",
    tools=[HackerNewsTools()],
)

web_searcher = Agent(
    name="Web Searcher",
    model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("LOCAL_MODEL") == "false" else Ollama(id="llama3.2:latest"),
    role="Searches the web for information on a topic",
    tools=[DuckDuckGoTools()],
    # add_datetime_to_instructions=True,
)

article_reader = Agent(
    name="Article Reader",
    role="Reads articles from URLs.",
    tools=[Newspaper4kTools()],
)


news_agency_team = Team(
    name="HackerNews Team",
    mode="coordinate",
    model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("LOCAL_MODEL") == "false" else Ollama(id="llama3.2:latest"),
    members=[hn_researcher, web_searcher, article_reader],
    instructions=[
        "First, search hackernews for what the user is asking about.",
        "Then, ask the article reader to read the links for the stories to get more information.",
        "Important: you must provide the article reader with the links to read.",
        "Then, ask the web searcher to search for each story to get more information.",
        "Finally, provide a thoughtful and engaging summary.",
    ],
    # response_model=Article,
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
    show_members_responses=True,
)