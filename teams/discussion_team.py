from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from agno.team.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.hackernews import HackerNewsTools
import os
from dotenv import load_dotenv

load_dotenv(override=True)

reddit_researcher = Agent(
    name="Reddit Researcher",
    role="Research a topic on Reddit",
    model=(
        OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        if os.getenv("LOCAL_MODEL") == "false"
        else Ollama(id="llama3.2:latest")
    ),
    tools=[DuckDuckGoTools()],
    add_name_to_instructions=True,
    instructions=dedent(
        """
    あなたはRedditの研究者です。
    Redditで調査するトピックが与えられます。
    Redditで最も関連性の高い投稿を見つける必要があります。
    """
    ),
)

hackernews_researcher = Agent(
    name="HackerNews Researcher",
    model=(
        OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        if os.getenv("LOCAL_MODEL") == "false"
        else Ollama(id="llama3.2:latest")
    ),
    role="Research a topic on HackerNews.",
    tools=[HackerNewsTools()],
    add_name_to_instructions=True,
    instructions=dedent(
        """
    あなたはHackerNewsの研究者です。
    HackerNewsで調査するトピックが与えられます。
    HackerNewsで最も関連性の高い投稿を見つける必要があります。
    """
    ),
)


discussion_team = Team(
    name="Discussion Team",
    mode="collaborate",
    model=(
        OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        if os.getenv("LOCAL_MODEL") == "false"
        else Ollama(id="llama3.2:latest")
    ),
    members=[
        reddit_researcher,
        hackernews_researcher,
    ],
    instructions=[
        "あなたはディスカッションマスターです。",
        "チームがコンセンサスに達したと思ったら、議論を止める必要があります。",
    ],
    success_criteria="チームがコンセンサスに達しました。",
    enable_agentic_context=True,
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
    show_members_responses=True,
)

# discussion_team.run(
#     message="Start the discussion on the topic: 'What is the best way to learn to code?'",
#     workflow_id="discussion-team",
# )
