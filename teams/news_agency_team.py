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
    model=(
        OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        if os.getenv("LOCAL_MODEL") == "false"
        else Ollama(id="llama3.2:latest")
    ),
    role="HackerNewsから最新の記事を取得します。",
    tools=[HackerNewsTools()],
)

web_searcher = Agent(
    name="Web Searcher",
    model=(
        OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        if os.getenv("LOCAL_MODEL") == "false"
        else Ollama(id="llama3.2:latest")
    ),
    role="トピックに関する情報をウェブで検索します",
    tools=[DuckDuckGoTools()],
    # add_datetime_to_instructions=True,
)

article_reader = Agent(
    name="Article Reader",
    role="URLから記事を読み込みます。",
    tools=[Newspaper4kTools()],
)


news_agency_team = Team(
    name="HackerNews Team",
    mode="coordinate",
    model=(
        OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        if os.getenv("LOCAL_MODEL") == "false"
        else Ollama(id="llama3.2:latest")
    ),
    members=[hn_researcher, web_searcher, article_reader],
    instructions=[
        "まず、ユーザーが質問していることについてHackerNewsを検索してください。",
        "次に、記事リーダーに詳細情報を得るためにストーリーのリンクを読むよう依頼してください。",
        "重要：記事リーダーに読むべきリンクを提供する必要があります。",
        "続いて、ウェブ検索者に各ストーリーについてさらに情報を得るための検索を依頼してください。",
        "最後に、思慮深く魅力的な要約を提供してください。",
    ],
    # response_model=Article,
    show_tool_calls=True,
    markdown=True,
    debug_mode=True,
    show_members_responses=True,
)
