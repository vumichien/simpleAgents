from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.models.openai import OpenAIChat
from agno.tools.googlecalendar import GoogleCalendarTools
from dotenv import load_dotenv
import os
from textwrap import dedent

load_dotenv(override=True)

credentials_path = os.getenv("GOOGLE_CALENDAR_CREDENTIALS")
token_path = os.getenv("GOOGLE_CALENDAR_TOKEN")

calendar_agent = Agent(
    name="Calendar Agent",
    tools=[
        GoogleCalendarTools(credentials_path=credentials_path, token_path=token_path)
    ],
    show_tool_calls=True,
    markdown=True,
    model=(
        OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
        if os.getenv("LOCAL_MODEL") == "false"
        else Ollama(id="llama3.2:latest")
    ),
    debug_mode=True,
    instructions=dedent(
        """\
    あなたはGoogleカレンダーの管理を手伝う便利なアシスタントです。
    Googleカレンダーを管理するために以下のツールを使用できます：
    
    GoogleCalendarToolsには以下の機能があります：
    - list_events(limit: int = 10, date_from: str = today) - これらのパラメータのみ受け付けます
    - create_event(start_datetime, end_datetime, title, description, location, timezone, attendees)
    
    重要：list_eventsでは、timezoneをパラメータとして含めないでください。date_fromは「2025-04-14」のようなISO形式である必要があります。
    
    質問と同じ言語でツールを使用して質問に答える必要があります。
    回答を作り上げたり、他の情報を提供したりしないでください。                
    """
    ),
)

# calendar_agent.print_response("Create a event on 2025-04-14 at 10:00 and end time is 2025-04-14 at 11:00, timezone is Asia/Tokyo, with title 'test AI'")
# calendar_agent.print_response("List events for April 14, 2025")
