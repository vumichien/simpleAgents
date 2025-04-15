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
    model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("LOCAL_MODEL") == "false" else Ollama(id="llama3.2:latest"),
    debug_mode=True,
    instructions=dedent(
        """\
    You are a helpful assistant that can help me manage my Google Calendar.
    You can use the following tools to manage my Google Calendar:
    
    GoogleCalendarTools has these functions:
    - list_events(limit: int = 10, date_from: str = today) - Only accepts these parameters
    - create_event(start_datetime, end_datetime, title, description, location, timezone, attendees)
    
    IMPORTANT: For list_events, do NOT include timezone as a parameter. The date_from should be in ISO format like '2025-04-14'.
    
    You will have to answer the question based on the tools with the same language as the question.
    Don't make up an answer and don't give any other information.                
    """
    ),
)

# calendar_agent.print_response("Create a event on 2025-04-14 at 10:00 and end time is 2025-04-14 at 11:00, timezone is Asia/Tokyo, with title 'test AI'")
# calendar_agent.print_response("List events for April 14, 2025")
