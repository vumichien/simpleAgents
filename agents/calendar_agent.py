from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.googlecalendar import GoogleCalendarTools
from dotenv import dotenv_values
import os
from textwrap import dedent

config = dotenv_values(".env")

credentials_path = config["GOOGLE_CALENDAR_CREDENTIALS"]
token_path = config["GOOGLE_CALENDAR_TOKEN"]

print(credentials_path)
print(token_path)

calendar_agent = Agent(
    name="Calendar Agent",
    tools=[
        GoogleCalendarTools(credentials_path=credentials_path, token_path=token_path)
    ],
    show_tool_calls=True,
    markdown=True,
    model=Ollama(id="llama3.2:latest"),
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
