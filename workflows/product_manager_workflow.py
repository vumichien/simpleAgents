import os
from datetime import datetime
from typing import Dict, List, Optional

from agno.agent.agent import Agent
from agno.run.response import RunEvent, RunResponse
from agno.storage.postgres import PostgresStorage
from agno.tools.slack import SlackTools
from agno.utils.log import logger
from agno.workflow.workflow import Workflow
from agno.models.ollama import Ollama
from agno.models.openai.chat import OpenAIChat
from agno.storage.postgres import PostgresStorage
from pydantic import BaseModel, Field
import requests
import io
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

class Task(BaseModel):
    task_title: str = Field(..., description="The title of the task")
    task_description: Optional[str] = Field(
        None, description="The description of the task"
    )
    task_assignee: Optional[str] = Field(None, description="The assignee of the task")

class TaskList(BaseModel):
    tasks: List[Task] = Field(..., description="A list of tasks")


class ProductManagerWorkflow(Workflow):
    description: str = "Generate tasks and send slack notifications to the team from meeting notes."
    task_agent: Agent = Agent(
        name="Task Agent",
        model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("LOCAL_MODEL") == "false" else Ollama(id="llama3.2:latest"),
        instructions=[
            "Given a meeting note, generate a list of tasks with titles, descriptions and assignees."
        ],
        response_model=TaskList,
    )

    slack_agent: Agent = Agent(
        name="Slack Agent",
        model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("LOCAL_MODEL") == "false" else Ollama(id="llama3.2:latest"),
        instructions=[
            "Send a slack notification to the #dtm_japan_it_week channel with a heading (bold text) including the current date and tasks in the following format: ",
            "*Title*: <issue_title>",
            "*Description*: <issue_description>",
            "*Assignee*: <issue_assignee>",
            "*Issue Link*: <issue_link>",
        ],
        tools=[SlackTools(token=os.getenv("SLACK_TOKEN"))],
        show_tool_calls=True
    )

    def extract_text_from_pdf(self, pdf_url: str) -> str:
        """Download PDF from URL and extract text content"""
        try:
            # Download PDF content
            response = requests.get(pdf_url)
            response.raise_for_status()

            # Create PDF reader object
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PdfReader(pdf_file)

            # Extract text from all pages
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text()

            return text_content

        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return ""
        
    def get_tasks_from_cache(self, current_date: str) -> Optional[TaskList]:
        if "meeting_notes" in self.session_state:
            for cached_tasks in self.session_state["meeting_notes"]:
                if cached_tasks["date"] == current_date:
                    return cached_tasks["tasks"]
        return None

    def get_tasks_from_meeting_notes(self, meeting_notes: str) -> Optional[TaskList]:
        num_tries = 0
        tasks: Optional[TaskList] = None
        while tasks is None and num_tries < 3:
            num_tries += 1
            try:
                response: RunResponse = self.task_agent.run(meeting_notes)
                if (
                    response
                    and response.content
                    and isinstance(response.content, TaskList)
                ):
                    tasks = response.content
                else:
                    logger.warning("Invalid response from task agent, trying again...")
            except Exception as e:
                logger.warning(f"Error generating tasks: {e}")

        return tasks

    def run(
        self, meeting_notes_url: str, use_cache: bool = False
    ) -> RunResponse:
        logger.info(f"Generating tasks from meeting notes: {meeting_notes_url}")
        current_date = datetime.now().strftime("%Y-%m-%d")
        meeting_notes = self.extract_text_from_pdf(meeting_notes_url)
        if use_cache:
            tasks: Optional[TaskList] = self.get_tasks_from_cache(current_date)
        else:
            tasks = self.get_tasks_from_meeting_notes(meeting_notes)

        if tasks is None or len(tasks.tasks) == 0:
            return RunResponse(
                run_id=self.run_id,
                event=RunEvent.workflow_completed,
                content="Sorry, could not generate tasks from meeting notes.",
            )

        if "meeting_notes" not in self.session_state:
            self.session_state["meeting_notes"] = []
        self.session_state["meeting_notes"].append(
            {"date": current_date, "tasks": tasks.model_dump_json()}
        )

        # Send slack notification with tasks
        if tasks:
            logger.info(
                f"Sending slack notification with tasks: {tasks.model_dump_json()}"
            )
            slack_response: RunResponse = self.slack_agent.run(
                tasks.model_dump_json()
            )
            logger.info(f"Slack response: {slack_response}")

        return slack_response


# Create the workflow
product_manager_workflow = ProductManagerWorkflow(
    session_id="product-manager",
    storage=PostgresStorage(
        table_name="product_manager_workflows",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        auto_upgrade_schema=True,
        mode="workflow",
    ),
)

meeting_notes_url = "https://documents1692.s3.ap-northeast-1.amazonaws.com/meeting_note.pdf"

# Run workflow
product_manager_workflow.run(meeting_notes_url=meeting_notes_url)