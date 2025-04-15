import io
from datetime import datetime, timedelta
from textwrap import dedent
from typing import List, Tuple

import requests
from agno.run.response import RunResponse

from agno.tools.googlecalendar import GoogleCalendarTools
from agno.run.response import RunResponse

try:
    from pypdf import PdfReader
except ImportError:
    raise ImportError(
        "pypdf is not installed. Please install it using `pip install pypdf`"
    )
from agno.agent.agent import Agent

from agno.models.openai.chat import OpenAIChat
from agno.models.ollama import Ollama
from agno.tools.resend import ResendTools
from agno.utils.log import logger
from agno.workflow.workflow import Workflow
from pydantic import BaseModel, Field
from dotenv import load_dotenv          
import os

load_dotenv(override=True)

credentials_path = os.getenv("GOOGLE_CALENDAR_CREDENTIALS")
token_path = os.getenv("GOOGLE_CALENDAR_TOKEN")


class ScreeningResult(BaseModel):
    name: str = Field(description="The name of the candidate")
    email: str = Field(description="The email of the candidate")
    score: float = Field(description="The score of the candidate from 0 to 10")
    feedback: str = Field(description="The feedback for the candidate")


class CandidateScheduledCall(BaseModel):
    name: str = Field(description="The name of the candidate")
    email: str = Field(description="The email of the candidate")
    call_time: str = Field(description="The time of the call")
    url: str = Field(description="The url of the call")


class Email(BaseModel):
    subject: str = Field(description="The subject of the email")
    body: str = Field(description="The body of the email")


current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class EmployeeRecruitmentWorkflow(Workflow):
    # Define a fixed workflow_id to identify this workflow

    description: str = dedent(
        """\
    An intelligent employee recruitment workflow that screens candidates, schedules interviews, 
    and sends emails to selected candidates.
    """
    )

    screening_agent: Agent = Agent(
        description="You are an HR agent that screens candidates for a job interview.",
        # model=Ollama(id="llama3.2:latest"),
        model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("LOCAL_MODEL") == "false" else Ollama(id="llama3.2:latest"),
        instructions=dedent(
            """
            You are an expert HR agent that screens candidates for a job interview.
            You are given a candidate's name and resume and job description.
            You need to screen the candidate and determine if they are a good fit for the job.
            You need to provide a score for the candidate from 0 to 10.
            You need to provide a feedback for the candidate on why they are a good fit or not.
            """
        ),
        response_model=ScreeningResult,
    )

    interview_scheduler_agent: Agent = Agent(
        description="You are an interview scheduler agent that schedules interviews for candidates.",
        # model=Ollama(id="llama3.2:latest"),
        model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("LOCAL_MODEL") == "false" else Ollama(id="llama3.2:latest"),
        instructions=dedent(
            """
            You are an interview scheduler agent that schedules interviews for candidates.
            You need to schedule interviews for the candidates using the Google Calendar tool.
            You need to schedule the interview for the candidate at the earliest possible time between 10am and 4pm.
            Check if the candidate and interviewer are available at the time and if the time is free in the calendar.
            You are in Tokyo (GMT+9) timezone and the current time is {current_time}. So schedule the call in future time with reference to current time.
            
            IMPORTANT SCHEDULING RULES:
            - Only schedule interviews on business days (Monday to Friday)
            - Never schedule interviews on weekends (Saturday or Sunday)
            - Avoid scheduling on public holidays
            
            IMPORTANT: When using the GoogleCalendarTools.create_event function, you must follow these steps:
            1. Call list_events() first to check for existing events
            2. Create a new event with specific parameters:
                - start_datetime and end_datetime should be ISO format (YYYY-MM-DDTHH:MM:SS)
                - Make sure to include the timezone parameter as 'Asia/Tokyo'
                - Include both the candidate and interviewer in the attendees list
                - Set a descriptive title and add details in the description
            3. Make sure to extract and store the event URL from the response
            4. Format the call_time as a human-readable string in the response
            5. ALWAYS ensure both call_time and url are non-empty in your response
            6. CRITICALLY IMPORTANT: The event must be scheduled for 2025 or later, not in the past. Schedule at least 2 days in the future from the current date.
            """
        ),
        tools=[
            GoogleCalendarTools(
                credentials_path=credentials_path, token_path=token_path
            ),
        ],
        response_model=CandidateScheduledCall,
        show_tool_calls=True,
    )

    email_writer_agent: Agent = Agent(
        description="You are an expert email writer agent that writes emails to selected candidates.",
        # model=Ollama(id="llama3.2:latest"),
        model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("LOCAL_MODEL") == "false" else Ollama(id="llama3.2:latest"),
        instructions=dedent(
            """
            You are an expert email writer agent that writes emails to selected candidates.
            You need to write an email and send it to the candidates using the Resend tool.
            You represent the company and the job position.
            You need to write an email that is concise and to the point.
            You need to write an email that is friendly and professional.
            You need to write an email that is not too formal and not too informal.
            The body of the email should be a detailed explanation of the job position and the candidate's qualifications.
            IMPORTANT: Format the email using HTML tags for proper structure.
            Follow this structure for your email formatting:
            "<p>Dear [Candidate Name],</p>",
            "<p>I am pleased to inform you that...</p>",
            "<p>Your interview is scheduled for [Date] at [Time]...</p>",
            "<p>During this interview...</p>",
            "<p>Please feel free to reach out if you have any questions.</p>",
            "<p>Best regards,<br>",
            "[Your Name]<br>",
            "[Your Title]<br>",
            "[Your Email]</p>",
            "Make sure to use HTML tags like <p>, <br>, <strong>, <ul>/<li> for better formatting.",
            """
        ),
        response_model=Email,
    )

    email_sender_agent: Agent = Agent(
        description="You are an expert email sender agent that sends emails to selected candidates.",
        # model=Ollama(id="llama3.2:latest"),
        model=OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("LOCAL_MODEL") == "false" else Ollama(id="llama3.2:latest"),
        instructions=dedent(
            """
            You are an expert email sender agent that sends emails to selected candidates.
            You need to send an email to the candidate using the Resend tool.
            You will be given the email subject and body and you need to send it to the candidate.
            IMPORTANT: The email body is already in HTML format, do not modify it.
            "Send the HTML content exactly as provided to ensure proper formatting in the email.",
            """
        ),
        tools=[ResendTools(from_email="onboarding@resend.dev")],
        show_tool_calls=True,
    )

    def is_holiday(self, date: datetime) -> bool:
        """Check if a date is a public holiday
        Currently just a simple check for major holidays in Japan
        In a production system, you would use a proper calendar API

        Args:
            date: Date to check

        Returns:
            bool: True if it's a holiday, False otherwise
        """
        # Get month and day
        month, day = date.month, date.day

        # List of common Japanese holidays - this is just a simple check
        # In a real application, you should use a proper calendar API
        holidays = [
            (1, 1),  # New Year's Day
            (2, 11),  # National Foundation Day
            (3, 21),  # Spring Equinox
            (4, 29),  # Showa Day
            (5, 3),  # Constitution Day
            (5, 4),  # Greenery Day
            (5, 5),  # Children's Day
            (8, 11),  # Mountain Day
            (9, 23),  # Autumn Equinox
            (11, 3),  # Culture Day
            (11, 23),  # Labor Thanksgiving Day
            (12, 23),  # Emperor's Birthday
            (12, 31),  # New Year's Eve
            (12, 30),  # Year End
            (12, 29),  # Year End
        ]

        return (month, day) in holidays

    def get_future_interview_time(self, days_from_now: int = 3) -> Tuple[str, str]:
        """Generate future interview time slots that are guaranteed to be in the future
        and only on business days (Monday to Friday, excluding holidays)

        Args:
            days_from_now: Number of days from today for the interview

        Returns:
            Tuple containing start and end times in ISO format
        """
        import random

        future_date = datetime.now() + timedelta(days=days_from_now)

        # Ensure the date is a business day (weekday and not a holiday)
        weekday = future_date.weekday()
        is_weekend = weekday >= 5  # Saturday (5) or Sunday (6)

        # Keep adjusting the date until we get a business day
        max_attempts = 10
        attempts = 0

        while (is_weekend or self.is_holiday(future_date)) and attempts < max_attempts:
            if is_weekend:
                # Add days to get to Monday
                days_to_add = 7 - weekday + 1 if weekday == 6 else 1
                future_date = future_date + timedelta(days=days_to_add)
                logger.info(
                    f"Adjusted weekend date to next weekday: {future_date.strftime('%Y-%m-%d')}"
                )
            elif self.is_holiday(future_date):
                # Add 1 day if it's a holiday
                future_date = future_date + timedelta(days=1)
                logger.info(
                    f"Adjusted holiday date to next day: {future_date.strftime('%Y-%m-%d')}"
                )

            # Recheck conditions
            weekday = future_date.weekday()
            is_weekend = weekday >= 5
            attempts += 1

        start_time = future_date.replace(
            hour=random.randint(14, 18), minute=0, second=0, microsecond=0
        )
        # Schedule for 1 hour
        end_time = start_time + timedelta(hours=1)

        # Format as ISO strings
        start_iso = start_time.strftime("%Y-%m-%dT%H:%M:%S")
        end_iso = end_time.strftime("%Y-%m-%dT%H:%M:%S")

        logger.info(
            f"Interview scheduled on {future_date.strftime('%A, %Y-%m-%d')} at {start_time.strftime('%H:%M')}"
        )

        return start_iso, end_iso

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

    def run(self, candidate_resume_urls: str) -> RunResponse:
        feedback = ""
        call_time = ""
        selected_candidates = []
        job_description = dedent(
            """
            We are hiring for backend and systems engineers!
            Join our team building the future of agentic software

            Apply if:
            🧠 You know your way around Python, typescript, docker, and AWS.
            ⚙️ Love to build in public and contribute to open source.
            🚀 Are ok dealing with the pressure of an early-stage startup.
            🏆 Want to be a part of the biggest technological shift since the internet.
            🌟 Bonus: experience with infrastructure as code.
            🌟 Bonus: starred Agno repo.
            """
        )
        if not candidate_resume_urls:
            raise Exception("candidate_resume_urls cannot be empty")

        # Get future interview times
        start_time, end_time = self.get_future_interview_time(days_from_now=5)
        logger.info(f"Default interview time slot: {start_time} to {end_time}")
        candidate_resume_urls = [candidate_resume_urls]
        for resume_url in candidate_resume_urls:
            # Extract text from PDF resume
            if resume_url in self.session_state:
                resume_content = self.session_state[resume_url]
            else:
                resume_content = self.extract_text_from_pdf(resume_url)
                self.session_state[resume_url] = resume_content
            screening_result = None

            if resume_content:
                # Screen the candidate
                input = f"Candidate resume: {resume_content}, Job description: {job_description}"
                screening_result = self.screening_agent.run(input)
                feedback = screening_result.content.feedback
                logger.info(f"Screening result: {screening_result}")
            else:
                logger.error(f"Could not process resume from URL: {resume_url}")

            if (
                screening_result
                and screening_result.content
                and screening_result.content.score > 7.0
            ):
                selected_candidates.append(screening_result.content)

        for selected_candidate in selected_candidates:
            try:
                # Update the input message to include the suggested time
                input = f"Schedule a 1hr call with Candidate name: {selected_candidate.name}, Candidate email: {selected_candidate.email} and the interviewer would be Tran Trung Thanh CTO with email onboarding@resend.dev. Suggested time slot is from {start_time} to {end_time} with timezone Asia/Tokyo."
                scheduled_call = self.interview_scheduler_agent.run(input)

                # Debug the scheduled call response
                logger.info(f"Scheduled call response: {scheduled_call}")
                call_time = scheduled_call.content.call_time

                if scheduled_call.content:
                    logger.info(
                        f"Call details: name='{scheduled_call.content.name}' email='{scheduled_call.content.email}' call_time='{scheduled_call.content.call_time}' url='{scheduled_call.content.url}'"
                    )
                else:
                    logger.error("No content in scheduled_call response")

                # Create fallback if Google Calendar fails (empty call_time or url)
                if (
                    not scheduled_call.content
                    or not scheduled_call.content.call_time
                    or not scheduled_call.content.url
                ):
                    logger.warning(
                        "Using fallback scheduling as Google Calendar failed"
                    )
                    # Use the utility function to get future dates
                    start_iso, end_iso = self.get_future_interview_time(days_from_now=3)
                    formatted_date = start_iso.replace("T", " ")

                    # Update or create the response with fallback values
                    if not scheduled_call.content:
                        from agno.run.response import RunResponse

                        scheduled_call = RunResponse(
                            content=CandidateScheduledCall(
                                name=selected_candidate.name,
                                email=selected_candidate.email,
                                call_time=formatted_date,
                                url="https://meet.google.com/manual-scheduling-required",
                            )
                        )
                    else:
                        scheduled_call.content.call_time = formatted_date
                        scheduled_call.content.url = (
                            "https://meet.google.com/manual-scheduling-required"
                        )

                    logger.info(f"Created fallback schedule: {formatted_date}")

                # Verify the event is scheduled in the future (2025 or later)
                if scheduled_call.content and scheduled_call.content.call_time:
                    try:
                        # Try to parse the date string, which might be in various formats
                        call_time_str = scheduled_call.content.call_time
                        # First check if the year is present and is at least 2025
                        current_year = datetime.now().year
                        if (
                            str(current_year) not in call_time_str
                            or "2023" in call_time_str
                            or "2024" in call_time_str
                        ):
                            logger.warning(
                                f"Event scheduled in the past or incorrect year: {call_time_str}"
                            )
                            # Use the utility function to get future dates
                            start_iso, _ = self.get_future_interview_time(
                                days_from_now=4
                            )
                            formatted_date = start_iso.replace("T", " ")
                            scheduled_call.content.call_time = formatted_date
                            logger.info(f"Updated to future date: {formatted_date}")
                    except Exception as e:
                        logger.error(f"Error parsing date: {str(e)}")

                if (
                    scheduled_call.content
                    and scheduled_call.content.url
                    and scheduled_call.content.call_time
                ):
                    try:
                        # Construct a well-formatted email using HTML
                        # Format meeting URL for better display
                        meeting_url = scheduled_call.content.url
                        meeting_link = f'<a href="{meeting_url}">Click here to join the meeting</a>'

                        input = dedent(
                            f"""
                            Write a well-formatted HTML email to:
                            - Candidate name: {selected_candidate.name}
                            - Candidate email: {selected_candidate.email}
                            - Interview scheduled at: {scheduled_call.content.call_time}
                            - Meeting link: {meeting_link} (this is already formatted as HTML, include it exactly as provided)
                            - Job Position: Backend and Systems Engineer
                            - Company: Detomo Inc.
                            
                            Congratulate them for passing the initial screening and being selected for an interview.
                            
                            The email should be from Tran Trung Thanh, CTO (email: thanh_tt@detomo.co.jp)
                            
                            IMPORTANT:
                            1. Format the email using HTML tags (<p>, <br>, <strong>, etc.) for proper structure
                            2. Make sure the content is well-organized with clear sections
                            3. Include a proper signature with the sender's name, title, and company name (Detomo Inc.)
                            4. Include the meeting link exactly as provided (it's already an HTML anchor tag)
                            5. The email should look professional when rendered in an email client
                            6. Make sure to mention the job position (Backend and Systems Engineer) and company name (Detomo Inc.) in the email body
                        """
                        ).strip()

                        email = self.email_writer_agent.run(input)

                        # Debug email content
                        if email.content:
                            logger.info(
                                f"Email content: subject='{email.content.subject}', body preview='{email.content.body[:50]}...'"
                            )
                        else:
                            logger.error(
                                "Email writer agent failed to generate email content"
                            )

                        if email.content:
                            try:
                                # Ensure the HTML has proper structure
                                html_body = email.content.body
                                # If the body doesn't include basic HTML structure, add it
                                if not html_body.strip().startswith(
                                    "<!DOCTYPE html>"
                                ) and not html_body.strip().startswith("<html>"):
                                    html_body = dedent(
                                        f"""
                                    <!DOCTYPE html>
                                    <html>
                                    <head>
                                        <meta charset="UTF-8">
                                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                                        <style>
                                            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                                            p {{ margin-bottom: 16px; }}
                                            .signature {{ margin-top: 24px; color: #555; }}
                                        </style>
                                    </head>
                                    <body>
                                        {html_body}
                                    </body>
                                    </html>
                                    """
                                    ).strip()

                                # Prepare the email sending command
                                input = dedent(
                                    f"""
                                    Send email to {selected_candidate.email} with:
                                    
                                    Subject: {email.content.subject}
                                    
                                    HTML Body: 
                                    {html_body}
                                    
                                    IMPORTANT: The body is already in HTML format, send it as-is without modification.
                                """
                                ).strip()

                                email_response = self.email_sender_agent.run(input)
                                logger.info(f"Email sending result: {email_response}")
                            except Exception as e:
                                logger.error(f"Error sending email: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error writing email: {str(e)}")
            except Exception as e:
                logger.error(
                    f"Error scheduling interview for {selected_candidate.name}: {str(e)}"
                )
                # Continue with next candidate
                continue

        from agno.run.response import RunResponse

        return RunResponse(
            content=(
                f"This candidate was selected for the interview. \n\n Feedback: {feedback}  \n\n The meeting invitation is sent to the candidate at {call_time}"
                if len(selected_candidates) > 0
                else f"This candidate was not selected for the interview. \n\n Feedback: {feedback}"
            ),
            workflow_id=self.workflow_id,
        )


employee_recruiter_workflow = EmployeeRecruitmentWorkflow(
    workflow_id="employee-recruiter-workflow",
    debug_mode=True,
)
# result = employee_recruiter_workflow.run(
#     candidate_resume_urls=[
#         "https://documents1692.s3.ap-northeast-1.amazonaws.com/resume_sample.pdf"
#     ],
#     job_description="""
# We are hiring for backend and systems engineers!
# Join our team building the future of agentic software

# Apply if:
# 🧠 You know your way around Python, typescript, docker, and AWS.
# ⚙️ Love to build in public and contribute to open source.
# 🚀 Are ok dealing with the pressure of an early-stage startup.
# 🏆 Want to be a part of the biggest technological shift since the internet.
# 🌟 Bonus: experience with infrastructure as code.
# 🌟 Bonus: starred Agno repo.
#     """,
# )
# print(result.content)
