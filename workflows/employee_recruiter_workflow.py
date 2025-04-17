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
from agno.storage.postgres import PostgresStorage
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
    インテリジェントな従業員採用ワークフローです。
    候補者をスクリーニングし、面接をスケジュールし、選択された候補者にメールを送信します。
    """
    )

    screening_agent: Agent = Agent(
        description="あなたは面接のために候補者を選考する人事担当者です",
        model=(
            OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
            if os.getenv("LOCAL_MODEL") == "false"
            else Ollama(id="gemma3:12b")
        ),
        instructions=dedent(
            """
            あなたは求人面接の候補者をスクリーニングする専門のHRエージェントです。
            候補者の名前と履歴書、求人情報が与えられます。
            候補者をスクリーニングし、その仕事に適しているかどうかを判断する必要があります。
            候補者に0から10までのスコアを付ける必要があります。
            候補者が適任である理由、またはそうでない理由についてフィードバックを提供する必要があります。
            フィードバックを箇条書きで提示する必要があります。
            
            """
        ),
        response_model=ScreeningResult,
    )

    interview_scheduler_agent: Agent = Agent(
        description="You are an interview scheduler agent that schedules interviews for candidates.",
        model=(
            OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
            if os.getenv("LOCAL_MODEL") == "false"
            else Ollama(id="llama3.2:latest")
        ),
        instructions=dedent(
            """
            あなたは候補者の面接をスケジュールする面接スケジューラーエージェントです。
            Googleカレンダーツールを使用して候補者の面接をスケジュールする必要があります。
            午前10時から午後4時までの最も早い可能な時間に候補者の面接をスケジュールしてください。
            候補者と面接官がその時間に利用可能かどうか、およびカレンダーの時間が空いているかどうかを確認してください。
            あなたは東京（GMT+9）タイムゾーンにいて、現在の時刻は{current_time}です。そのため、現在の時刻を参考に将来の時間に通話をスケジュールしてください。
            
            重要なスケジュール規則：
            - 面接は営業日（月曜日から金曜日）にのみスケジュールしてください
            - 週末（土曜日または日曜日）には絶対に面接をスケジュールしないでください
            - 祝日にスケジュールすることを避けてください
            
            重要：GoogleCalendarTools.create_event関数を使用する際には、以下の手順に従う必要があります：
            1. まずlist_events()を呼び出して既存のイベントを確認する
            2. 特定のパラメータで新しいイベントを作成する：
                - start_datetimeとend_datetimeはISO形式である必要があります（YYYY-MM-DDTHH:MM:SS）
                - timezoneパラメータとして'Asia/Tokyo'を必ず含めてください
                - 候補者と面接官の両方を出席者リストに含めてください
                - 説明的なタイトルを設定し、詳細を説明欄に追加してください
            3. 応答からイベントURLを抽出して保存してください
            4. レスポンスではcall_timeを人間が読みやすい文字列形式にしてください
            5. 常にレスポンスでcall_timeとurlの両方が空でないことを確認してください
            6. 非常に重要：イベントは2025年以降に予定され、過去ではないようにしてください。現在の日付から少なくとも2日後にスケジュールしてください。
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
        model=(
            OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
            if os.getenv("LOCAL_MODEL") == "false"
            else Ollama(id="suzume-multi:latest ")
        ),
        instructions=dedent(
            """
            あなたは選ばれた候補者にメールを書く専門のメール作成エージェントです。
            Resendツールを使用して候補者にメールを書き、送信する必要があります。
            あなたは会社と求人ポジションを代表しています。
            簡潔で要点を押さえたメールを書く必要があります。
            フレンドリーでプロフェッショナルなメールを書く必要があります。
            あまりにも形式ばっておらず、あまりにもカジュアルでもないメールを書く必要があります。
            メールの本文は、求人ポジションと候補者の資格についての詳細な説明であるべきです。
            重要：適切な構造のためにHTMLタグを使用してメールをフォーマットしてください。
            メールのフォーマットには以下の構造に従ってください：
            "<p>[候補者名]様、</p>",
            "<p>ご連絡いたします...</p>",
            "<p>面接は[日付]の[時間]に予定されています...</p>",
            "<p>この面接では...</p>",
            "<p>ご質問があればいつでもお問い合わせください。</p>",
            "<p>敬具、<br>",
            "[あなたの名前]<br>",
            "[あなたの役職]<br>",
            "[あなたのメール]</p>",
            "より良いフォーマットのために<p>、<br>、<strong>、<ul>/<li>などのHTMLタグを使用してください。",
            """
        ),
        response_model=Email,
    )

    email_sender_agent: Agent = Agent(
        description="You are an expert email sender agent that sends emails to selected candidates.",
        # model=Ollama(id="llama3.2:latest"),
        model=(
            OpenAIChat(id="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
            if os.getenv("LOCAL_MODEL") == "false"
            else Ollama(id="llama3.2:latest")
        ),
        instructions=dedent(
            """
            あなたは選ばれた候補者にメールを送信する専門のメール送信エージェントです。
            Resendツールを使用して候補者にメールを送信する必要があります。
            メールの件名と本文が与えられるので、それを候補者に送信する必要があります。
            重要：メール本文はすでにHTML形式になっているため、変更しないでください。
            "適切なフォーマットをメールで確保するために、提供されたHTMLコンテンツをそのまま送信してください。"
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
            バックエンドおよびシステムエンジニアを募集しています！
            エージェント型ソフトウェアの未来を構築するチームに参加しませんか

            以下の条件に当てはまる方は応募してください：
            🧠 Python、TypeScript、Docker、AWSに精通している方。
            ⚙️ 公開での開発とオープンソースへの貢献が好きな方。
            🚀 初期スタートアップのプレッシャーに対応できる方。
            🏆 インターネット以来最大の技術的変革の一部になりたい方。
            🌟 ボーナス：Infrastructure as Codeの経験がある方。
            🌟 ボーナス：Agnoリポジトリにスターを付けている方。
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
                        meeting_link = f'<a href="{meeting_url}">ミーティングに参加するにはこちらをクリックしてください</a>'

                        input = dedent(
                            f"""
                            以下の情報でHTML形式のメールを作成してください：
                            - 候補者名: {selected_candidate.name}
                            - 候補者メール: {selected_candidate.email}
                            - 面接予定時間: {scheduled_call.content.call_time}
                            - ミーティングリンク: {meeting_link} (これはすでにHTML形式になっています。提供されたとおりに含めてください)
                            - 職種: バックエンドおよびシステムエンジニア
                            - 会社名: デトモ株式会社
                            
                            初期スクリーニングに合格し、面接に選ばれたことをお祝いするメールを作成してください。
                            
                            メールの送信者はCTOのTran Trung Thanh（メール: thanh_tt@detomo.co.jp）です。
                            
                            重要事項:
                            1. 適切な構造のためにHTMLタグ（<p>、<br>、<strong>など）を使用してメールをフォーマットしてください
                            2. コンテンツが明確なセクションで整理されていることを確認してください
                            3. 送信者の名前、役職、会社名（デトモ株式会社）を含む適切な署名を含めてください
                            4. ミーティングリンクは提供されたとおりに含めてください（すでにHTMLアンカータグになっています）
                            5. メールはメールクライアントでレンダリングされた際にプロフェッショナルに見えるようにしてください
                            6. メール本文に職種（バックエンドおよびシステムエンジニア）と会社名（デトモ株式会社）を必ず言及してください
                            """
                        ).strip()

                        email = self.email_writer_agent.run(input)

                        # Debug email content
                        if email.content:
                            logger.info(
                                f"メール内容: 件名='{email.content.subject}', 本文プレビュー='{email.content.body}...'"
                            )
                        else:
                            logger.error(
                                "メール作成エージェントがメール内容の生成に失敗しました"
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
                                    以下の内容で{selected_candidate.email}にメールを送信してください：
                                    
                                    件名: {email.content.subject}
                                    
                                    HTML本文: 
                                    {html_body}
                                    
                                    重要: 本文はすでにHTML形式になっているため、変更せずにそのまま送信してください。
                                """
                                ).strip()

                                email_response = self.email_sender_agent.run(input)
                                logger.info(f"メール送信結果: {email_response}")
                            except Exception as e:
                                logger.error(f"メール送信エラー: {str(e)}")
                    except Exception as e:
                        logger.error(f"メール作成エラー: {str(e)}")
            except Exception as e:
                logger.error(
                    f"{selected_candidate.name}の面接スケジュール中にエラーが発生しました: {str(e)}"
                )
                # Continue with next candidate
                continue

        from agno.run.response import RunResponse

        return RunResponse(
            content=(
                f"仕事内容: {job_description} \n\n この候補者は面接に選ばれました。 \n\n フィードバック: {feedback}  \n\n 会議の招待状は候補者に {call_time} に送信されました。　\n\n 会議リンク: {meeting_url}"
                if len(selected_candidates) > 0
                else f"仕事内容: {job_description} \n\n この候補者は面接に選ばれませんでした。 \n\n フィードバック: {feedback}"
            ),
            workflow_id=self.workflow_id,
        )


employee_recruiter_workflow = EmployeeRecruitmentWorkflow(
    workflow_id="employee-recruiter-workflow",
    storage=PostgresStorage(
        table_name="employee_recruiter_workflows",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        auto_upgrade_schema=True,
        mode="workflow",
    ),
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
