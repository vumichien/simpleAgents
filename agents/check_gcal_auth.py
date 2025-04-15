import os
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import datetime

# Load environment variables
load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/calendar"]


def check_google_calendar():
    """Verify Google Calendar authentication and list upcoming events"""

    credentials_path = os.getenv("GOOGLE_CALENDAR_CREDENTIALS")
    token_path = os.getenv("GOOGLE_CALENDAR_TOKEN")

    if not credentials_path or not os.path.exists(credentials_path):
        print(
            f"❌ Error: Google Calendar credentials file not found at {credentials_path}"
        )
        return False

    print(f"✅ Found credentials file at: {credentials_path}")

    creds = None
    # Check if token file exists
    if token_path and os.path.exists(token_path):
        print(f"✅ Found token file at: {token_path}")
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    else:
        print(f"❌ Token file not found at {token_path}")

    # Check if credentials are valid
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("🔄 Refreshing expired credentials...")
            creds.refresh(Request())
            print("✅ Credentials refreshed successfully")
        else:
            print("🔑 Need to authorize app...")
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
            print("✅ Authorization successful")

        # Save the credentials for future use
        if token_path:
            with open(token_path, "w") as token:
                token.write(creds.to_json())
            print(f"💾 Token saved to {token_path}")

    try:
        # Build the Google Calendar service
        service = build("calendar", "v3", credentials=creds)
        print("✅ Google Calendar service created successfully")

        # Try to get upcoming events as a test
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        print("📅 Fetching upcoming events...")
        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=now,
                maxResults=10,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )

        events = events_result.get("items", [])
        if not events:
            print("ℹ️ No upcoming events found")
        else:
            print(f"✅ Found {len(events)} upcoming events")
            for event in events:
                start = event["start"].get("dateTime", event["start"].get("date"))
                print(f"   - {start} : {event['summary']}")

        # Attempt to create a test event
        print("\n🧪 Testing event creation...")
        # Create event 5 days from now
        start_time = (
            (datetime.datetime.now() + datetime.timedelta(days=5))
            .replace(hour=10, minute=0, second=0)
            .isoformat()
        )
        end_time = (
            (datetime.datetime.now() + datetime.timedelta(days=5))
            .replace(hour=11, minute=0, second=0)
            .isoformat()
        )

        event = {
            "summary": "Test Event - Please Delete",
            "location": "Virtual Meeting",
            "description": "This is a test event created to check Google Calendar integration",
            "start": {"dateTime": start_time, "timeZone": "Asia/Tokyo"},
            "end": {"dateTime": end_time, "timeZone": "Asia/Tokyo"},
            "attendees": [{"email": "onboarding@resend.dev"}],
        }

        created_event = (
            service.events().insert(calendarId="primary", body=event).execute()
        )
        print(f"✅ Test event created successfully: {created_event.get('htmlLink')}")

        # Delete the test event immediately
        try:
            service.events().delete(
                calendarId="primary", eventId=created_event["id"]
            ).execute()
            print("🗑️ Test event deleted successfully")
        except Exception as e:
            print(f"❌ Error deleting test event: {str(e)}")

        return True

    except HttpError as error:
        print(f"❌ API Error: {error}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False


if __name__ == "__main__":
    print("🔍 Checking Google Calendar integration...")
    success = check_google_calendar()
    if success:
        print("\n✅ Google Calendar integration is working correctly!")
    else:
        print(
            "\n❌ Google Calendar integration has issues. Please fix the problems above."
        )
