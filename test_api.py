import asyncio
import httpx
import json
import sys


async def test_stock_research_api():
    """
    Simple test client for the stock research API
    """
    # Default message about Apple stock
    message = "Tell me about Apple stock and its recent performance"

    # Use command line argument if provided
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])

    print(f"Sending query: '{message}'")

    # API endpoint
    url = "http://localhost:8000/api/chat/stock-research"

    # Request payload
    payload = {"message": message, "session_id": "test-session"}

    try:
        async with httpx.AsyncClient() as client:
            # Send the request
            response = await client.post(url, json=payload)

            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()

                # Pretty print the content
                print("\n=== Response Content ===")
                if isinstance(result.get("content"), str):
                    print(result["content"])
                else:
                    print(json.dumps(result.get("content"), indent=2))

                # Show if there were any errors
                if result.get("event") == "RunError":
                    print("\n=== Error ===")
                    print(result.get("content"))
            else:
                print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error connecting to API: {str(e)}")


if __name__ == "__main__":
    asyncio.run(test_stock_research_api())
