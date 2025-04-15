# simpleAgents

# local postresql

docker run -d -e POSTGRES_DB=ai -e POSTGRES_USER=ai -e POSTGRES_PASSWORD=ai -e PGDATA=/var/lib/postgresql/data/pgdata -v pgvolume:/var/lib/postgresql/data -p 5532:5432 --name pgvector agnohq/pgvector:16

docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant

REST API: localhost:6333
Web UI: localhost:6333/dashboard
GRPC API: localhost:6334

## Stock Research Team API

### Error with Async Iteration

If you encounter this error with the stock research team: 
```
'async for' requires an object with __aiter__ method, got RunResponse
```

Use the safe API we've created to prevent this error.

### Running the API Server

To run the API server:

```bash
python api.py
```

This starts a FastAPI server on port 8000.

### API Endpoints

#### Stock Research Team Chat

```
POST /api/chat/stock-research
```

Example request:

```json
{
  "message": "Tell me about Apple stock",
  "session_id": "optional-session-id"
}
```

This safely handles the async iteration issue by using a wrapper function that sets `stream=False` and handles any errors that might occur.

### Programmatic Usage

You can also use the safe wrapper function directly in your code:

```python
from teams.stock_research_team import safe_team_run

async def my_function():
    response = await safe_team_run("Tell me about Apple stock")
    print(response.content)
```

This approach avoids the async iteration error by ensuring stream is disabled and properly handling any exceptions.

## Testing the API

We've provided a simple test client to verify that the API is working correctly:

1. First, start the API server:
   ```bash
   python api.py
   ```

2. In a separate terminal, run the test client:
   ```bash
   python test_api.py
   ```
   
   You can also specify a custom query:
   ```bash
   python test_api.py Tell me about Tesla stock
   ```

The test client will send a request to the API and display the response.

## Troubleshooting

If you encounter any issues with the API, check the server logs for detailed error messages. The most common issues are:

1. The `'async for' requires an object with __aiter__ method, got RunResponse` error that our solution addresses
2. Network connectivity issues
3. Missing dependencies - make sure you have all required packages installed

## Extending the Solution

To adapt this solution for other teams or agents, follow the same pattern:

1. Create a safe wrapper function similar to `safe_team_run` for each team
2. Set `stream=False` to avoid async iteration issues
3. Add proper error handling
4. Use the wrapper in your API endpoints