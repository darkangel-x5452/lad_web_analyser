import asyncio
import json
import os
from urllib.parse import urljoin
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from aiolimiter import AsyncLimiter

from dotenv import load_dotenv
load_dotenv()  # loads .env from current directory

# Configuration
API_KEY = os.environ["BRAVE_SEARCH_API_KEY"]
API_HOST = "https://api.search.brave.com"
API_RATE_LIMIT = AsyncLimiter(1, 1)

API_PATH = {
    "web": urljoin(API_HOST, "res/v1/web/search"),
    "summarizer_search": urljoin(API_HOST, "res/v1/summarizer/search"),
}

API_HEADERS = {
    "web": {"X-Subscription-Token": API_KEY},
    "summarizer": {"X-Subscription-Token": API_KEY},
}

async def get_summary(session: ClientSession) -> None:
    query = os.environ["DEFAULT_QUERY"]  # e.g., "what is the second highest mountain"
    # Step 1: Get web search results with summary flag
    async with session.get(
        API_PATH["web"],
        params={"q": query, "summary": 1},
        headers=API_HEADERS["web"],
    ) as response:
        data = await response.json()

        if response.status != 200:
            print("Error fetching web results")
            return

    # Step 2: Extract summary key
    summary_key = data.get("summarizer", {}).get("key")

    if not summary_key:
        print("No summary available for this query")
        return

    # Step 3: Fetch the summary
    async with session.get(
        url=API_PATH["summarizer_search"],
        params={"key": summary_key, "entity_info": 1},
        headers=API_HEADERS["summarizer"],
    ) as response:
        summary_data = await response.json()
        print(json.dumps(summary_data, indent=2))

async def main():
    async with API_RATE_LIMIT:
        async with ClientSession(
            connector=TCPConnector(limit=1),
            timeout=ClientTimeout(20),
        ) as session:
            await get_summary(session=session)

asyncio.run(main())