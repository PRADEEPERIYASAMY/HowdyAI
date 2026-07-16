import os
import requests
import json
import logging
import PyPDF2
import io
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


async def scrape_pdf_content_async(session, url):
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10.0)) as response:
            if response.status == 200:
                content_bytes = await response.read()
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content_bytes))
                num_pages = min(len(pdf_reader.pages), 10)
                content = ''
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    content += page.extract_text() or ''
                return {'content': content}
            else:
                logging.error(f"Error downloading PDF from {url}: Status code {response.status}")
                return {'error': f"Error downloading PDF: Status code {response.status}"}
    except Exception as e:
        logging.error(f"Error scraping PDF content from {url}: {e}")
        return {'error': str(e)}


async def scrape_content_async(session, url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    logger.info(f"Retrieving content from {url}")
    try:
        def fetch():
            import requests
            r = requests.get(url, timeout=8.0, headers=headers)
            r.raise_for_status()
            return r.text
        content = await asyncio.to_thread(fetch)
        return {'content': content}
    except Exception as e:
        logger.error(f"Failed to retrieve {url}: {e}")
        return {'content': '', 'error': f"Failed to retrieve {url}"}


def brave_search_engine(config, query, broad=False):
    logger.info(f"Searching Brave for: {query} (broad={broad})")
    results = []

    if not config.BRAVE_API_KEY:
        logger.error("BRAVE_API_KEY is not set in config.")
        return []

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": config.BRAVE_API_KEY
    }
    
    brave_results = []
    
    if not broad:
        # First try TAMU-scoped search for highly relevant results
        tamu_query = f"site:tamu.edu {query}"
        params = {"q": tamu_query, "count": config.NUM_SEARCH_RESULTS}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            brave_results = data.get("web", {}).get("results", [])
            logger.info(f"TAMU-scoped Brave search returned {len(brave_results)} results.")
        except Exception as e:
            logger.error(f"Brave TAMU-scoped search failed: {e}")
            
    # If explicitly broad, or fallback to general search if scoped search is insufficient
    if broad or len(brave_results) < 2:
        if not broad:
            logger.info("Falling back to broader Brave search internally.")
        try:
            params = {"q": f"Texas A&M University TAMU {query}", "count": config.NUM_SEARCH_RESULTS}
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            brave_results = data.get("web", {}).get("results", [])
        except Exception as e:
            logger.error(f"Brave broad search failed: {e}")
            return []

    if not brave_results:
        print("No search results found.")
        return []

    async def fetch_item_async(session, item):
        result_url = item.get('url', '')
        if not result_url:
            return None

        if result_url.lower().endswith('.pdf'):
            metadata = await scrape_pdf_content_async(session, result_url)
        else:
            metadata = await scrape_content_async(session, result_url)

        if 'error' in metadata:
            logger.warning(f"Scraping failed for {result_url}, falling back to Brave snippet.")
            metadata = {'content': item.get('description', '')}

        return {
            'url': result_url,
            'title': item.get('title', ''),
            'description': item.get('description', ''),
            'metadata': metadata
        }

    async def fetch_all():
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_item_async(session, item) for item in brave_results]
            return await asyncio.gather(*tasks)

    results_raw = asyncio.run(fetch_all())
    results = [r for r in results_raw if r]

    with open(os.path.join(config.BASE_PATH, "search_results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    logger.debug("Scraped results stored in search_results.json")
    return results
