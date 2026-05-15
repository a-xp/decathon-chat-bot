import asyncio
import httpx
import aiosqlite
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_CATEGORIES = [
    "1-vidy-sporta",
    "520-muzhchiny",
    "583-zhenshinam",
    "650-detyam",
    "697-aksessuary"
]

DB_PATH = "products.db"
BASE_URL = "https://decathlon.kz/"
API_SUFFIX = "?_data=routes%2F%24category"
DEFAULT_PARALLELISM = 1
TIMEOUT = 30.0
DELAY = 0.5

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://decathlon.kz/",
    "X-Requested-With": "XMLHttpRequest"
}

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY,
                name TEXT,
                slug TEXT UNIQUE,
                parent_id INTEGER,
                level INTEGER
            )
        """)
        await db.commit()

async def fetch_category(client, slug, semaphore, db, parent_id=None, level=0, seen=None):
    if seen is None:
        seen = set()
    
    if slug in seen:
        return
    seen.add(slug)

    data = None
    # Narrow semaphore scope to just the network request
    async with semaphore:
        url = f"{BASE_URL}{slug}{API_SUFFIX}"
        logger.info(f"Fetching category: {slug} (level {level})")
        try:
            # Use a shorter inner timeout just in case, but keep global TIMEOUT
            response = await client.get(url, timeout=TIMEOUT, headers=HEADERS)
            logger.info(f"Response status for {slug}: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            await asyncio.sleep(DELAY)
        except Exception as e:
            logger.error(f"Error fetching {slug}: {e}")
            return

    if not data:
        return

    cat_data = data.get("category", {})
    if not cat_data:
        logger.warning(f"No category data for {slug}")
        return

    cat_id = cat_data.get("id")
    name = cat_data.get("name")
    
    # Save to DB
    try:
        await db.execute(
            "INSERT OR REPLACE INTO categories (id, name, slug, parent_id, level) VALUES (?, ?, ?, ?, ?)",
            (cat_id, name, slug, parent_id, level)
        )
        await db.commit()
    except Exception as e:
        logger.error(f"Error saving {slug} to DB: {e}")

    # Recurse
    child_categories = cat_data.get("child_categories", [])
    if child_categories:
        logger.info(f"Category {slug} has {len(child_categories)} children. Starting recursion...")
    
    tasks = []
    for child in child_categories:
        child_slug = child.get("link_rewrite")
        if child_slug:
            tasks.append(fetch_category(client, child_slug, semaphore, db, cat_id, level + 1, seen))
    
    if tasks:
        await asyncio.gather(*tasks)

async def main():
    await init_db()
    parallelism = int(os.getenv("PARALLELISM", DEFAULT_PARALLELISM))
    semaphore = asyncio.Semaphore(parallelism)
    
    seen = set()
    async with httpx.AsyncClient(follow_redirects=True) as client:
        async with aiosqlite.connect(DB_PATH) as db:
            logger.info(f"Starting category scraping with parallelism={parallelism}...")
            tasks = [fetch_category(client, slug, semaphore, db, seen=seen) for slug in ROOT_CATEGORIES]
            await asyncio.gather(*tasks)
    
    logger.info("Category scraping finished.")

if __name__ == "__main__":
    asyncio.run(main())
