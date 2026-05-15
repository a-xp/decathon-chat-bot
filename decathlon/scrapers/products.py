import asyncio
import httpx
import aiosqlite
import os
import json
import logging
import urllib.parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "products.db"
GRAPHQL_URL = "https://decathlon.kz/api/graphql"
PRODUCT_DETAIL_URL_TEMPLATE = "https://decathlon.kz/p/{handle}?_data=routes%2Fp.%24product"
DEFAULT_PARALLELISM = 1
DELAY = 0.5

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Referer": "https://decathlon.kz/",
    "X-Requested-With": "XMLHttpRequest"
}

GET_COLLECTION_QUERY = """
query getCollection(
    $id: ID!
    $after: String
    $first: Int
    $language: LanguageCode
) @inContext(language: $language) {
    collection(id: $id) {
        products(
            first: $first
            after: $after
        ) {
            nodes {
                id
                handle
                tags
            }
            pageInfo {
                hasNextPage
                endCursor
            }
        }
    }
}
"""

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DROP TABLE IF EXISTS products")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id TEXT PRIMARY KEY,
                handle TEXT,
                title TEXT,
                description TEXT,
                brand TEXT,
                model_code TEXT,
                price REAL,
                compare_at_price REAL,
                available INTEGER,
                image_url TEXT,
                tags TEXT,
                category_id INTEGER,
                raw_json TEXT
            )
        """)
        await db.commit()

async def fetch_full_product_details(client, supermodel, handle, semaphore):
    full_handle = f"{supermodel}_{handle}"
    encoded_handle = urllib.parse.quote(full_handle)
    url = PRODUCT_DETAIL_URL_TEMPLATE.format(handle=encoded_handle)
    
    async with semaphore:
        try:
            logger.info(f"Fetching full details for {full_handle}")
            response = await client.get(url, headers=HEADERS, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            await asyncio.sleep(DELAY)
            return data.get("product")
        except Exception as e:
            logger.error(f"Error fetching full details for {full_handle}: {e}")
            return None

async def fetch_products_for_category(client, category_id, semaphore, db, seen_products):
    gid = f"gid://shopify/Collection/{category_id}"
    has_next_page = True
    after_cursor = None
    
    while has_next_page:
        data = None
        async with semaphore:
            variables = {
                "id": gid,
                "first": 48,
                "after": after_cursor,
                "language": "RU"
            }
            payload = {
                "query": GET_COLLECTION_QUERY,
                "variables": variables,
                "operationName": "getCollection"
            }
            
            logger.info(f"Fetching product list for category {category_id} (cursor: {after_cursor})")
            try:
                response = await client.post(GRAPHQL_URL, json=payload, headers=HEADERS, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                await asyncio.sleep(DELAY)
            except Exception as e:
                logger.error(f"Error fetching product list for category {category_id}: {e}")
                break

        if not data:
            break

        collection = data.get("data", {}).get("collection")
        if not collection:
            logger.warning(f"Collection not found for {gid}")
            break

        products_data = collection.get("products", {})
        nodes = products_data.get("nodes", [])
        page_info = products_data.get("pageInfo", {})
        
        for node in nodes:
            p_id = node.get("id")
            if p_id in seen_products:
                continue
            
            handle = node.get("handle")
            tags = node.get("tags", [])
            supermodel = next((t.split(":")[1] for t in tags if t.startswith("supermodel:")), None)
            
            if not supermodel:
                logger.warning(f"No supermodel tag for product {p_id}")
                continue
            
            full_product = await fetch_full_product_details(client, supermodel, handle, semaphore)
            if not full_product:
                continue
            
            seen_products.add(p_id)
            
            title = full_product.get("title")
            description = full_product.get("description")
            brand = (full_product.get("brand") or {}).get("value")
            model_code = (full_product.get("model") or {}).get("value")
            
            price_info = full_product.get("priceRange", {}).get("minVariantPrice", {})
            price = float(price_info.get("amount", 0)) if price_info else 0.0
            
            compare_info = full_product.get("compareAtPriceRange", {}).get("minVariantPrice", {})
            compare_at_price = float(compare_info.get("amount", 0)) if compare_info else 0.0
            
            available = 1 if full_product.get("availableForSale") else 0
            image_url = (full_product.get("featuredImage") or {}).get("url")
            
            await db.execute("""
                INSERT OR REPLACE INTO products 
                (id, handle, title, description, brand, model_code, price, compare_at_price, available, image_url, tags, category_id, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                p_id, f"{supermodel}_{handle}", title, description, brand, model_code, 
                price, compare_at_price, available, image_url, 
                json.dumps(full_product.get("tags", [])), category_id, json.dumps(full_product)
            ))
        
        await db.commit()
        logger.info(f"Saved {len(nodes)} products for category {category_id}")
        
        has_next_page = page_info.get("hasNextPage")
        after_cursor = page_info.get("endCursor")

async def main():
    await init_db()
    parallelism = int(os.getenv("PARALLELISM", DEFAULT_PARALLELISM))
    semaphore = asyncio.Semaphore(parallelism)
    
    seen_products = set()
    
    async with httpx.AsyncClient() as client:
        async with aiosqlite.connect(DB_PATH) as db:
            async with db.execute("SELECT id FROM categories") as cursor:
                categories = await cursor.fetchall()
            
            logger.info(f"Found {len(categories)} categories. Starting detailed product scraping...")
            
            for (cat_id,) in categories:
                await fetch_products_for_category(client, cat_id, semaphore, db, seen_products)
    
    logger.info("Product scraping finished.")

if __name__ == "__main__":
    asyncio.run(main())
