import requests
import time
import random
import json
import csv
import argparse
import logging
import re
from bs4 import BeautifulSoup
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional
from pathlib import Path

try:
    from fake_useragent import UserAgent
    UA = UserAgent()
except ImportError:
    UA = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class Product:
    rank: int
    asin: str
    title: str
    brand: str
    price: Optional[float]
    currency: str
    rating: Optional[float]
    review_count: Optional[int]
    category: str
    subcategory: str
    badge: str
    image_url: str
    product_url: str
    scraped_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    page_number: int = 1
    title_word_count: int = 0
    has_price: bool = False
    price_tier: str = ""
    review_tier: str = ""

    def __post_init__(self):
        self.title_word_count = len(self.title.split()) if self.title else 0
        self.has_price = self.price is not None
        if self.price:
            if self.price < 500:
                self.price_tier = "budget"
            elif self.price < 2000:
                self.price_tier = "mid"
            else:
                self.price_tier = "premium"
        if self.review_count is not None:
            if self.review_count < 10:
                self.review_tier = "new"
            elif self.review_count < 100:
                self.review_tier = "low"
            elif self.review_count < 1000:
                self.review_tier = "medium"
            else:
                self.review_tier = "high"


# ─── Category Map ─────────────────────────────────────────────────────────────

CATEGORY_URLS = {
    "electronics":         "https://www.amazon.in/gp/bestsellers/electronics",
    "jewellery":           "https://www.amazon.in/gp/bestsellers/jewelry/ref=zg_bs_jewelry_sm",
    "beauty":              "https://www.amazon.in/gp/bestsellers/beauty/ref=zg_bs_beauty_sm",
    "kitchen":             "https://www.amazon.in/gp/bestsellers/kitchen/ref=zg_bs_kitchen_sm",
    "books":               "https://www.amazon.in/gp/bestsellers/books",
    "toys":                "https://www.amazon.in/gp/bestsellers/toys/ref=zg_bs_nav_toys_0",
    "clothing_women":      "https://www.amazon.in/gp/bestsellers/apparel/1968025031",
    "hair_care":           "https://www.amazon.in/gp/bestsellers/beauty/9851597031/ref=zg_bs_nav_beauty_1",
    "skincare":            "https://www.amazon.in/gp/bestsellers/beauty/1374407031/ref=zg_bs_nav_beauty_1",
    "sun_care":            "https://www.amazon.in/gp/bestsellers/beauty/1374443031/ref=zg_bs_nav_beauty_2_1374407031",
    "body_care":           "https://www.amazon.in/gp/bestsellers/beauty/1374408031/ref=zg_bs_nav_beauty_2_1374407031",
    "pet_care":            "https://www.amazon.in/gp/bestsellers/pet-supplies/ref=zg_bs_nav_pet-supplies_0",
    "household_cleaners":  "https://www.amazon.in/gp/bestsellers/hpc/1374544031/ref=zg_bs_nav_hpc_2_1374515031",
    "toilet_cleaners":     "https://www.amazon.in/gp/bestsellers/hpc/1374558031/ref=zg_bs_nav_hpc_3_1374544031",
    "food":                "https://www.amazon.in/gp/bestsellers/grocery/ref=zg_bs_nav_grocery_0",
}


# ─── HTTP Session ─────────────────────────────────────────────────────────────

def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(_random_headers())
    return session


def _random_headers() -> dict:
    user_agent = (
        UA.random if UA else
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
    return {
        "User-Agent": user_agent,
        "Accept-Language": "en-IN,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }


def fetch_page(url: str, session: requests.Session, retries: int = 3) -> Optional[str]:
    for attempt in range(retries):
        try:
            session.headers.update(_random_headers())
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                log.info(f"  Fetched [{resp.status_code}]: {url}")
                return resp.text
            elif resp.status_code == 503:
                log.warning(f"  503 — blocked. Waiting... (attempt {attempt+1})")
                time.sleep(10 + random.uniform(5, 15))
            elif resp.status_code == 404:
                log.error(f"  404 — Not found: {url}")
                return None
            else:
                log.warning(f"  Status {resp.status_code} for {url}")
        except requests.exceptions.RequestException as e:
            log.error(f"  Request error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt + random.uniform(1, 3))
    return None


# ─── Parsers ──────────────────────────────────────────────────────────────────

def parse_bsr_page(html: str, category: str, page_num: int) -> list:
    soup = BeautifulSoup(html, "lxml")
    products = []

    items = (
        soup.select("div.zg-grid-general-faceout") or
        soup.select("li.zg-item-immersion") or
        soup.select("div[data-asin]")
    )

    log.info(f"  Found {len(items)} items on page {page_num}")

    for i, item in enumerate(items, start=1):
        try:
            product = _parse_item(item, category, page_num, i)
            if product:
                products.append(product)
        except Exception as e:
            log.debug(f"  Failed item {i}: {e}")

    return products


def _parse_item(item, category: str, page_num: int, position: int) -> Optional[Product]:
    asin = item.get("data-asin") or _extract_asin_from_link(item)
    if not asin:
        return None

    rank_el = (
        item.select_one("span.zg-bdg-text") or
        item.select_one("span.p13n-sc-badge-label-substring") or
        item.select_one("div#zg-rank-list-ranked-badge")
    )
    rank_text = rank_el.get_text(strip=True) if rank_el else str(position)
    rank = _parse_rank(rank_text, position)

    title_el = (
        item.select_one("div._cDEzb_p13n-sc-css-line-clamp-3_g3dy1") or
        item.select_one("span.a-size-base.a-color-base.a-text-normal") or
        item.select_one("a.a-link-normal span") or
        item.select_one("div.p13n-sc-truncate-desktop-type2") or
        item.select_one("div[class*='p13n-sc-truncate']")
    )
    title = title_el.get_text(strip=True) if title_el else ""
    if not title:
        return None

    brand_el = (
        item.select_one("span.a-size-small.a-color-base") or
        item.select_one("div.a-row span:nth-of-type(2)") or
        item.select_one("span[class*='a-color-secondary']")
    )
    brand = brand_el.get_text(strip=True) if brand_el else _extract_brand_from_title(title)

    price, currency = _parse_price(item)

    rating_el = item.select_one("span.a-icon-alt")
    rating = None
    if rating_el:
        m = re.search(r"([\d.]+)\s+out", rating_el.get_text())
        if m:
            rating = float(m.group(1))

    review_el = (
        item.select_one("span.a-size-small") or
        item.select_one("a[href*='customerReviews'] span")
    )
    review_count = None
    if review_el:
        rc_text = review_el.get_text(strip=True).replace(",", "")
        m = re.search(r"(\d+)", rc_text)
        if m:
            review_count = int(m.group(1))

    badge_el = item.select_one("span.a-badge-text")
    badge = badge_el.get_text(strip=True) if badge_el else ""

    img_el = item.select_one("img.a-dynamic-image") or item.select_one("img")
    image_url = img_el.get("src", "") if img_el else ""

    link_el = item.select_one("a.a-link-normal")
    href = link_el.get("href", "") if link_el else ""
    product_url = f"https://www.amazon.in{href}" if href.startswith("/") else href

    parts = category.replace("_", " ").title().split(" > ")
    cat_path = category.replace("_", " ").title()
    subcategory = parts[-1] if parts else category

    return Product(
        rank=rank, asin=asin, title=title, brand=brand,
        price=price, currency=currency, rating=rating,
        review_count=review_count, category=cat_path,
        subcategory=subcategory, badge=badge,
        image_url=image_url, product_url=product_url,
        page_number=page_num,
    )


def _extract_asin_from_link(item) -> Optional[str]:
    link = item.select_one("a[href*='/dp/']")
    if link:
        m = re.search(r"/dp/([A-Z0-9]{10})", link.get("href", ""))
        if m:
            return m.group(1)
    return None


def _parse_rank(text: str, fallback: int) -> int:
    cleaned = re.sub(r"[^\d]", "", text)
    return int(cleaned) if cleaned else fallback


def _parse_price(item) -> tuple:
    price_el = (
        item.select_one("span._cDEzb_p13n-sc-price_3mJ9Z") or
        item.select_one("span.p13n-sc-price") or
        item.select_one("span.a-price-whole") or
        item.select_one("span[class*='price']")
    )
    if not price_el:
        return None, "INR"
    text = price_el.get_text(strip=True).replace(",", "").replace("\xa0", "")
    currency = "INR" if "₹" in text or "Rs" in text else "USD"
    m = re.search(r"[\d.]+", text)
    return (float(m.group()), currency) if m else (None, currency)


def _extract_brand_from_title(title: str) -> str:
    words = title.split()
    return words[0] if words else ""


def scrape_category_tree(html: str) -> list:
    soup = BeautifulSoup(html, "lxml")
    categories = []
    nav_items = soup.select("ul.a-nostyle li") or soup.select("#zg_browseRoot li")
    for item in nav_items:
        link = item.select_one("a")
        if link:
            categories.append({
                "name": link.get_text(strip=True),
                "url": "https://www.amazon.in" + link.get("href", ""),
                "is_selected": "bold" in item.get("class", []) or item.select_one("span.zg_selected") is not None,
            })
    return categories


def extract_keywords_from_products(products: list) -> list:
    STOPWORDS = {
        "with", "for", "and", "the", "in", "of", "a", "an", "to",
        "is", "by", "on", "at", "from", "as", "-", "&", "|", "/",
        "pack", "set", "piece", "pcs", "nos", "no", "new", "free",
        "black", "white", "red", "blue", "green", "yellow", "pink",
        "purple", "grey", "gray",
    }
    keyword_data = {}

    for product in products:
        title = product.title.lower()
        tokens = re.findall(r"[a-z][a-z0-9\-']+", title)
        bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]
        all_terms = [t for t in tokens + bigrams if t not in STOPWORDS and len(t) > 2]

        for term in all_terms:
            if term not in keyword_data:
                keyword_data[term] = {
                    "keyword": term, "frequency": 0,
                    "avg_rank": 0, "total_rank": 0,
                    "categories": set(), "avg_rating": 0,
                    "total_rating": 0, "avg_reviews": 0,
                    "total_reviews": 0, "products_count": 0,
                }
            kd = keyword_data[term]
            kd["frequency"] += 1
            kd["total_rank"] += product.rank
            kd["categories"].add(product.category)
            if product.rating:
                kd["total_rating"] += product.rating
                kd["products_count"] += 1
            if product.review_count:
                kd["total_reviews"] += product.review_count

    result = []
    for kd in keyword_data.values():
        count = kd["frequency"]
        kd["avg_rank"] = round(kd["total_rank"] / count, 1)
        kd["avg_rating"] = round(kd["total_rating"] / kd["products_count"], 2) if kd["products_count"] else 0
        kd["avg_reviews"] = round(kd["total_reviews"] / count, 0)
        kd["category_count"] = len(kd["categories"])
        kd["categories"] = list(kd["categories"])
        del kd["total_rank"], kd["total_rating"], kd["total_reviews"], kd["products_count"]
        result.append(kd)

    result.sort(key=lambda x: x["frequency"], reverse=True)
    return result


# ─── Single-category scraper ──────────────────────────────────────────────────

def scrape_bsr(category: str, pages: int = 2, delay: tuple = (3, 7)) -> dict:
    if category.startswith("http"):
        base_url = category
        cat_name = "custom"
    else:
        base_url = CATEGORY_URLS.get(category)
        cat_name = category
        if not base_url:
            raise ValueError(f"Unknown category '{category}'. Options: {list(CATEGORY_URLS.keys())}")

    session = make_session()
    all_products = []
    category_tree = []

    for page in range(1, pages + 1):
        url = base_url if page == 1 else f"{base_url}?pg={page}"
        log.info(f"  Page {page}/{pages}: {url}")

        html = fetch_page(url, session)
        if not html:
            log.warning(f"  Skipping page {page} — fetch failed")
            continue

        products = parse_bsr_page(html, cat_name, page)
        all_products.extend(products)

        if page == 1:
            category_tree = scrape_category_tree(html)

        wait = random.uniform(*delay)
        log.info(f"  Waiting {wait:.1f}s...")
        time.sleep(wait)

    keywords = extract_keywords_from_products(all_products)

    return {
        "products": [asdict(p) for p in all_products],
        "keywords": keywords[:200],
        "category_tree": category_tree,
        "metadata": {
            "category": cat_name,
            "pages_scraped": pages,
            "total_products": len(all_products),
            "total_keywords": len(keywords),
            "scraped_at": datetime.utcnow().isoformat(),
        }
    }


# ─── NEW: Scrape ALL categories ───────────────────────────────────────────────

def scrape_all_categories(
    pages: int = 2,
    delay: tuple = (3, 7),
    between_categories: tuple = (5, 12),
    output_dir: str = "data/raw",
    categories: list = None,           # pass a subset, or None = all
) -> dict:
    """
    Loop over every category in CATEGORY_URLS (or a subset) and scrape them
    one by one.  Saves per-category files immediately so you don't lose data
    if a later category fails.

    Returns a combined summary dict.
    """
    target_cats = categories if categories else list(CATEGORY_URLS.keys())
    total = len(target_cats)

    combined_products = []
    combined_keywords = {}
    summary = []

    print(f"\n{'='*65}")
    print(f"  Scraping {total} categories  |  {pages} page(s) each")
    print(f"{'='*65}\n")

    for idx, cat in enumerate(target_cats, start=1):
        print(f"\n[{idx}/{total}] ── Category: {cat.upper()} ──")

        try:
            data = scrape_bsr(cat, pages=pages, delay=delay)
            save_results(data, output_dir=output_dir)

            combined_products.extend(data["products"])

            # Merge keywords — accumulate frequencies across categories
            for kw in data["keywords"]:
                key = kw["keyword"]
                if key not in combined_keywords:
                    combined_keywords[key] = kw.copy()
                    combined_keywords[key]["categories"] = set(kw["categories"])
                else:
                    combined_keywords[key]["frequency"]    += kw["frequency"]
                    combined_keywords[key]["total_rank"]    = (
                        combined_keywords[key].get("avg_rank", 0) +
                        kw.get("avg_rank", 0)
                    )
                    combined_keywords[key]["categories"].update(kw["categories"])

            summary.append({
                "category":       cat,
                "status":         "ok",
                "products":       data["metadata"]["total_products"],
                "keywords":       data["metadata"]["total_keywords"],
                "pages_scraped":  data["metadata"]["pages_scraped"],
            })

            print(f"  -> {data['metadata']['total_products']} products, "
                  f"{data['metadata']['total_keywords']} keywords")

        except Exception as e:
            log.error(f"  Category '{cat}' failed: {e}")
            summary.append({"category": cat, "status": f"error: {e}",
                             "products": 0, "keywords": 0, "pages_scraped": 0})

        # Polite gap between categories (skip after last one)
        if idx < total:
            gap = random.uniform(*between_categories)
            log.info(f"  Cooling down {gap:.1f}s before next category...")
            time.sleep(gap)

    # Finalise combined keywords list
    for kd in combined_keywords.values():
        if isinstance(kd.get("categories"), set):
            kd["categories"] = list(kd["categories"])
            kd["category_count"] = len(kd["categories"])

    combined_kw_list = sorted(
        combined_keywords.values(),
        key=lambda x: x["frequency"],
        reverse=True
    )

    # Save master combined files
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    master_json = f"{output_dir}/ALL_CATEGORIES_{ts}.json"
    with open(master_json, "w") as f:
        json.dump({
            "products":  combined_products,
            "keywords":  combined_kw_list[:500],
            "summary":   summary,
            "metadata": {
                "total_categories": total,
                "total_products":   len(combined_products),
                "total_keywords":   len(combined_kw_list),
                "scraped_at":       datetime.utcnow().isoformat(),
            }
        }, f, indent=2, default=str)
    log.info(f"Saved master JSON: {master_json}")

    # Master products CSV
    if combined_products:
        master_csv = f"{output_dir}/ALL_CATEGORIES_{ts}_products.csv"
        with open(master_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=combined_products[0].keys())
            writer.writeheader()
            writer.writerows(combined_products)
        log.info(f"Saved master CSV: {master_csv}")

    # Master keywords CSV
    if combined_kw_list:
        kw_csv = f"{output_dir}/ALL_CATEGORIES_{ts}_keywords.csv"
        with open(kw_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=combined_kw_list[0].keys())
            writer.writeheader()
            writer.writerows(combined_kw_list[:500])
        log.info(f"Saved master keywords CSV: {kw_csv}")

    # Print final summary table
    print(f"\n{'='*65}")
    print(f"  {'CATEGORY':<25} {'STATUS':<10} {'PRODUCTS':>8} {'KEYWORDS':>9}")
    print(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*9}")
    for row in summary:
        status_icon = "ok" if row["status"] == "ok" else "FAIL"
        print(f"  {row['category']:<25} {status_icon:<10} "
              f"{row['products']:>8} {row['keywords']:>9}")
    print(f"  {'─'*55}")
    print(f"  {'TOTAL':<25} {'':10} {len(combined_products):>8} "
          f"{len(combined_kw_list):>9}")
    print(f"{'='*65}\n")

    return {"summary": summary, "master_file": master_json}


# ─── Save results ─────────────────────────────────────────────────────────────

def save_results(data: dict, output_dir: str = "data/raw"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cat = data["metadata"]["category"]
    ts  = datetime.utcnow().strftime("%Y%m%d_%H%M")

    json_path = f"{output_dir}/{cat}_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    if data["products"]:
        csv_path = f"{output_dir}/{cat}_{ts}_products.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=data["products"][0].keys())
            writer.writeheader()
            writer.writerows(data["products"])

    if data["keywords"]:
        kw_path = f"{output_dir}/{cat}_{ts}_keywords.csv"
        with open(kw_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=data["keywords"][0].keys())
            writer.writeheader()
            writer.writerows(data["keywords"])

    return json_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Amazon BSR Scraper — all categories or one")
    parser.add_argument(
        "--category", default="all",
        help=(
            f"'all' to scrape every category (default), "
            f"or one of: {list(CATEGORY_URLS.keys())}, "
            f"or a full URL"
        )
    )
    parser.add_argument("--pages",      type=int,   default=2,   help="BSR pages per category (50 products each)")
    parser.add_argument("--output",     default="data/raw",       help="Output directory")
    parser.add_argument("--delay-min",  type=float, default=3.0,  help="Min seconds between page requests")
    parser.add_argument("--delay-max",  type=float, default=7.0,  help="Max seconds between page requests")
    parser.add_argument("--gap-min",    type=float, default=5.0,  help="Min seconds between categories")
    parser.add_argument("--gap-max",    type=float, default=12.0, help="Max seconds between categories")
    args = parser.parse_args()

    delay = (args.delay_min, args.delay_max)
    gap   = (args.gap_min, args.gap_max)

    if args.category == "all":
        # ── Scrape every category ──────────────────────────────────────────
        scrape_all_categories(
            pages=args.pages,
            delay=delay,
            between_categories=gap,
            output_dir=args.output,
        )
    else:
        # ── Single category (original behaviour) ──────────────────────────
        data = scrape_bsr(args.category, pages=args.pages, delay=delay)
        save_results(data, output_dir=args.output)

        print(f"\n{'='*60}")
        print(f"  Category : {data['metadata']['category']}")
        print(f"  Products : {data['metadata']['total_products']}")
        print(f"  Keywords : {data['metadata']['total_keywords']}")
        print(f"\n  Top 10 Products:")
        for p in data["products"][:10]:
            print(f"    #{p['rank']:>3}  {p['title'][:55]:<55}  ₹{p['price'] or '?'}")
        print(f"\n  Top 20 Keywords:")
        for kw in data["keywords"][:20]:
            print(f"    {kw['keyword']:<30} freq={kw['frequency']:>3}  avg_rank={kw['avg_rank']:>6}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()