import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# KEYWORD EXTRACTION
# ─────────────────────────────────────────────

STOPWORDS = [
    "with","and","for","the","to","from","up","hours","into",
    "your","pack","set","new","best","of","in","on","a"
]

def extract_keywords(text):
    words = str(text).lower().split()
    return [
        w for w in words
        if len(w) > 3 and w not in STOPWORDS
    ]


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_data():
    products = pd.read_csv("data/raw/ALL_CATEGORIES_20260324_1313_products.csv")
    keywords = pd.read_csv("data/raw/ALL_CATEGORIES_20260324_1313_keywords.csv")
    return products, keywords


# ─────────────────────────────────────────────
# CLEAN
# ─────────────────────────────────────────────

def clean(products, keywords):

    products["title"] = products["title"].astype(str).str.lower()
    products["category"] = products["category"].astype(str).str.lower()
    keywords["keyword"] = keywords["keyword"].astype(str).str.lower()

    for col in ["price", "rating", "review_count", "rank"]:
        products[col] = pd.to_numeric(products[col], errors="coerce")

    for col in ["frequency", "avg_rank", "avg_rating", "avg_reviews"]:
        keywords[col] = pd.to_numeric(keywords[col], errors="coerce")

    return products, keywords


# ─────────────────────────────────────────────
# MERGE (IMPROVED)
# ─────────────────────────────────────────────

def merge(products, keywords):

    # ❌ REMOVE OLD MATCHING
    # products["keyword"] = products["title"].apply(match)

    # ✅ NEW: extract keywords from title
    products["keywords_list"] = products["title"].apply(extract_keywords)

    # explode for better merging
    temp = products.explode("keywords_list")

    # merge with keyword stats
    temp = temp.merge(
        keywords,
        left_on="keywords_list",
        right_on="keyword",
        how="left"
    )

    return temp


# ─────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────

def features(df):

    df["title_len"] = df["title"].apply(lambda x: len(str(x).split()))
    df["log_reviews"] = np.log1p(df["review_count"].fillna(0))

    df["competition_score"] = df["avg_reviews"] / (df["review_count"] + 1)
    df["competition_score"] = df["competition_score"].clip(0, 1)

    df["suggested_bid"] = (
        df["avg_reviews"].fillna(0) / 1000 +
        df["avg_rating"].fillna(0) * 2
    )

    return df


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

def run():

    products, keywords = load_data()
    products, keywords = clean(products, keywords)

    df = merge(products, keywords)
    df = features(df)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/final_dataset.csv", index=False)

    print("✅ Saved → data/processed/final_dataset.csv")


if __name__ == "__main__":
    run()