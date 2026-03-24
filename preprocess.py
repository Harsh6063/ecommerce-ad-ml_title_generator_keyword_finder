import pandas as pd
import numpy as np
from pathlib import Path


def load_data():
    products = pd.read_csv("data/raw/ALL_CATEGORIES_20260324_1313_products.csv")
    keywords = pd.read_csv("data/raw/ALL_CATEGORIES_20260324_1313_keywords.csv")
    return products, keywords


def clean(products, keywords):
    products["title"] = products["title"].astype(str).str.lower()
    products["category"] = products["category"].astype(str).str.lower()
    keywords["keyword"] = keywords["keyword"].astype(str).str.lower()

    for col in ["price", "rating", "review_count", "rank"]:
        products[col] = pd.to_numeric(products[col], errors="coerce")

    for col in ["frequency", "avg_rank", "avg_rating", "avg_reviews"]:
        keywords[col] = pd.to_numeric(keywords[col], errors="coerce")

    return products, keywords


def merge(products, keywords):
    def match(title):
        for kw in keywords["keyword"]:
            if kw in title:
                return kw
        return None

    products["keyword"] = products["title"].apply(match)
    df = products.merge(keywords, on="keyword", how="left")
    return df


def features(df):
    df["title_len"] = df["title"].apply(lambda x: len(str(x).split()))
    df["log_reviews"] = np.log1p(df["review_count"].fillna(0))

    df["competition_score"] = df["avg_reviews"] / (df["review_count"] + 1)
    df["competition_score"] = df["competition_score"].clip(0, 1)

    df["suggested_bid"] = (
        df["avg_reviews"].fillna(0) / 1000 + df["avg_rating"].fillna(0) * 2
    )

    return df


def run():
    p, k = load_data()
    p, k = clean(p, k)
    df = merge(p, k)
    df = features(df)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/final_dataset.csv", index=False)

    print("Saved → data/processed/final_dataset.csv")


if __name__ == "__main__":
    run()