import pandas as pd
import joblib
from ragengine import generate_title

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

df = pd.read_csv("data/processed/final_dataset.csv")
model = joblib.load("models/best_model.pkl")


# ─────────────────────────────────────────────
# HELPER: FILTER DATA
# ─────────────────────────────────────────────

def filter_subset(category, keyword):
    category = category.strip().lower()
    keyword = keyword.strip().lower()

    subset = df[
        df["category"].astype(str).str.strip().str.lower() == category
    ]

    # fuzzy match
    subset_kw = subset[
        subset["title"].astype(str).str.contains(keyword, case=False, na=False)
    ]

    # fallback if no match
    if len(subset_kw) == 0:
        return subset

    return subset_kw


# ─────────────────────────────────────────────
# KEYWORD SUGGESTION
# ─────────────────────────────────────────────

def suggest_keywords(category, keyword):

    subset = filter_subset(category, keyword)

    if "keywords_list" not in subset.columns:
        return ["no keywords"]

    temp = subset.explode("keywords_list")

    keywords = (
        temp["keywords_list"]
        .dropna()
        .astype(str)
        .value_counts()
        .head(10)
        .index
        .tolist()
    )

    return keywords


# ─────────────────────────────────────────────
# COMPETITION SCORE
# ─────────────────────────────────────────────

def get_competition(category, keyword):

    subset = filter_subset(category, keyword)

    if len(subset) == 0:
        return 0.3, "LOW"

    # weighted competition
    score = (
        subset["competition_score"].fillna(0).mean() * 0.6 +
        (subset["review_count"].fillna(0).mean() / 100000) * 0.4
    )

    score = min(score, 1.0)

    if score > 0.7:
        level = "HIGH"
    elif score > 0.4:
        level = "MEDIUM"
    else:
        level = "LOW"

    return round(score, 2), level


# ─────────────────────────────────────────────
# BID PREDICTION
# ─────────────────────────────────────────────

def predict_bid(category, keyword):

    subset = filter_subset(category, keyword)

    if len(subset) == 0:
        return 10.0

    # aggregate features
    row = subset.mean(numeric_only=True).to_frame().T

    features = [
        "price", "rating", "review_count",
        "frequency", "avg_rank", "avg_rating", "avg_reviews",
        "title_len", "log_reviews"
    ]

    # handle missing columns safely
    for col in features:
        if col not in row.columns:
            row[col] = 0

    row = row[features]

    try:
        pred = model.predict(row)[0]
        return round(float(pred), 2)
    except:
        return 10.0


# ─────────────────────────────────────────────
# FINAL PIPELINE
# ─────────────────────────────────────────────

def get_output(category, brand, k1, k2):

    suggested = suggest_keywords(category, k1)

    score, level = get_competition(category, k1)

    bid = predict_bid(category, k1)

    # title generation (RAG + LLM)
    try:
        title = generate_title(category, brand, k1, k2, suggested)
    except:
        title = f"{brand} {k1} {k2} best quality product"

    return {
        "title": title,
        "keywords": suggested,
        "competition": level,
        "score": score,
        "bid": f"₹{bid}"
    }