import pandas as pd
import joblib
from ragengine import generate_title

df = pd.read_csv("data/processed/final_dataset.csv")
model = joblib.load("models/best_model.pkl")

STOPWORDS = ["with","and","for","the","to","from","up","hours","into","your"]

def suggest_keywords(category, keyword):

    subset = df[
        df["category"].str.lower() == category.lower()
    ]

    # explode keywords
    temp = subset.explode("keywords_list")

    # filter by input keyword similarity
    temp = temp[
        temp["keywords_list"].str.contains(keyword, na=False)
    ]

    if len(temp) == 0:
        # fallback: return top keywords in category
        return (
            subset.explode("keywords_list")["keywords_list"]
            .value_counts()
            .head(10)
            .index
            .tolist()
        )

    return (
        temp["keywords_list"]
        .value_counts()
        .head(10)
        .index
        .tolist()
    )

def get_competition(category, keyword):

    subset = df[
        (df["category"] == category.lower()) &
        (df["title"].str.contains(keyword, na=False))
    ]

    if len(subset) == 0:
        return 0.3, "LOW"

    score = subset["competition_score"].mean()

    level = "HIGH" if score > 0.7 else "MEDIUM" if score > 0.3 else "LOW"

    return round(score, 2), level


def predict_bid(category, keyword):

    subset = df[
        (df["category"] == category.lower()) &
        (df["title"].str.contains(keyword, na=False))
    ]

    if len(subset) == 0:
        return 10.0

    row = subset.iloc[[0]]

    return round(model.predict(row)[0], 2)


# 🔥 FINAL PIPELINE
def get_output(category, brand, k1, k2):

    suggested = suggest_keywords(category, k1)

    score, level = get_competition(category, k1)
    bid = predict_bid(category, k1)

    title = generate_title(category, brand, k1, k2, suggested)

    return {
        "title": title,
        "keywords": suggested,
        "competition": level,
        "score": score,
        "bid": f"₹{bid}"
    }