import pandas as pd
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from groq import Groq

# ─────────────────────────────────────────────
# LOAD ENV VARIABLES
# ─────────────────────────────────────────────
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

client = Groq(api_key=groq_api_key)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("data/processed/final_dataset.csv")

texts = df["title"].dropna().tolist()

# embeddings (free)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# vector DB
vectorstore = FAISS.from_texts(texts, embeddings)
retriever = vectorstore.as_retriever()


# ─────────────────────────────────────────────
# TITLE GENERATION
# ─────────────────────────────────────────────
def generate_title(category, brand, k1, k2, extra_keywords):

    query = f"{category} {k1} {k2}"
    docs = retriever.invoke(query)

    examples = "\n".join([d.page_content for d in docs[:5]])

    all_keywords = list(set([k1, k2] + extra_keywords))

    prompt = f"""
    Generate a high-converting Amazon product title within 200 words as per Amazon guidelines.

    Category: {category}
    Brand: {brand}

    Keywords:
    {", ".join(all_keywords)}

    Example titles:
    {examples}

    Rules:
    - Include brand name
    - Include keywords naturally
    - SEO optimized
    - 12 to 18 words
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()