# crawlers/api_crawler.py

"""
=========================================================
UNIFIED API CRAWLER
=========================================================

Sources:
- NewsAPI
- Tiingo
- UK Companies House
- Japan EDINET

Pipeline:
company_queue
    ↓
API Crawlers
    ↓
scraped_sentences

=========================================================
"""

import os
import time
import requests
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")

from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

from database.supabase_client import supabase

# =========================================================
# ENV
# =========================================================

load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
COMPANIES_HOUSE_API_KEY = os.getenv("COMPANIES_HOUSE_API_KEY")

MAX_DEPTH = 3
NEWS_LIMIT = 5

HEADERS = {
    "User-Agent": "SupplyChainNLPBot/1.0"
}

# =========================================================
# NLTK
# =========================================================

nltk.download("punkt")

# =========================================================
# HELPERS
# =========================================================

def split_sentences(text):

    if not text:
        return []

    return nltk.sent_tokenize(text)


def is_valid_sentence(sentence):

    if not sentence:
        return False

    sentence = sentence.strip()

    if len(sentence) < 40:
        return False

    keywords = [
        "supplier",
        "supplies",
        "agreement",
        "manufacturing",
        "partner",
        "ownership",
        "subsidiary",
        "acquired",
        "factory",
        "semiconductor",
        "procurement",
        "contract"
    ]

    lowered = sentence.lower()

    return any(
        keyword in lowered
        for keyword in keywords
    )


# =========================================================
# DATABASE INSERT
# =========================================================

def push_sentence(
    source_company,
    source_url,
    document_type,
    raw_sentence,
    accession_number=None,
    filing_date=None
):

    try:

        response = (
            supabase
            .table("scraped_sentences")
            .insert({

                # =============================================
                # RAW DATA
                # =============================================

                "source_company": source_company,

                "source_url": source_url,

                "document_type": document_type,

                "raw_sentence": raw_sentence,

                # =============================================
                # NLP PLACEHOLDERS
                # =============================================

                "masked_sentence": None,

                "extracted_entities": None,

                "llm_processed": False,

                # =============================================
                # RELATION EXTRACTION PLACEHOLDERS
                # =============================================

                "relation_id": None,

                "relation_type": None,

                "entity_from": None,

                "entity_to": None,

                "confidence_score": None,

                "reasoning": None,

                # =============================================
                # FILING METADATA
                # =============================================

                "accession_number": accession_number,

                "filing_date": filing_date

            })
            .execute()
        )

        print(
            f"✅ Stored sentence for {source_company}"
        )

        return response

    except Exception as e:

        print(f"❌ Insert failed: {e}")


# =========================================================
# COMPANY QUEUE
# =========================================================

def load_company_queue():

    response = (
        supabase
        .table("company_queue")
        .select("*")
        .lte("tier_level", MAX_DEPTH)
        .execute()
    )

    return response.data


# =========================================================
# NEWSAPI
# =========================================================

def crawl_newsapi(company_name):

    print(f"\n📰 NewsAPI → {company_name}")

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": company_name,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": NEWS_LIMIT,
        "apiKey": NEWSAPI_KEY
    }

    try:

        response = requests.get(
            url,
            params=params,
            headers=HEADERS
        )

        response.raise_for_status()

        data = response.json()

        articles = data.get("articles", [])

        for article in articles:

            text = " ".join([
                str(article.get("title", "")),
                str(article.get("description", "")),
                str(article.get("content", ""))
            ])

            sentences = split_sentences(text)

            for sentence in sentences:

                if is_valid_sentence(sentence):

                    push_sentence(
                        source_company=company_name,

                        source_url=article.get("url"),

                        document_type="NEWS",

                        raw_sentence=sentence
                    )

        print("✅ NewsAPI complete")

    except Exception as e:

        print(f"❌ NewsAPI error: {e}")


# =========================================================
# TIINGO
# =========================================================

def crawl_tiingo(company_name):

    print(f"\n📈 Tiingo → {company_name}")

    url = "https://api.tiingo.com/tiingo/news"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Token {TIINGO_API_KEY}"
    }

    params = {
        "query": company_name,
        "limit": NEWS_LIMIT
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            params=params
        )

        response.raise_for_status()

        articles = response.json()

        for article in articles:

            text = " ".join([
                str(article.get("title", "")),
                str(article.get("description", "")),
                str(article.get("content", ""))
            ])

            sentences = split_sentences(text)

            for sentence in sentences:

                if is_valid_sentence(sentence):

                    push_sentence(
                        source_company=company_name,

                        source_url=article.get("url"),

                        document_type="NEWS",

                        raw_sentence=sentence
                    )

        print("✅ Tiingo complete")

    except Exception as e:

        print(f"❌ Tiingo error: {e}")


# =========================================================
# UK COMPANIES HOUSE
# =========================================================

def crawl_companies_house(company_name):

    print(f"\n🏢 Companies House → {company_name}")

    url = "https://api.company-information.service.gov.uk/search/companies"

    params = {
        "q": company_name,
        "items_per_page": 5
    }

    try:

        response = requests.get(
            url,
            params=params,
            auth=HTTPBasicAuth(
                COMPANIES_HOUSE_API_KEY,
                ""
            )
        )

        response.raise_for_status()

        data = response.json()

        items = data.get("items", [])

        for item in items:

            sic_codes = item.get("sic_codes", [])

            valid = any(
                str(code).startswith("26")
                for code in sic_codes
            )

            if not valid:
                continue

            text = " ".join([
                str(item.get("title", "")),
                str(item.get("snippet", "")),
                str(item.get("company_status", ""))
            ])

            sentences = split_sentences(text)

            for sentence in sentences:

                if is_valid_sentence(sentence):

                    push_sentence(
                        source_company=company_name,

                        source_url=(
                            "https://find-and-update.company-information.service.gov.uk/"
                        ),

                        document_type="UK",

                        raw_sentence=sentence
                    )

        print("✅ Companies House complete")

    except Exception as e:

        print(f"❌ Companies House error: {e}")


# =========================================================
# EDINET
# =========================================================

def crawl_edinet():

    print("\n🇯🇵 EDINET")

    url = (
        "https://disclosure.edinet-fsa.go.jp/api/v2/documents.json"
    )

    params = {
        "date": "2026-05-22",
        "type": 2
    }

    try:

        response = requests.get(
            url,
            params=params,
            headers=HEADERS
        )

        response.raise_for_status()

        data = response.json()

        filings = data.get("results", [])[:10]

        for filing in filings:

            text = " ".join([
                str(filing.get("filerName", "")),
                str(filing.get("docDescription", ""))
            ])

            sentences = split_sentences(text)

            for sentence in sentences:

                if is_valid_sentence(sentence):

                    push_sentence(
                        source_company=filing.get("filerName"),

                        source_url=(
                            "https://disclosure.edinet-fsa.go.jp/"
                        ),

                        document_type="JP",

                        raw_sentence=sentence,

                        accession_number=filing.get("docID"),

                        filing_date=filing.get(
                            "submitDateTime"
                        )
                    )

        print("✅ EDINET complete")

    except Exception as e:

        print(f"❌ EDINET error: {e}")


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_pipeline():

    companies = load_company_queue()

    print(
        f"\n🚀 Loaded {len(companies)} companies"
    )

    for company in companies:

        company_name = company["company_name"]

        print("\n================================================")
        print(f"PROCESSING: {company_name}")
        print("================================================")

        crawl_newsapi(company_name)

        time.sleep(2)

        crawl_tiingo(company_name)

        time.sleep(2)

        crawl_companies_house(company_name)

        time.sleep(2)

    crawl_edinet()

    print("\n✅ PIPELINE COMPLETE")


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    run_pipeline()