# =========================================================
# crawlers/newsapi_crawler.py
# =========================================================

import os
import hashlib
import requests

from dotenv import load_dotenv

from utils.text_utils import (
    split_sentences,
    is_valid_sentence
)

from utils.stats import increment

from utils.push_sentence import (
    push_sentence
)

# =========================================================
# ENV
# =========================================================

load_dotenv()

NEWSAPI_KEY = os.getenv(
    "NEWSAPI_KEY"
)

HEADERS = {
    "User-Agent": "SupplyChainNLPBot/1.0"
}

NEWS_LIMIT = 10

SEEN_HASHES = set()

# =========================================================
# HASHING
# =========================================================

def generate_hash(text):

    return hashlib.md5(
        text.encode("utf-8")
    ).hexdigest()

# =========================================================
# MAIN CRAWLER
# =========================================================

def crawl_newsapi(company_name):

    print(
        f"\n📰 NewsAPI → "
        f"{company_name}"
    )

    url = (
        "https://newsapi.org/v2/everything"
    )

    params = {

        "q": (
            f'"{company_name}" AND '
            '('
            'supplier OR '
            'manufacturing OR '
            'semiconductor OR '
            'partner OR '
            'procurement OR '
            'factory OR '
            'supply chain'
            ')'
        ),

        "language": "en",

        "sortBy": "publishedAt",

        "pageSize": NEWS_LIMIT,

        "apiKey": NEWSAPI_KEY
    }

    try:

        # =====================================
        # REQUEST
        # =====================================

        increment(
            "newsapi_requests"
        )

        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        articles = data.get(
            "articles",
            []
        )

        increment(
            "articles_found",
            len(articles)
        )

        print(
            f"📰 Found "
            f"{len(articles)} articles"
        )

        # =====================================
        # PROCESS ARTICLES
        # =====================================

        for article in articles:

            increment(
                "articles_processed"
            )

            text = " ".join([

                str(article.get(
                    "title",
                    ""
                )),

                str(article.get(
                    "description",
                    ""
                )),

                str(article.get(
                    "content",
                    ""
                ))
            ])

            sentences = split_sentences(
                text
            )

            increment(
                "sentences_extracted",
                len(sentences)
            )

            # =================================
            # PROCESS SENTENCES
            # =================================

            for sentence in sentences:

                if not is_valid_sentence(
                    sentence
                ):
                    continue

                increment(
                    "sentences_valid"
                )

                # =============================
                # DEDUP
                # =============================

                sentence_hash = generate_hash(
                    sentence
                )

                if sentence_hash in SEEN_HASHES:

                    increment(
                        "duplicates_skipped"
                    )

                    continue

                SEEN_HASHES.add(
                    sentence_hash
                )

                # =============================
                # INSERT
                # =============================

                push_sentence(

                    source_company=(
                        company_name
                    ),

                    source_url=article.get(
                        "url"
                    ),

                    document_type="NEWS",

                    raw_sentence=sentence,

                    reasoning=(

                        f"{article.get('source', {}).get('name')} | "

                        f"{article.get('title')}"
                    ),

                    accession_number=(
                        article.get("url")
                    ),

                    filing_date=(
                        article.get(
                            "publishedAt"
                        )
                    )
                )

        print("✅ NewsAPI complete")

    except Exception as e:

        increment(
            "request_failures"
        )

        print(
            f"❌ NewsAPI error: {e}"
        )