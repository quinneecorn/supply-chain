# =========================================================
# crawlers/tiingo_crawler.py
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

TIINGO_API_KEY = os.getenv(
    "TIINGO_API_KEY"
)

HEADERS = {
    "Authorization": (
        f"Token {TIINGO_API_KEY}"
    )
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

def crawl_tiingo(company_name):

    if not TIINGO_API_KEY:

        print(
            "⚠️ Tiingo disabled "
            "(missing API key)"
        )

        return

    print(
        f"\n📈 Tiingo → "
        f"{company_name}"
    )

    url = (
        "https://api.tiingo.com/"
        "tiingo/news"
    )

    params = {

        "query": company_name,

        "limit": NEWS_LIMIT
    }

    try:

        increment(
            "tiingo_requests"
        )

        response = requests.get(
            url,
            headers=HEADERS,
            params=params,
            timeout=20
        )

        if response.status_code == 403:

            print(
                "⚠️ Tiingo News API "
                "not enabled"
            )

            increment(
                "request_failures"
            )

            return

        response.raise_for_status()

        articles = response.json()

        increment(
            "articles_found",
            len(articles)
        )

        print(
            f"📈 Found "
            f"{len(articles)} articles"
        )

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
                    "summary",
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

            for sentence in sentences:

                if not is_valid_sentence(
                    sentence
                ):
                    continue

                increment(
                    "sentences_valid"
                )

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

                push_sentence(

                    source_company=(
                        company_name
                    ),

                    source_url=article.get(
                        "url"
                    ),

                    document_type="NEWS",

                    raw_sentence=sentence,

                    reasoning=article.get(
                        "title"
                    ),

                    accession_number=str(
                        article.get("id")
                    ),

                    filing_date=article.get(
                        "publishedDate"
                    )
                )

        print("✅ Tiingo complete")

    except Exception as e:

        increment(
            "request_failures"
        )

        print(
            f"❌ Tiingo error: {e}"
        )