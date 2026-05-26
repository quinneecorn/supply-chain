# =========================================================
# crawlers/companies_house_crawler.py
# =========================================================

import os
import hashlib
import requests

from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

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

COMPANIES_HOUSE_API_KEY = (
    os.getenv(
        "COMPANIES_HOUSE_API_KEY"
    )
)

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

def crawl_companies_house(
    company_name
):

    print(
        f"\n🏢 Companies House → "
        f"{company_name}"
    )

    url = (
        "https://api.company-information."
        "service.gov.uk/search/companies"
    )

    params = {
        "q": company_name,
        "items_per_page": 5
    }

    try:

        increment(
            "companies_house_requests"
        )

        response = requests.get(

            url,

            params=params,

            auth=HTTPBasicAuth(
                COMPANIES_HOUSE_API_KEY,
                ""
            ),

            timeout=20
        )

        response.raise_for_status()

        data = response.json()

        items = data.get(
            "items",
            []
        )

        print(
            f"🏢 Found "
            f"{len(items)} companies"
        )

        for item in items:

            text = " ".join([

                str(item.get(
                    "title",
                    ""
                )),

                str(item.get(
                    "snippet",
                    ""
                )),

                str(item.get(
                    "company_status",
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

                    source_url=(
                        "https://find-and-update."
                        "company-information."
                        "service.gov.uk/"
                    ),

                    document_type="UK",

                    raw_sentence=sentence,

                    reasoning=item.get(
                        "title"
                    ),

                    accession_number=item.get(
                        "company_number"
                    )
                )

        print(
            "✅ Companies House complete"
        )

    except Exception as e:

        increment(
            "request_failures"
        )

        print(
            f"❌ Companies House error: {e}"
        )