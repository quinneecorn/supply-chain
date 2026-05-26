# =========================================================
# crawlers/edinet_crawler.py
# =========================================================

import hashlib
import requests

from datetime import (
    datetime,
    timedelta
)

from utils.text_utils import (
    split_sentences,
    is_valid_sentence
)

from utils.stats import increment

from utils.push_sentence import (
    push_sentence
)

# =========================================================
# CONFIG
# =========================================================

HEADERS = {
    "User-Agent": "SupplyChainNLPBot/1.0"
}

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

def crawl_edinet():

    print(
        "\n🇯🇵 EDINET "
        "(Last 30 Days)"
    )

    for days_back in range(30):

        target_date = (
            datetime.today()
            - timedelta(days=days_back)
        ).strftime("%Y-%m-%d")

        url = (
            "https://disclosure."
            "edinet-fsa.go.jp/"
            "api/v2/documents.json"
        )

        params = {
            "date": target_date,
            "type": 1
        }

        try:

            increment(
                "edinet_requests"
            )

            response = requests.get(

                url,

                params=params,

                headers=HEADERS,

                timeout=30
            )

            response.raise_for_status()

            data = response.json()

            filings = data.get(
                "results",
                []
            )

            print(
                f"🇯🇵 {target_date} "
                f"→ {len(filings)} filings"
            )

            for filing in filings:

                text = " ".join([

                    str(filing.get(
                        "filerName",
                        ""
                    )),

                    str(filing.get(
                        "docDescription",
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

                    sentence_hash = (
                        generate_hash(
                            sentence
                        )
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
                            filing.get(
                                "filerName"
                            )
                        ),

                        source_url=(
                            "https://disclosure."
                            "edinet-fsa.go.jp/"
                        ),

                        document_type="JP",

                        raw_sentence=sentence,

                        reasoning=(
                            filing.get(
                                "docDescription"
                            )
                        ),

                        accession_number=(
                            filing.get(
                                "docID"
                            )
                        ),

                        filing_date=(
                            filing.get(
                                "submitDateTime"
                            )
                        )
                    )

        except Exception as e:

            increment(
                "request_failures"
            )

            print(
                f"❌ EDINET error: {e}"
            )

    print("✅ EDINET complete")