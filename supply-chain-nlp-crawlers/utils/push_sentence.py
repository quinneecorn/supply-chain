# =========================================================
# utils/push_sentence.py
# =========================================================

from database.supabase_client import (
    supabase
)

from utils.stats import (
    increment
)

# =========================================================
# INSERT SENTENCE
# =========================================================

def push_sentence(

    source_company,

    source_url,

    document_type,

    raw_sentence,

    reasoning=None,

    accession_number=None,

    filing_date=None
):

    try:

        supabase.table(
            "scraped_sentences"
        ).upsert(

            {

                "source_company": (
                    source_company
                ),

                "source_url": (
                    source_url
                ),

                "document_type": (
                    document_type
                ),

                "raw_sentence": (
                    raw_sentence
                ),

                "masked_sentence": None,

                "extracted_entities": None,

                "llm_processed": False,

                "relation_id": None,

                "relation_type": None,

                "entity_from": None,

                "entity_to": None,

                "confidence_score": None,

                "reasoning": reasoning,

                "accession_number": (
                    accession_number
                ),

                "filing_date": (
                    filing_date
                )

            },

            on_conflict="raw_sentence"

        ).execute()

        increment(
            "sentences_inserted"
        )

    except Exception as e:

        error_message = str(e)

        if "duplicate key value" in error_message:

            increment(
                "duplicates_skipped"
            )

            return

        increment(
            "request_failures"
        )

        print(
            f"❌ Insert failed: {e}"
        )