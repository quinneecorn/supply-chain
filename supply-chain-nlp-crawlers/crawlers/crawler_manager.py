# =========================================================
# crawlers/crawler_manager.py
# =========================================================

import time
import logging

from database.supabase_client import (
    supabase
)

from utils.stats import (
    PIPELINE_STATS,
    increment,
    print_stats
)

from crawlers.newsapi_crawler import (
    crawl_newsapi
)

from crawlers.tiingo_crawler import (
    crawl_tiingo
)

from crawlers.companies_house_crawler import (
    crawl_companies_house
)

from crawlers.edinet_crawler import (
    crawl_edinet
)

# =========================================================
# CONFIG
# =========================================================

MAX_DEPTH = 3

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(

    level=logging.INFO,

    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    )
)

logger = logging.getLogger(__name__)

# =========================================================
# LOAD COMPANY QUEUE
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
# MAIN PIPELINE
# =========================================================

def run_pipeline():

    start_time = time.time()

    # =====================================
    # LOAD COMPANIES
    # =====================================

    companies = load_company_queue()

    increment(
        "companies_loaded",
        len(companies)
    )

    print(
        f"\n🚀 Loaded "
        f"{len(companies)} companies"
    )

    # =====================================
    # PROCESS COMPANIES
    # =====================================

    for company in companies:

        company_name = (
            company["company_name"]
        )

        increment(
            "companies_processed"
        )

        print("\n================================")
        print(
            f"PROCESSING: "
            f"{company_name}"
        )
        print("================================")

        # =================================
        # NEWSAPI
        # =================================

        try:

            crawl_newsapi(
                company_name
            )

        except Exception as e:

            increment(
                "request_failures"
            )

            print(
                f"❌ NewsAPI failed: "
                f"{e}"
            )

        time.sleep(2)

        # =================================
        # TIINGO
        # =================================

        try:

            crawl_tiingo(
                company_name
            )

        except Exception as e:

            increment(
                "request_failures"
            )

            print(
                f"❌ Tiingo failed: "
                f"{e}"
            )

        time.sleep(2)

        # =================================
        # COMPANIES HOUSE
        # =================================

        try:

            crawl_companies_house(
                company_name
            )

        except Exception as e:

            increment(
                "request_failures"
            )

            print(
                f"❌ Companies House "
                f"failed: {e}"
            )

        time.sleep(2)

    # =====================================
    # EDINET
    # =====================================

    try:

        crawl_edinet()

    except Exception as e:

        increment(
            "request_failures"
        )

        print(
            f"❌ EDINET failed: {e}"
        )

    # =====================================
    # FINAL STATS
    # =====================================

    total_runtime = round(
        time.time() - start_time,
        2
    )

    PIPELINE_STATS[
        "execution_time_seconds"
    ] = total_runtime

    print("\n✅ PIPELINE COMPLETE")

    print_stats()

# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":

    run_pipeline()