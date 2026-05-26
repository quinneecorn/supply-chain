# =========================================================
# utils/stats.py
# =========================================================

import logging
from collections import defaultdict

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
# GLOBAL STATS
# =========================================================

PIPELINE_STATS = defaultdict(int)

# =========================================================
# INCREMENT
# =========================================================

def increment(metric, amount=1):

    PIPELINE_STATS[metric] += amount

# =========================================================
# PRINT STATS
# =========================================================

def print_stats():

    logger.info("\n" + "=" * 60)

    logger.info(
        "PIPELINE STATISTICS"
    )

    logger.info("=" * 60)

    for key, value in PIPELINE_STATS.items():

        formatted_key = (

            key.replace("_", " ")
            .title()
        )

        logger.info(
            f"{formatted_key:<35} : {value}"
        )

    logger.info("=" * 60 + "\n")