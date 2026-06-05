"""Temporary script: inspect masked_sentence + relation_id samples from Supabase."""

from __future__ import annotations

import random

from config import ENTITY_FROM, ENTITY_TO, SUPABASE_KEY, SUPABASE_TABLE, SUPABASE_URL
from supabase import create_client


def main() -> None:
    print(f"URL:   {SUPABASE_URL}")
    print(f"Table: {SUPABASE_TABLE}\n")

    client = create_client(SUPABASE_URL, SUPABASE_KEY)

    response = (
        client.table(SUPABASE_TABLE)
        .select("id, masked_sentence, relation_id")
        .not_.is_("masked_sentence", "null")
        .limit(200)
        .execute()
    )
    rows = response.data or []
    if not rows:
        print("No rows with non-null masked_sentence found.")
        return

    sample = random.sample(rows, k=min(5, len(rows)))

    for i, row in enumerate(sample, start=1):
        masked = row.get("masked_sentence") or ""
        relation_id = row.get("relation_id")
        has_from = ENTITY_FROM in masked
        has_to = ENTITY_TO in masked

        print("=" * 72)
        print(f"Sample {i} | id={row.get('id')}")
        print(f"relation_id: {relation_id!r}")
        print(f"has {ENTITY_FROM}: {has_from} | has {ENTITY_TO}: {has_to}")
        print(f"masked_sentence:\n{masked}")
        print()


if __name__ == "__main__":
    main()
