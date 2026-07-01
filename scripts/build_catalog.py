"""
Cleans the raw scraped SHL catalog JSON and normalizes it into the schema
the retrieval + agent layers consume.

Source file has two known issues we handle explicitly (not silently):
  1. At least one record contains a literal newline inside a JSON string
     value (an artifact of the scraper), which makes the file invalid JSON
     under strict parsing. We parse with strict=False, which permits
     control characters inside strings without changing any data values.
  2. The catalog exposes category names in `keys` (e.g. "Personality &
     Behavior") but the API contract (and the conversation traces) use
     single-letter test_type codes (P, K, A, B, C, S, D). We map every
     category name to its code. Items with more than one category get a
     comma-joined code string (e.g. "K,S"), matching the trace examples.

Run:
    python scripts/build_catalog.py
Produces:
    data/catalog.json  -- normalized records ready for embedding/indexing
"""
import json
import sys
from pathlib import Path

RAW_PATH = Path(__file__).parent.parent / "data" / "shl_product_catalog.json"
OUT_PATH = Path(__file__).parent.parent / "data" / "catalog.json"

# Mapping from full SHL catalog category name -> single-letter test_type code.
# These letters match exactly what appears in the provided conversation
# traces (C1-C10), e.g. "P" for Personality & Behavior, "K" for Knowledge &
# Skills, "A" for Ability & Aptitude, "S" for Simulations, "B" for Biodata &
# Situational Judgment, "C" for Competencies, "D" for Development & 360.
CATEGORY_TO_CODE = {
    "Ability & Aptitude": "A",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
    "Assessment Exercises": "E",  # not seen in traces but present in raw data; assigned for completeness
}


def derive_test_type(keys: list[str]) -> str:
    codes = []
    for k in keys:
        code = CATEGORY_TO_CODE.get(k)
        if code is None:
            # Fail loudly rather than silently dropping an unmapped category --
            # we want to know if SHL's catalog adds a new category type.
            raise ValueError(f"Unmapped category encountered: {k!r}")
        codes.append(code)
    # Preserve catalog order, de-dup, keep stable.
    seen = []
    for c in codes:
        if c not in seen:
            seen.append(c)
    return ",".join(seen) if seen else ""


def slug_from_link(link: str) -> str:
    return link.rstrip("/").split("/")[-1]


def main():
    if not RAW_PATH.exists():
        print(f"Raw catalog not found at {RAW_PATH}", file=sys.stderr)
        sys.exit(1)

    raw_text = RAW_PATH.read_text(encoding="utf-8")
    # strict=False permits the embedded literal-newline control character
    # found in at least one record without altering any string content.
    records = json.loads(raw_text, strict=False)

    normalized = []
    dropped = []
    seen_slugs = set()

    for r in records:
        if r.get("status") != "ok":
            dropped.append((r.get("entity_id"), "status != ok"))
            continue

        link = r.get("link", "").strip()
        if not link:
            dropped.append((r.get("entity_id"), "missing link"))
            continue

        slug = slug_from_link(link)
        if slug in seen_slugs:
            dropped.append((r.get("entity_id"), f"duplicate slug {slug}"))
            continue
        seen_slugs.add(slug)

        name = " ".join(r.get("name", "").split())  # collapses embedded newlines/extra whitespace

        # Known scraper defect: at least one record's `name` was truncated
        # because a literal newline in the source HTML landed mid-string
        # (e.g. "Microsoft \n365 (New)" -> "Microsoft 365 (New)" after the
        # whitespace collapse above, silently dropping "Excel"). The slug
        # in `link` is unaffected by this defect, so we reconstruct any
        # name with a suspicious gap by cross-checking against the slug.
        # We only ever ADD back a word that is present in the slug but
        # missing from the cleaned name -- we never invent new content.
        slug_words = set(slug.replace("-new", "").replace("-", " ").split())
        name_words_lower = {w.lower() for w in name.split()}
        if name.lower().startswith("microsoft ") and "excel" in slug_words and "excel" not in name_words_lower:
            name = name.replace("Microsoft ", "Microsoft Excel ", 1)
        elif name.lower().startswith("microsoft ") and "word" in slug_words and "word" not in name_words_lower:
            name = name.replace("Microsoft ", "Microsoft Word ", 1)
        elif name.lower().startswith("microsoft ") and "powerpoint" in slug_words and "powerpoint" not in name_words_lower:
            name = name.replace("Microsoft ", "Microsoft PowerPoint ", 1)

        keys = r.get("keys", []) or []

        try:
            test_type = derive_test_type(keys)
        except ValueError as e:
            print(f"WARNING: {e} (entity_id={r.get('entity_id')}, name={name!r}) -- skipping", file=sys.stderr)
            dropped.append((r.get("entity_id"), str(e)))
            continue

        normalized.append({
            "id": r.get("entity_id"),
            "slug": slug,
            "name": name,
            "url": link,
            "test_type": test_type,
            "categories": keys,
            "description": " ".join((r.get("description") or "").split()),
            "duration": r.get("duration") or "",
            "duration_raw": r.get("duration_raw") or "",
            "languages": r.get("languages") or [],
            "job_levels": r.get("job_levels") or [],
            "remote": r.get("remote") == "yes",
            "adaptive": r.get("adaptive") == "yes",
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(normalized, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Normalized {len(normalized)} records -> {OUT_PATH}")
    if dropped:
        print(f"Dropped {len(dropped)} records:")
        for entity_id, reason in dropped:
            print(f"  - {entity_id}: {reason}")


if __name__ == "__main__":
    main()
