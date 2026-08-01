"""Builds frontend/authors.json: {author_id: {name, avatar}} from
messages.jsonl + frontend/avatars/. Output lives in frontend/ since it's a
static site asset, not private data — only messages.jsonl/finetune_dataset.jsonl
(the raw chat log) stay out of the deployed site."""
import json
from pathlib import Path

MESSAGES_FILE = Path("messages.jsonl")
AVATARS_DIR = Path("frontend/avatars")
OUTPUT_FILE = Path("frontend/authors.json")


def main():
    authors = {}
    with MESSAGES_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            authors[str(record["author_id"])] = record["author_name"]

    me = json.loads(Path("me.json").read_text(encoding="utf-8")) if Path("me.json").exists() else None
    if me:
        authors[str(me["id"])] = me["name"]

    result = {}
    for author_id, name in authors.items():
        avatar_path = AVATARS_DIR / f"{author_id}.png"
        result[author_id] = {
            "name": name,
            "avatar": f"avatars/{author_id}.png" if avatar_path.exists() else None,
        }

    OUTPUT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(result)} authors to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
