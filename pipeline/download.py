"""
Downloads full message history (+ attachment links) from a Discord channel
using your own account (discord.py-self), for personal dataset creation.

Requires DISCORD_TOKEN in .env. Uses your account's session — respect
Discord's ToS/rate limits at your own risk (self-botting is against ToS).
"""
import argparse
import asyncio
import json
import os
from pathlib import Path

import discord
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.environ["DISCORD_TOKEN"]
CHANNEL_ID = int(os.environ.get("CHANNEL_ID", "920127033508524122"))
OUTPUT_FILE = Path("messages.jsonl")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Max number of messages to fetch (for a quick test run). Default: all.",
)
args = parser.parse_args()

client = discord.Client()


def already_downloaded_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.add(json.loads(line)["id"])
    return ids


def message_to_record(message: discord.Message) -> dict:
    return {
        "id": message.id,
        "timestamp": message.created_at.isoformat(),
        "author_id": message.author.id,
        "author_name": str(message.author),
        "content": message.content,
        "attachments": [
            {
                "filename": a.filename,
                "url": a.url,
                "content_type": a.content_type,
                "size": a.size,
            }
            for a in message.attachments
        ],
        "embeds": [e.to_dict() for e in message.embeds],
        "reply_to": (
            message.reference.message_id if message.reference else None
        ),
    }


@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    channel = await client.fetch_channel(CHANNEL_ID)
    print(f"Fetched channel: {channel} (limit={args.limit or 'all'})")

    seen = already_downloaded_ids(OUTPUT_FILE)
    count = 0

    with OUTPUT_FILE.open("a", encoding="utf-8") as f:
        async for message in channel.history(limit=args.limit, oldest_first=True):
            if message.id in seen:
                continue
            record = message_to_record(message)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            count += 1
            if count % 500 == 0:
                print(f"Downloaded {count} new messages...")

    print(f"Done. Downloaded {count} new messages to {OUTPUT_FILE}")
    await client.close()


if __name__ == "__main__":
    client.run(TOKEN)
