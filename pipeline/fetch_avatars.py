"""
Fetches profile pictures for every author who appears in messages.jsonl,
using the logged-in Discord account, and saves them once as PNGs under
frontend/avatars/{author_id}.png (part of the static site, not private data).
"""
import asyncio
import json
import os
from pathlib import Path

import aiohttp
import discord
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.environ["DISCORD_TOKEN"]
MESSAGES_FILE = Path("messages.jsonl")
AVATARS_DIR = Path("frontend/avatars")

client = discord.Client()


def unique_author_ids() -> set[int]:
    ids = set()
    with MESSAGES_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["author_id"])
    return ids


@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    AVATARS_DIR.mkdir(exist_ok=True)

    Path("me.json").write_text(
        json.dumps({"id": client.user.id, "name": str(client.user)}), encoding="utf-8"
    )

    author_ids = unique_author_ids()
    author_ids.add(client.user.id)
    print(f"{len(author_ids)} unique authors")

    async with aiohttp.ClientSession() as session:
        for author_id in author_ids:
            out_path = AVATARS_DIR / f"{author_id}.png"
            if out_path.exists():
                continue
            try:
                user = await client.fetch_user(author_id)
            except discord.NotFound:
                print(f"User {author_id} not found, skipping")
                continue
            except Exception as e:
                print(f"Error fetching user {author_id}: {e}")
                continue

            avatar_asset = user.display_avatar.with_size(256)
            async with session.get(str(avatar_asset.url)) as resp:
                if resp.status == 200:
                    out_path.write_bytes(await resp.read())
                    print(f"Saved avatar for {user} ({author_id})")
                else:
                    print(f"Failed to download avatar for {author_id}: HTTP {resp.status}")

    print("Done fetching avatars.")
    await client.close()


if __name__ == "__main__":
    client.run(TOKEN)
