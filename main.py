#!/usr/bin/env python3
"""
Daily automated "Dark Psychology" video factory.

Gemini (script + image prompts) -> Pollinations (images) -> edge-tts (voice)
-> FFmpeg Ken Burns (render) -> YouTube / Facebook / Instagram / TikTok upload.

Daily output:
  3 x short (9:16, ~45s)   -> YouTube Shorts + Facebook Reels + Instagram Reels
  1 x long  (16:9, ~5 min) -> YouTube + Facebook Page video
  2 x tiktok (9:16, ~75s)  -> TikTok
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import edge_tts
import requests
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("bot")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
VOICE = os.getenv("TTS_VOICE", "en-US-GuyNeural")
GRAPH = os.getenv("META_GRAPH_VERSION", "v24.0")
TIKTOK_PRIVACY = os.getenv("TIKTOK_PRIVACY_LEVEL", "SELF_ONLY")  # PUBLIC_TO_EVERYONE after TikTok audit
DRY_RUN = os.getenv("DRY_RUN", "false").strip().lower() in ("1", "true", "yes")
PLATFORMS = {p.strip().lower() for p in os.getenv("PLATFORMS", "youtube,facebook,instagram,tiktok").split(",")}

FPS = 25
SIZES = {"short": (1080, 1920), "tiktok": (1080, 1920), "long": (1280, 720)}
SPEC = {
    "short": {"seconds": 45, "scenes": 7},
    "tiktok": {"seconds": 75, "scenes": 10},
    "long": {"seconds": 300, "scenes": 20},
}
STYLE = os.getenv(
    "IMAGE_STYLE",
    "3D Pixar-style cartoon render, cinematic dark moody lighting, dramatic shadows, "
    "deep purple and teal color palette, highly detailed, 8k, no text, no letters, no watermark",
)

WORK = Path("work")
OUT = Path("output")

_client: genai.Client | None = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def run(cmd: list[str]) -> str:
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return res.stdout
    except subprocess.CalledProcessError as e:
        log.error("Command failed: %s\n%s", " ".join(cmd[:6]), (e.stderr or "")[-1500:])
        raise


def media_duration(path: Path) -> float:
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=nw=1:nk=1", str(path)])
    return float(out.strip())


def check_json(resp: requests.Response, what: str) -> dict:
    if not resp.ok:
        raise RuntimeError(f"{what} failed: HTTP {resp.status_code} {resp.text[:600]}")
    try:
        return resp.json()
    except ValueError:
        return {}


def clean_tags(tags: list[str]) -> list[str]:
    out = []
    for t in tags:
        t = re.sub(r"[^A-Za-z0-9_]", "", str(t))
        if t and t.lower() not in [x.lower() for x in out]:
            out.append(t)
    return out[:12]


# --------------------------------------------------------------------------- #
# 1. Gemini: topics + scripts
# --------------------------------------------------------------------------- #
def gemini_json(prompt: str, retries: int = 4):
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = _client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.9,
                    response_mime_type="application/json",
                    max_output_tokens=30000,
                ),
            )
            text = (resp.text or "").strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
            return json.loads(text)
        except Exception as e:  # noqa: BLE001
            last = e
            wait = 20 * (attempt + 1)
            log.warning("Gemini attempt %d failed (%s). Retrying in %ds", attempt + 1, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"Gemini failed: {last}")


def plan_topics(n: int = 6) -> list[str]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = (
        "You are a viral content strategist for a high-CPM 'Dark Psychology' educational channel "
        "(manipulation tactics, persuasion, body language, cognitive biases, narcissistic patterns, "
        "influence, negotiation, and how to protect yourself from manipulation). "
        f"Today is {today}. Generate {n} unique, specific, curiosity-driven video topics. "
        'Return JSON only: {"topics": ["...", "..."]}'
    )
    topics = [str(t) for t in gemini_json(prompt).get("topics", []) if str(t).strip()]
    fallback = ["The silent treatment trick", "Why love bombing works", "The foot-in-the-door trap",
                "Gaslighting phrases to spot", "How scarcity controls you", "The mirroring manipulation"]
    for f in fallback:
        if len(topics) >= n:
            break
        topics.append(f)
    return topics[:n]


def gen_script(kind: str, topic: str) -> dict:
    spec = SPEC[kind]
    words = int(spec["seconds"] * 2.5)
    per_scene = max(8, words // spec["scenes"])
    prompt = f"""Write a {spec['seconds']}-second voiceover script for a Dark Psychology video about: "{topic}".

Rules:
- Exactly {spec['scenes']} scenes. Total narration about {words} words (about {per_scene} words per scene).
- Scene 1 is a shocking, curiosity-driven hook of under 15 words.
- Plain English, short punchy dramatic sentences. Educational tone: awareness and self-protection.
- No real people or brands, no medical/legal claims, no violence or gore.
- Last scene: a short call to follow for more.
- image_prompt: a detailed English visual description of the scene (characters, setting, lighting, camera angle).
  No text, letters, or real people in the image.
- title: max 90 characters, catchy but not misleading. description: 2-3 sentences. hashtags: 8 single words without '#'.

Return JSON only:
{{"title": "...", "description": "...", "hashtags": ["..."],
  "scenes": [{{"narration": "...", "image_prompt": "..."}}]}}"""
    data = gemini_json(prompt)
    scenes = [s for s in data.get("scenes", [])
              if isinstance(s, dict) and s.get("narration") and s.get("image_prompt")]
    if len(scenes) < 3:
        raise RuntimeError("Gemini returned too few valid scenes")
    data["scenes"] = scenes
    data["title"] = str(data.get("title") or topic)[:95]
    data["description"] = str(data.get("description") or topic)
    data["hashtags"] = clean_tags(data.get("hashtags") or ["darkpsychology", "psychology"])
    return data


# --------------------------------------------------------------------------- #
# 2. Images (Pollinations) + voice (edge-tts)
# --------------------------------------------------------------------------- #
def fallback_image(path: Path, w: int, h: int) -> None:
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"color=c=0x1b1030:s={w}x{h}", "-frames:v", "1", str(path)])


def gen_image(prompt: str, w: int, h: int, path: Path, seed: int) -> None:
    url = "https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt[:900])
    params = {"width": w, "height": h, "seed": seed, "nologo": "true", "model": "flux"}
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, timeout=180)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image") and len(r.content) > 5000:
                path.write_bytes(r.content)
                return
            log.warning("Pollinations HTTP %s (attempt %d)", r.status_code, attempt + 1)
        except requests.RequestException as e:
            log.warning("Pollinations error (attempt %d): %s", attempt + 1, e)
        time.sleep(8 * (attempt + 1))
    log.error("Image generation failed, using fallback background")
    fallback_image(path, w, h)


def tts(text: str, path: Path) -> None:
    async def _go() -> None:
        com = edge_tts.Communicate(text, VOICE, rate="-4%", pitch="-4Hz")
        await com.save(str(path))

    for attempt in range(4):
        try:
            asyncio.run(_go())
            if path.exists() and path.stat().st_size > 1000:
                return
        except Exception as e:  # noqa: BLE001
            log.warning("edge-tts attempt %d failed: %s", attempt + 1, e)
        time.sleep(4 * (attempt + 1))
    raise RuntimeError("edge-tts failed")


# --------------------------------------------------------------------------- #
# 3. FFmpeg: Ken Burns + concat
# --------------------------------------------------------------------------- #
def make_clip(img: Path, audio: Path, w: int, h: int, idx: int, out: Path) -> None:
    dur = media_duration(audio) + 0.35
    n = max(int(dur * FPS), 2)
    motions = [
        ("1+0.25*on/{n}", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),            # zoom in
        ("1.25-0.25*on/{n}", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),         # zoom out
        ("1.2", "(iw-iw/zoom)*on/{n}", "ih/2-(ih/zoom/2)"),                   # pan left -> right
        ("1.2", "(iw-iw/zoom)*(1-on/{n})", "ih/2-(ih/zoom/2)"),               # pan right -> left
    ]
    z, x, y = [m.format(n=n) for m in motions[idx % len(motions)]]
    vf = (
        f"scale={w*2}:{h*2}:force_original_aspect_ratio=increase,crop={w*2}:{h*2},"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={n}:s={w}x{h}:fps={FPS},format=yuv420p"
    )
    run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(img), "-i", str(audio),
        "-vf", vf, "-af", "apad", "-frames:v", str(n),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-r", str(FPS),
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        "-shortest", str(out),
    ])


def build_video(kind: str, script: dict, name: str) -> Path:
    w, h = SIZES[kind]
    d = WORK / name
    d.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []
    base_seed = int(time.time()) % 100000
    for i, sc in enumerate(script["scenes"]):
        log.info("[%s] scene %d/%d", name, i + 1, len(script["scenes"]))
        audio = d / f"s{i:02d}.mp3"
        img = d / f"s{i:02d}.jpg"
        clip = d / f"c{i:02d}.mp4"
        tts(re.sub(r"\s+", " ", sc["narration"]).strip(), audio)
        gen_image(f"{sc['image_prompt']}. {STYLE}", w, h, img, base_seed + i)
        make_clip(img, audio, w, h, i, clip)
        clips.append(clip)
        time.sleep(1)

    listing = d / "list.txt"
    listing.write_text("".join(f"file '{c.resolve()}'\n" for c in clips), encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    final = OUT / f"{name}.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(final)])
    log.info("[%s] rendered %s (%.1fs, %.1f MB)", name, final, media_duration(final),
             final.stat().st_size / 1e6)
    shutil.rmtree(d, ignore_errors=True)
    return final


# --------------------------------------------------------------------------- #
# 4. Uploaders
# --------------------------------------------------------------------------- #
def caption(script: dict) -> str:
    tags = " ".join("#" + t for t in script["hashtags"])
    return f"{script['title']}\n\n{script['description']}\n\n{tags}"[:2100]


def upload_youtube(path: Path, script: dict, is_short: bool) -> str:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    info = json.loads(os.environ["YOUTUBE_CREDENTIALS"])
    creds = Credentials.from_authorized_user_info(info, ["https://www.googleapis.com/auth/youtube.upload"])
    yt = build("youtube", "v3", credentials=creds, cache_discovery=False)

    title = script["title"]
    if is_short and "#shorts" not in title.lower() and len(title) <= 90:
        title += " #Shorts"
    desc = script["description"] + "\n\n" + " ".join("#" + t for t in script["hashtags"])
    if is_short:
        desc += " #Shorts"
    body = {
        "snippet": {"title": title[:100], "description": desc[:4900],
                    "tags": script["hashtags"], "categoryId": "27"},
        "status": {"privacyStatus": "public", "selfDeclaredMadeForKids": False,
                   "containsSyntheticMedia": True},
    }
    media = MediaFileUpload(str(path), mimetype="video/mp4", chunksize=8 * 1024 * 1024, resumable=True)
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = None
    while resp is None:
        _, resp = req.next_chunk()
    return resp["id"]


def upload_facebook_long(path: Path, script: dict) -> str:
    token, page = os.environ["META_ACCESS_TOKEN"], os.environ["PAGE_ID"]
    with open(path, "rb") as f:
        r = requests.post(
            f"https://graph-video.facebook.com/{GRAPH}/{page}/videos",
            data={"access_token": token, "title": script["title"][:250], "description": caption(script)},
            files={"source": f}, timeout=3600,
        )
    return str(check_json(r, "Facebook video").get("id"))


def upload_facebook_reel(path: Path, script: dict) -> str:
    token, page = os.environ["META_ACCESS_TOKEN"], os.environ["PAGE_ID"]
    base = f"https://graph.facebook.com/{GRAPH}/{page}/video_reels"
    start = check_json(requests.post(base, data={"upload_phase": "start", "access_token": token},
                                     timeout=120), "FB reel start")
    video_id, upload_url = start["video_id"], start["upload_url"]
    size = path.stat().st_size
    with open(path, "rb") as f:
        up = requests.post(upload_url, data=f, timeout=3600, headers={
            "Authorization": f"OAuth {token}", "offset": "0", "file_size": str(size)})
    check_json(up, "FB reel upload")
    fin = requests.post(base, data={
        "access_token": token, "video_id": video_id, "upload_phase": "finish",
        "video_state": "PUBLISHED", "description": caption(script)}, timeout=300)
    check_json(fin, "FB reel finish")
    return video_id


def _ig_user_id(token: str, page: str) -> str:
    if os.getenv("IG_USER_ID"):
        return os.environ["IG_USER_ID"]
    r = requests.get(f"https://graph.facebook.com/{GRAPH}/{page}",
                     params={"fields": "instagram_business_account", "access_token": token}, timeout=60)
    acc = check_json(r, "IG account lookup").get("instagram_business_account", {}).get("id")
    if not acc:
        raise RuntimeError("No Instagram business account linked to this Facebook Page")
    return acc


def upload_instagram_reel(path: Path, script: dict) -> str:
    token, page = os.environ["META_ACCESS_TOKEN"], os.environ["PAGE_ID"]
    ig = _ig_user_id(token, page)
    c = check_json(requests.post(f"https://graph.facebook.com/{GRAPH}/{ig}/media", data={
        "media_type": "REELS", "upload_type": "resumable", "caption": caption(script),
        "share_to_feed": "true", "access_token": token}, timeout=120), "IG container")
    cid, uri = c["id"], c["uri"]
    size = path.stat().st_size
    with open(path, "rb") as f:
        up = requests.post(uri, data=f, timeout=3600, headers={
            "Authorization": f"OAuth {token}", "offset": "0", "file_size": str(size)})
    check_json(up, "IG upload")
    for _ in range(40):
        st = check_json(requests.get(f"https://graph.facebook.com/{GRAPH}/{cid}",
                                     params={"fields": "status_code,status", "access_token": token},
                                     timeout=60), "IG status")
        code = st.get("status_code")
        if code == "FINISHED":
            break
        if code in ("ERROR", "EXPIRED"):
            raise RuntimeError(f"IG processing {code}: {st.get('status')}")
        time.sleep(10)
    else:
        raise RuntimeError("IG processing timeout")
    pub = check_json(requests.post(f"https://graph.facebook.com/{GRAPH}/{ig}/media_publish",
                                   data={"creation_id": cid, "access_token": token}, timeout=120), "IG publish")
    return str(pub.get("id"))


def tiktok_token() -> str:
    key, secret, refresh = (os.getenv("TIKTOK_CLIENT_KEY"), os.getenv("TIKTOK_CLIENT_SECRET"),
                            os.getenv("TIKTOK_REFRESH_TOKEN"))
    if key and secret and refresh:  # access tokens expire every 24h -> refresh automatically
        r = requests.post("https://open.tiktokapis.com/v2/oauth/token/", data={
            "client_key": key, "client_secret": secret,
            "grant_type": "refresh_token", "refresh_token": refresh}, timeout=60)
        tok = check_json(r, "TikTok refresh").get("access_token")
        if tok:
            return tok
    return os.environ["TIKTOK_ACCESS_TOKEN"]


def upload_tiktok(path: Path, script: dict) -> str:
    token = tiktok_token()
    size = path.stat().st_size
    if size > 64 * 1024 * 1024:
        raise RuntimeError("Video larger than 64MB single-chunk limit")
    init = check_json(requests.post(
        "https://open.tiktokapis.com/v2/post/publish/video/init/",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8"},
        json={
            "post_info": {"title": caption(script)[:2100], "privacy_level": TIKTOK_PRIVACY,
                          "disable_duet": False, "disable_comment": False, "disable_stitch": False,
                          "is_aigc": True},
            "source_info": {"source": "FILE_UPLOAD", "video_size": size,
                            "chunk_size": size, "total_chunk_count": 1},
        }, timeout=120), "TikTok init")
    data = init.get("data") or {}
    upload_url, publish_id = data.get("upload_url"), data.get("publish_id")
    if not upload_url:
        raise RuntimeError(f"TikTok init response: {init}")
    with open(path, "rb") as f:
        up = requests.put(upload_url, data=f, timeout=1800, headers={
            "Content-Type": "video/mp4", "Content-Length": str(size),
            "Content-Range": f"bytes 0-{size - 1}/{size}"})
    if up.status_code not in (200, 201, 206):
        raise RuntimeError(f"TikTok upload failed: {up.status_code} {up.text[:400]}")
    return str(publish_id)


# --------------------------------------------------------------------------- #
# 5. Orchestration
# --------------------------------------------------------------------------- #
def has(*names: str) -> bool:
    return all(os.getenv(n) for n in names)


def safe_upload(results: list, label: str, platform: str, needs: tuple, fn, *args) -> None:
    if platform not in PLATFORMS:
        return
    if DRY_RUN:
        results.append((label, "DRY_RUN"))
        return
    if not has(*needs):
        results.append((label, "SKIPPED (missing secrets: " + ", ".join(needs) + ")"))
        return
    try:
        results.append((label, f"OK id={fn(*args)}"))
    except Exception as e:  # noqa: BLE001
        log.error("%s failed: %s", label, e)
        results.append((label, f"FAILED: {str(e)[:200]}"))


def main() -> int:
    if not os.getenv("GEMINI_API_KEY"):
        log.error("GEMINI_API_KEY is missing")
        return 1
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        log.error("FFmpeg/ffprobe not installed")
        return 1
    WORK.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)

    topics = plan_topics(6)
    log.info("Topics: %s", topics)
    jobs = [("short", topics[0]), ("short", topics[1]), ("short", topics[2]),
            ("long", topics[3]), ("tiktok", topics[4]), ("tiktok", topics[5])]
    stamp = datetime.now(timezon.utc)
