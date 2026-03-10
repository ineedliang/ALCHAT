<div align="center">

<br>

```
 ________  ___          ________  ___  ___  ________  _________
|\   __  \|\  \        |\   ____\|\  \|\  \|\   __  \|\___   ___\
\ \  \|\  \ \  \       \ \  \___|\ \  \\\  \ \  \|\  \|___ \  \_|
 \ \   __  \ \  \       \ \  \    \ \   __  \ \   __  \   \ \  \
  \ \  \ \  \ \  \____   \ \  \____\ \  \ \  \ \  \ \  \   \ \  \
   \ \__\ \__\ \_______\  \ \_______\ \__\ \__\ \__\ \__\   \ \__\
    \|__|\|__|\|_______|   \|_______|\|__|\|__|\|__|\|__|    \|__|
```

### A fully local AI assistant. Beautiful, capable, entirely yours.

*60+ models &middot; Live web search &middot; Voice I/O &middot; OBS streaming &middot; Persistent memory &middot; Dark neon UI*

<br>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.6-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![PyQt6](https://img.shields.io/badge/UI-PyQt6-41cd52?style=for-the-badge&logo=qt&logoColor=white)](https://doc.qt.io/qtforpython-6/)
[![HuggingFace](https://img.shields.io/badge/Models-HuggingFace-ffd21e?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-e94560?style=for-the-badge)](LICENSE)
[![Windows](https://img.shields.io/badge/Platform-Windows-0078d4?style=for-the-badge&logo=windows&logoColor=white)](https://microsoft.com/windows)

<br>
<br>

</div>

---

<br>

## The pitch

You want a real AI on your machine. Not a tab in your browser. Not a subscription. Not your conversations training someone else's model. A desktop app that loads any HuggingFace model, remembers everything you've ever said to it, searches the live web on demand, reads you the weather, pipes output straight into OBS while you stream -- and looks good doing it.

That's AI Chat.

Single Python file. Isolated venv. Run `setup.bat` once, then `run.bat` forever.

<br>

---

<br>

## Quick Start

```bat
git clone https://github.com/yourusername/ai-chat
cd ai-chat

setup.bat     <- creates venv, installs everything, verifies your GPU
run.bat       <- launch
```

> **First launch** downloads the default model (`Qwen3-0.6B`, ~1 GB).
> Subsequent launches are instant -- models cache to `~/ai_models`.

<br>

---

<br>

## Features

<br>

### Models -- 60+ across 10 families

Open the **Load Model** dialog and pick from a curated list, or type any HuggingFace model ID directly. The app downloads it, shows a live progress panel with per-file download bars and stage indicators, and you're talking within minutes.

| Family | Size Range | Strengths |
|---|---|---|
| **Qwen3** (recommended) | 0.6B to 32B | Best overall, excellent default |
| **Qwen2.5 Instruct** | 0.5B to 72B | Versatile, strong instruction following |
| **Qwen2.5 Coder** | 1.5B to 32B | Code generation and review |
| **Microsoft Phi-3 / 3.5** | mini, small, medium | Punches above its weight class |
| **Meta Llama 3.1 / 3.2** | 1B to 70B | Well-rounded, widely supported |
| **Mistral / Mixtral** | 7B, 8x7B, Nemo | Strong reasoning |
| **Google Gemma 2 / 3** | 2B to 27B | Fast inference |
| **DeepSeek R1 Distill** | 1.5B, 7B, 8B | Reasoning and coding tasks |
| **SmolLM2 / TinyLlama** | 360M to 1.7B | Minimal VRAM, snappy responses |
| **Yi / Falcon / Zephyr / Vicuna** | varies | Variety and legacy support |

**4-bit NF4 quantization** (bitsandbytes) is a single checkbox -- halves VRAM usage with minimal quality loss. VRAM estimates shown before you commit to a download.

<br>

### Chat

- **Streaming output** -- tokens appear as they generate, no waiting
- **Animated chat bubbles** -- user right, assistant left, smooth entry animations
- **Syntax-highlighted code blocks** -- Python keywords, strings, and comments coloured automatically
- **One-click copy** on every code block and every response
- **Typing indicator** -- three animated dots while the model is thinking
- **Persona system** -- seven built-in characters, all editable via a live text box

<br>

### Live Web, Weather & News -- zero API keys

Just talk naturally. The app reads intent from your words.

```
"search the web for recent breakthroughs in fusion energy"
"what's the weather in Reykjavik this weekend"
"top tech news today"
```

- **Web search** -- scrapes DuckDuckGo HTML, parses result snippets, injects them as context before generating
- **Weather** -- uses your exact city phrase as a live search query, always current, no geolocation API
- **News** -- pulls from 30+ RSS feeds across 6 categories, all toggleable per-source in the sidebar

<br>

### Voice

- **Mic button** -- click to record, speech-to-text via Google STT
- **Wake word mode** -- say `hey chat` (configurable), the app captures your follow-up and auto-sends
- **TTS playback** -- responses read aloud, with live volume and speed sliders
- All audio on background threads -- voice never blocks the UI

<br>

### OBS Streaming Integration

Wire the app directly into your OBS scene. Connects per-send -- no persistent socket to babysit.

```
UserText  <- what you typed, appears instantly when you send
ChatText  <- AI response streams in live as tokens generate
```

**Three output modes:**

- **Stream** -- updates OBS source every 50 characters while generating
- **Paginate** -- breaks long responses into screen-sized chunks, advances on a timer
- **Ticker** -- scrolls text horizontally word by word, great for lower-thirds

**Auto scene switching** -- set a talking scene and an idle scene, the app switches between them as generation starts and stops.

<br>

### Persistent Memory

Every message is written to SQLite the moment it's sent. Persists across crashes, restarts, and power cuts.

```
~/ai_chat_saves/
|-- memory.db          <- all sessions and messages
`-- chat_20250310.json <- optional exports
```

On startup, the app automatically resumes your last session and renders the full message history. From the sidebar you can create, rename, and delete sessions, export to JSON, import JSON as a new session, and run full-text search across every message ever written.

<br>

### Activity Log

A slide-out panel on the right shows everything happening under the hood in real time.

| Colour | Level | What you see |
|---|---|---|
| Blue | `INFO` | Model events, fetch starts, scene switches |
| Yellow | `WARN` | Non-fatal issues, slow RSS, partial results |
| Red | `ERROR` | Failures with full error text |
| Green | `SUCCESS` | Completions -- model ready, search done, OBS sent |
| Gray | `DEBUG` | Token/sec, VRAM, raw OBS payloads (opt-in) |

Per-level filter checkboxes. Running event counter. One-click clear.

<br>

---

<br>

## Interface Layout

```
+--+----------------+-----------------------------------------+----------------+--+
|  |                |                                         |                |  |
|<<|   SIDEBAR      |             CHAT AREA                   |   LOG PANEL  >>|  |
|  |                |                                         |                |  |
|  |  Persona       |                                         | 21:04:01 INFO  |  |
|  |  ----------    |   +-------------------------------+     | Model loaded   |  |
|  |  Parameters    |   | You                           |     |                |  |
|  |  Temperature   |   | search the web for quantum    |     | 21:04:03 OK    |  |
|  |  Top-P         |   | computing breakthroughs 2025  |     | DuckDuckGo: 6  |  |
|  |  Max Tokens    |   +-------------------------------+     | results found  |  |
|  |  ----------    |                                         |                |  |
|  |  Sessions      |  +----------------------------------+   | 21:04:04 INFO  |  |
|  |  + New         |  | Assistant                        |   | Generating...  |  |
|  |  Rename        |  | Several major milestones were    |   |                |  |
|  |  Delete        |  | announced this year. Google's    |   | 21:04:07 OK    |  |
|  |  Search...     |  | Willow chip achieved...          |   | 54.2 tok/s     |  |
|  |  ----------    |  |                                  |   |                |  |
|  |  Wake Word     |  |   ```python                      |   | 21:04:07 INFO  |  |
|  |  OBS Settings  |  |   qubit_count = 105              |   | OBS -> ChatText|  |
|  |  News Sources  |  |   error_rate  = 0.001            |   | 21:04:07 OK    |  |
|  |                |  |   ```              [copy]         |   | OBS sent OK    |  |
|  |                |  +----------------------------------+   |                |  |
|  |                |  * * *                                  |                |  |
+--+----------------+-----------------------------------------+----------------+--+
|  [mic]  [ Ask me anything...                           ]  [  Send ->  ]        |
|  TTS [vol] [spd]   OBS   4-bit GPU                           Tokens: 2,847     |
+--------------------------------------------------------------------------------+
```

The sidebar and log panel slide in and out independently. The load panel replaces the chat area while a model is downloading and restores automatically when ready.

<br>

---

<br>

## VRAM Guide

All figures with **4-bit quantization enabled**. Disable it for full precision (~2x VRAM).

| Model size | Min VRAM | Runs well on |
|---|---|---|
| 0.5B - 1.7B | 1-2 GB or CPU | Any modern GPU, integrated graphics |
| 3B - 4B | 2.5-3 GB | GTX 1060, RTX 3050 |
| 7B - 8B | 4-6 GB | RTX 3060, RTX 4060 |
| 14B | 8-10 GB | RTX 3080, RTX 4070 |
| 27B - 32B | 16-22 GB | RTX 4090, A100 |
| 70B - 72B | 40+ GB | A100 80GB, multi-GPU |

No GPU? CPU mode works. Expect roughly 1-3 tok/s on a modern desktop CPU for a 3B model.

<br>

---

<br>

## Installation

### Prerequisites

- **Windows 10 or 11**
- **Python 3.10+** from [python.org](https://python.org) -- check **"Add Python to PATH"** during install
- **NVIDIA GPU** strongly recommended for models above 3B parameters

### What `setup.bat` does

```
[1/9]  Verify Python 3.10+
[2/9]  Create isolated venv/  (deletes and recreates if one already exists)
[3/9]  Upgrade pip, setuptools, wheel inside the venv
[4/9]  Detect GPU via nvidia-smi
[5/9]  Install PyTorch  (CUDA 12.6 -> 12.4 -> CPU, whichever succeeds)
[6/9]  Install PyAudio  (tries pipwin fallback on Windows if needed)
[7/9]  Install all remaining packages from requirements.txt
[8/9]  Write run.bat with the correct venv activation path
[9/9]  Verify every package -- prints [OK] or [!!] for each one
```

Everything installs into `venv/`. Your system Python is never modified.

### Launching and updating

```bat
run.bat       <- activates venv, launches the app, deactivates on exit
update.bat    <- upgrades all packages inside the venv
```

Manual launch:
```bat
venv\Scripts\activate
python main.py
```

<br>

---

<br>

## Configuration

Edit the constants at the top of `main.py`:

```python
# Model
DEFAULT_MODEL  = "Qwen/Qwen3-0.6B"            # loaded on startup
MODELS_DIR     = "~/ai_models"                 # cache (can be another drive)

# Paths
SAVES_DIR      = "~/ai_chat_saves"             # JSON exports
DB_PATH        = "~/ai_chat_saves/memory.db"   # SQLite database

# Voice
WAKE_WORD      = "hey chat"                    # trigger phrase for hands-free mode

# OBS
OBS_HOST       = "localhost"
OBS_PORT       = 8888
OBS_PASSWORD   = "your_password_here"          # change this
OBS_LINE_WIDTH = 60                            # characters before line wrap
OBS_MAX_LINES  = 8                             # lines per pagination page
OBS_PAGE_DELAY = 3.5                           # seconds between OBS pages
```

<br>

---

<br>

## Trigger Phrases

No slash commands. No special syntax. Just talk.

**Web search**
```
search the web for [query]          look up [query]
search the internet for [query]     can you google [query]
find online [query]
```

**Weather** -- your exact phrasing becomes the search query
```
what's the weather in [city]        will it snow in [city]
temperature in [city] this weekend  forecast for [city]
```

**News**
```
top news        latest headlines        breaking news
what's happening        news today
```

**Wake word**
```
hey chat [your question]
```

<br>

---

<br>

## Personas

Switch from the sidebar dropdown. Each persona is fully editable -- the system prompt is shown live and saved to the database on change.

| Persona | Behaviour |
|---|---|
| **Assistant** | Helpful, concise, gets straight to the point |
| **Coder** | Expert programmer -- always uses code blocks, brief explanations |
| **Creative Writer** | Vivid and imaginative, narrative-first responses |
| **Debate Partner** | Challenges every assumption, argues multiple sides, demands precision |
| **News Analyst** | Objective summaries with context and multiple perspectives noted |
| **Therapist** | Warm and empathetic, asks clarifying questions, avoids direct advice |
| **Tutor** | Patient step-by-step explanations with analogies, checks understanding |

<br>

---

<br>

## OBS Setup

1. In OBS: **Tools > WebSocket Server Settings > Enable**
2. Set port `8888` and a password -- update `OBS_PASSWORD` in `main.py`
3. Add two **Text (GDI+)** sources to your scene named exactly `UserText` and `ChatText`
4. Enable the **OBS** checkbox in the app controls bar
5. Optionally enter your talking and idle scene names in the sidebar OBS section

Recommended text source settings: monospace font (Consolas or Courier New), fixed width, word wrap off -- the app handles wrapping at `OBS_LINE_WIDTH` characters.

<br>

---

<br>

## Gated Models & HuggingFace Token

Some of the most popular models -- including **Meta Llama**, **Google Gemma**, and **Microsoft Phi** -- are *gated*. HuggingFace requires you to accept the model's license agreement and authenticate with a personal access token before you can download them. This is a one-time setup per model family.

**Models that require a token:**

| Requires token | No token needed |
|---|---|
| Meta Llama (all versions) | Qwen (all sizes) |
| Google Gemma (all versions) | Mistral / Mixtral |
| Microsoft Phi-3 / Phi-3.5 | DeepSeek |
| Yi-1.5 | SmolLM2 / TinyLlama / StableLM |

If you try to download a gated model without a token, the download will fail with a `401` or `403` error.

### Step 1 -- Create a HuggingFace account

Go to [huggingface.co](https://huggingface.co) and sign up for free.

### Step 2 -- Generate an access token

1. Click your profile picture (top right) then **Settings**
2. Left sidebar: **Access Tokens**
3. Click **New token**
4. Give it any name (e.g. `ai-chat`), set role to **Read**
5. Click **Generate token** and **copy it immediately** -- it won't be shown again

### Step 3 -- Accept the model license on HuggingFace

Navigate to the model's page on HuggingFace (for example, `meta-llama/Llama-3.1-8B-Instruct`) and click **Agree and access repository**. You need to do this once per model family in your browser while logged in. Without this step the download will fail even with a valid token.

### Step 4 -- Log in from inside the venv

```bat
venv\Scripts\activate
huggingface-cli login
```

Paste your token when prompted and press Enter. It gets saved to `~/.cache/huggingface/token` on your machine and the app picks it up automatically from that point on. You only need to do this once.

> **Note:** Your token is stored locally on your own machine only. It is not read, stored, or transmitted by this app in any way.

<br>

---

<br>

## Troubleshooting

**Model download fails with 401 or 403 error**
The model is gated. You need to: (1) accept the license on the model's HuggingFace page while logged in, and (2) run `huggingface-cli login` inside the venv with a valid access token. See the **Gated Models** section above.

**PyAudio won't install**
```bat
pip install pipwin
pipwin install pyaudio
```
Or download the `.whl` from [Christoph Gohlke's site](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) and install it manually.

**CUDA not detected after install**
```bat
venv\Scripts\activate
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```
If `nvidia-smi` works but PyTorch shows `False`, your driver may predate CUDA 12.x -- update to driver version 520 or newer.

**Model download stalls or corrupts**
Delete the model's subfolder inside `~/ai_models/` and retry from the Load Model dialog. Partial downloads are not resumed automatically.

**OBS not receiving text**
Verify: password matches, source names are exactly `UserText` and `ChatText` (case-sensitive), WebSocket server is running in OBS, and the OBS checkbox is ticked in the app.

**Generation never starts after model loads**
Check the log panel for error output. If Send is still greyed out, the model may still be loading -- the load panel will close automatically when it's ready.

<br>

---

<br>

## Project Structure

```
ai-chat/
|
|-- main.py              # entire application (~2800 lines, single file)
|-- requirements.txt     # pip dependencies
|
|-- setup.bat            # one-time environment setup
|-- run.bat              # launch the app
|-- update.bat           # upgrade all packages
|
|-- README.md
`-- venv/                # created by setup.bat, not committed
```

Runtime directories created automatically:

```
~/ai_models/             # HuggingFace model cache
~/ai_chat_saves/
|-- memory.db            # SQLite database -- sessions and all messages
`-- chat_*.json          # optional session exports
```

<br>

---

<br>

## Dependencies

| Package | Purpose |
|---|---|
| `torch` + CUDA | Inference engine |
| `transformers` | Model loading, tokenization, streaming generation |
| `bitsandbytes` | 4-bit NF4 GPU quantization |
| `accelerate` | Device mapping and mixed precision |
| `huggingface_hub` | Model downloads with per-file progress hooks |
| `PyQt6` | Desktop GUI framework |
| `pyttsx3` | Offline text-to-speech |
| `SpeechRecognition` | Microphone input to text |
| `pyaudio` | Audio capture backend |
| `requests` + `beautifulsoup4` + `lxml` | Web scraping and RSS feed parsing |
| `obs-websocket-py` | OBS WebSocket control |

All installed automatically into the venv by `setup.bat`.

<br>

---

<br>

## License

MIT. Use it, modify it, ship it.

<br>

---

<br>

<div align="center">

Built on &nbsp;[PyTorch](https://pytorch.org) &nbsp;&middot;&nbsp; [HuggingFace Transformers](https://huggingface.co/docs/transformers) &nbsp;&middot;&nbsp; [PyQt6](https://doc.qt.io/qtforpython-6/) &nbsp;&middot;&nbsp; [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)

<br>

*No cloud. No API keys. No subscriptions. Everything runs on your machine.*

<br>
<br>

</div>
