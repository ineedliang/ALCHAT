# AI Chat

A fully-featured local AI chat application built with PyQt6. Run large language models entirely on your own hardware with GPU acceleration, live web search, OBS streaming integration, voice I/O, and a rich dark-themed UI.

---

## Screenshots

> Chat window with animated bubbles, code highlighting, and the sliding log panel

```
[Left sidebar] << [Chat area] >> [Right log panel]
  Personas          Bubbles          Activity log
  News sources      Code blocks      Color-coded
  Temp/Top-P        Typing dots      Per-level filter
  Save/Load         Token counter
  Wake word
  OBS settings
```

---

## Features

### Core Chat
- Animated chat bubbles — user messages right-aligned, assistant left-aligned, fade-in on arrival
- Streaming token output — text appears word-by-word as the model generates
- Code syntax highlighting — auto-detected code blocks render with colors, line numbers, and a one-click Copy button
- Copy button on every assistant response
- Typing indicator with animated dots while the model is thinking

### Models
- 60+ preset models across 10 families (see full list below)
- Model downloader — pick from the categorized dialog or type any HuggingFace model ID
- Live VRAM estimate shown before you download
- 4-bit GPU quantization via bitsandbytes (NF4 + double quant) — cuts VRAM usage roughly in half
- CUDA 12.6 optimized — falls back to CPU automatically if no GPU is found
- Models cached to `~/ai_models` for offline reuse

### Web Search & Scraping
- General web search — say "search the web for...", "look up...", "google..." etc.
- Weather — say "what is the weather in Boston" and it searches that exact phrase live
- News headlines — say "top news" or "latest headlines" to pull from your selected RSS feeds
- Powered by DuckDuckGo HTML scraping — no API key required
- Fetches full page content when snippets are thin
- All scraped context is injected into the model prompt before generation

### Voice
- Mic input — click the mic button and speak
- Wake word detection — say "hey chat" to activate hands-free (configurable)
- TTS output — reads assistant responses aloud via pyttsx3
- Volume and speed sliders in the UI
- TTS runs in a dedicated background thread — never blocks the UI

### OBS Integration
- Streams user input to a `UserText` OBS text source in real time
- Streams assistant response to a `ChatText` OBS text source as it generates
- Full pagination after generation — every line displayed, nothing cut off
- Ticker scroll mode — scrolls text horizontally like a news ticker
- Auto scene switching — switch to a "talking" scene when generating, back to "idle" when done
- Word-wrap enforced at configurable character width

### Sidebar (left panel, toggle with `<<`)
| Section | What it does |
|---|---|
| Persona | 7 built-in AI personalities + editable system prompt |
| Model Parameters | Temperature, Top-P, Max Tokens sliders — live effect |
| Conversations | Save / Load / Clear chat history as JSON |
| Wake Word | Toggle background mic listening |
| OBS Settings | Ticker mode, scene-on-talk, scene-on-idle |
| News Sources | 30+ RSS feeds across 6 categories, checkbox per source |

### Activity Log (right panel, toggle with `>>` or `Log` button)
- Color-coded verbose output: INFO (blue), WARN (yellow), ERROR (red), SUCCESS (green), DEBUG (gray)
- Per-level filter checkboxes — hide DEBUG noise by default
- Logs model load progress, VRAM, token/sec, OBS sends, mic events, search queries, errors
- Clear button and event counter
- Auto-scrolls to latest entry

### Conversations
- Save any chat to JSON with its persona and full history
- Load previously saved chats — bubbles re-render from history
- Saved to `~/ai_chat_saves` by default

---

## Supported Model Families

| Family | Example Models |
|---|---|
| **Qwen3** (default) | 0.6B, 1.7B, 4B, 8B, 14B, 32B |
| **Qwen2.5 Instruct** | 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B |
| **Qwen2.5 Coder** | 1.5B, 7B, 14B, 32B |
| **Microsoft Phi** | Phi-2, Phi-3 mini/small/medium, Phi-3.5 mini |
| **Mistral / Mixtral** | Mistral 7B v0.2/v0.3, Mixtral 8x7B, Nemo |
| **Meta Llama** | Llama 3.2 1B/3B, Llama 3.1 8B/70B, Llama 3 8B |
| **Google Gemma** | Gemma 2 2B/9B/27B, Gemma 3 1B/4B/12B |
| **DeepSeek** | R1 Distill Qwen 1.5B/7B, R1 Distill Llama 8B, Coder |
| **Small / Fast** | TinyLlama 1.1B, StableLM 2 1.6B, SmolLM2 360M/1.7B |
| **Yi / Falcon / Other** | Yi 1.5 6B/9B/34B, Falcon 7B/11B, Zephyr 7B, Vicuna |

Any public HuggingFace model ID can be typed directly into the model dialog.

---

## Requirements

### Hardware
- Windows 10/11 (primary target), Linux/macOS should work with minor adjustments
- NVIDIA GPU recommended (any CUDA-capable card with 4+ GB VRAM for small models)
- CPU-only mode works for models up to ~3B parameters at reasonable speed

### VRAM Guide (with 4-bit quantization enabled)

| Model size | Minimum VRAM |
|---|---|
| 0.5B - 1.7B | 1 - 2 GB (or CPU) |
| 3B - 4B | 2.5 - 3 GB |
| 7B - 8B | 4 - 6 GB |
| 14B | 8 - 10 GB |
| 32B | 18 - 22 GB |
| 70B+ | 40+ GB |

---

## Installation

### Step 1 — Install PyTorch with CUDA 12.6

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

> For CPU-only, use the standard `pip install torch` instead.

### Step 2 — Install all other dependencies

```bash
pip install -r requirements.txt
```

### requirements.txt

```
transformers>=4.43.0
accelerate>=0.30.0
bitsandbytes>=0.43.0
huggingface_hub>=0.23.0
sentencepiece>=0.2.0
protobuf>=4.25.0
PyQt6>=6.6.0
pyttsx3>=2.90
SpeechRecognition>=3.10.0
pyaudio>=0.2.14
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=5.2.0
obs-websocket-py>=1.0.0
```

### PyAudio on Windows

PyAudio often fails with standard pip on Windows. If you hit an error:

```bash
pip install pipwin
pipwin install pyaudio
```

Or download the wheel directly from [Christoph Gohlke's site](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio).

---

## Running

```bash
python main.py
```

The app loads the default model (`Qwen/Qwen3-0.6B`) on startup. Progress is shown in the status bar and the activity log.

---

## OBS Setup

To use the OBS overlay features:

1. In OBS, go to **Tools > WebSocket Server Settings** and enable the server
2. Set your port (default `8888`) and password
3. Update `OBS_HOST`, `OBS_PORT`, and `OBS_PASSWORD` at the top of `main.py`
4. Create two **Text (GDI+)** sources in your scene:
   - `UserText` — for user input
   - `ChatText` — for AI responses
5. Enable **OBS** checkbox in the app and optionally set scene names in the sidebar

Recommended text source settings:
- Font: **Consolas** or **Courier New** (monospace keeps wrapping consistent)
- Uncheck **Wrap** in the source (the app handles wrapping at 60 chars by default)
- Enable **Use custom text extents** and fix the width/height

---

## Configuration

Key constants at the top of `main.py`:

```python
DEFAULT_MODEL  = "Qwen/Qwen3-0.6B"       # model loaded on startup
MODELS_DIR     = "~/ai_models"             # where models are cached
SAVES_DIR      = "~/ai_chat_saves"         # where conversations are saved
WAKE_WORD      = "hey chat"                # wake word phrase
OBS_HOST       = "localhost"
OBS_PORT       = 8888
OBS_PASSWORD   = "your_password_here"
OBS_LINE_WIDTH = 60                        # chars per line for OBS text
OBS_MAX_LINES  = 8                         # max lines shown at once
OBS_PAGE_DELAY = 3.5                       # seconds between OBS pages
```

---

## Voice Trigger Phrases

### Web Search
> "search the web for...", "search the internet for...", "look up...",
> "can you search for...", "google...", "find online..."

### Weather
> "what is the weather in Seattle", "temperature in London",
> "will it rain in Chicago", "forecast for Miami"

### News
> "top news", "latest headlines", "what's happening",
> "current events", "breaking news"

### Wake Word (background mic)
> **"hey chat"** followed by your question (configurable)

---

## Personas

| Name | Behavior |
|---|---|
| Assistant | Helpful, concise general assistant |
| Coder | Expert programmer, responds with clean code blocks |
| Creative Writer | Vivid and imaginative storytelling |
| Debate Partner | Challenges ideas, argues both sides |
| News Analyst | Objective summaries with multiple perspectives |
| Therapist | Warm, empathetic listener |
| Tutor | Patient step-by-step explanations |

The system prompt is fully editable in the sidebar for custom personas.

---

## News Sources

30+ RSS feeds organized by category. Toggle individual sources in the sidebar.

| Category | Sources |
|---|---|
| General / World | BBC, Reuters, AP, Al Jazeera, NPR, Guardian, CBC, Sky News |
| US News | CNN, Fox News, NBC, Washington Post, NYT, USA Today |
| Technology | Hacker News, Ars Technica, The Verge, Wired, TechCrunch, MIT Tech Review, Engadget |
| Science | NASA, New Scientist, Science Daily, Space.com, Live Science |
| Finance | CNBC, MarketWatch, Yahoo Finance, Bloomberg, CoinDesk |
| Sports | ESPN, BBC Sport, Sky Sports, The Athletic |

---

## Project Structure

```
main.py               - Full application (single file)
requirements.txt      - Python dependencies
~/ai_models/          - Downloaded model cache
~/ai_chat_saves/      - Saved conversation JSON files
```

---

## Dependencies

| Package | Purpose |
|---|---|
| PyQt6 | GUI framework |
| transformers | Model loading and inference |
| bitsandbytes | 4-bit GPU quantization |
| accelerate | Device mapping and mixed precision |
| huggingface_hub | Model downloading |
| torch | PyTorch backend |
| pyttsx3 | Text-to-speech |
| SpeechRecognition | Mic input and speech-to-text |
| pyaudio | Audio capture |
| requests | HTTP for web scraping |
| beautifulsoup4 + lxml | HTML parsing |
| obs-websocket-py | OBS WebSocket control |

---

## License

MIT License. Use freely, modify as needed.

---

## Acknowledgements

- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [Qwen3 by Alibaba Cloud](https://huggingface.co/Qwen)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [DuckDuckGo](https://duckduckgo.com) for search (no API key required)
- [Open-Meteo](https://open-meteo.com) weather API
- [OBS WebSocket](https://github.com/obsproject/obs-websocket)
