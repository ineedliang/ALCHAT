"""
AI Chat - Full Featured v3
==========================
Features:
  - Animated chat bubbles, code syntax highlighting, copy buttons
  - Sliding sidebar: news source picker by category, persona switcher,
    conversation save/load, temperature sliders, token counter,
    OBS ticker/scene switcher, wake word mic
  - 4-bit GPU (bitsandbytes + CUDA 12.6), model downloader
  - OBS dual-source streaming + pagination
  - TTS + mic voice input + wake word detection

pip install:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  pip install transformers bitsandbytes accelerate huggingface_hub
  pip install PyQt6 pyttsx3 SpeechRecognition pyaudio
  pip install requests beautifulsoup4 lxml obs-websocket-py
"""

import sys, os, re, json, queue, textwrap, threading, datetime, sqlite3
import requests
from bs4 import BeautifulSoup

import torch
import pyttsx3
import speech_recognition as sr

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer,
    BitsAndBytesConfig
)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QScrollArea, QLineEdit, QPushButton, QCheckBox, QLabel,
    QSlider, QComboBox, QDialog, QProgressBar, QFrame,
    QTextEdit, QSizePolicy, QFileDialog, QGroupBox, QSpinBox,
    QSplitter, QStackedWidget, QTabWidget
)
from PyQt6.QtCore import (
    QThread, pyqtSignal, Qt, QTimer, QPropertyAnimation,
    QEasingCurve, pyqtProperty, QSize, QPoint
)
from PyQt6.QtGui import (
    QFont, QColor, QPalette, QSyntaxHighlighter, QTextCharFormat,
    QTextCursor, QPainter, QLinearGradient, QBrush
)

# ==============================================================
# LOGGER  (singleton - safe to call from any thread)
# ==============================================================
class _Logger(QThread):
    """
    Thread-safe logger. Call logger.log(msg, level) from anywhere.
    Signals are emitted on the Qt main thread via a queue pump.
    Levels: DEBUG, INFO, WARN, ERROR, SUCCESS
    """
    log_message = pyqtSignal(str, str)   # (message, level)

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        super().__init__()          # always call QThread.__init__ first
        if hasattr(self, '_ready'):
            return                  # already initialized, skip rest
        self._q = queue.Queue()
        self._ready = True
        self.daemon = True
        self.start()

    def run(self):
        while True:
            msg, level = self._q.get()
            self.log_message.emit(msg, level)

    def log(self, msg: str, level: str = "INFO"):
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self._q.put((f"[{ts}] [{level}] {msg}", level))

    # Convenience shortcuts
    def debug(self, m): self.log(m, "DEBUG")
    def info(self,  m): self.log(m, "INFO")
    def warn(self,  m): self.log(m, "WARN")
    def error(self, m): self.log(m, "ERROR")
    def ok(self,    m): self.log(m, "SUCCESS")


logger = _Logger()


# ==============================================================
# CONFIG
# ==============================================================
DEFAULT_MODEL  = "Qwen/Qwen2.5-0.5B-Instruct"
MODELS_DIR     = os.path.expanduser("~/ai_models")
SAVES_DIR      = os.path.expanduser("~/ai_chat_saves")
DB_PATH        = os.path.expanduser("~/ai_chat_saves/memory.db")
OBS_HOST       = "localhost"
OBS_PORT       = 8888
OBS_PASSWORD   = "Jackass1!"
OBS_LINE_WIDTH = 60
OBS_MAX_LINES  = 8
OBS_PAGE_DELAY = 3.5
WAKE_WORD      = "hey chat"

PRESET_MODELS = [
    # --- Qwen3 (latest, 2025) ---
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",

    # --- Qwen2.5 Instruct ---
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",

    # --- Qwen2.5 Coder ---
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",

    # --- Microsoft Phi ---
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-mini-128k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-3-small-8k-instruct",
    "microsoft/Phi-3-medium-4k-instruct",
    "microsoft/phi-2",

    # --- Mistral / Mixtral ---
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-Nemo-Instruct-2407",

    # --- Meta Llama ---
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",

    # --- Google Gemma ---
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",

    # --- Small / Fast ---
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "stabilityai/stablelm-2-1_6b-chat",
    "stabilityai/stablelm-zephyr-3b",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",

    # --- DeepSeek ---
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    "deepseek-ai/deepseek-coder-6.7b-instruct",

    # --- Yi ---
    "01-ai/Yi-1.5-6B-Chat",
    "01-ai/Yi-1.5-9B-Chat",
    "01-ai/Yi-1.5-34B-Chat",

    # --- Falcon ---
    "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-11b",

    # --- OpenChat / Vicuna / Zephyr ---
    "openchat/openchat-3.5-0106",
    "HuggingFaceH4/zephyr-7b-beta",
    "lmsys/vicuna-7b-v1.5",
    "lmsys/vicuna-13b-v1.5",
]

PERSONAS = {
    "Assistant":      "You are a helpful, concise assistant.",
    "Coder":          "You are an expert programmer. Respond with clean, well-commented code and brief explanations. Use code blocks.",
    "Creative Writer":"You are a creative writing assistant. Be vivid, imaginative, and engaging.",
    "Debate Partner": "You are a sharp debate partner. Challenge ideas, argue both sides, and push for clarity.",
    "News Analyst":   "You are a news analyst. Summarize events objectively, highlight key context, and note different perspectives.",
    "Therapist":      "You are a compassionate listener. Respond warmly, ask clarifying questions, and avoid giving direct advice.",
    "Tutor":          "You are a patient tutor. Explain concepts step by step, use analogies, and check for understanding.",
}

# Full categorised news sources
ALL_NEWS_SOURCES = {
    "General / World": [
        ("BBC News",          "https://feeds.bbci.co.uk/news/rss.xml"),
        ("Reuters",           "https://feeds.reuters.com/reuters/topNews"),
        ("AP News",           "https://rsshub.app/apnews/topics/apf-topnews"),
        ("Al Jazeera",        "https://www.aljazeera.com/xml/rss/all.xml"),
        ("NPR",               "https://feeds.npr.org/1001/rss.xml"),
        ("The Guardian",      "https://www.theguardian.com/world/rss"),
        ("CBC News",          "https://www.cbc.ca/cmlink/rss-topstories"),
        ("Sky News",          "https://feeds.skynews.com/feeds/rss/world.xml"),
    ],
    "US News": [
        ("CNN",               "http://rss.cnn.com/rss/cnn_topstories.rss"),
        ("Fox News",          "https://moxie.foxnews.com/google-publisher/latest.xml"),
        ("NBC News",          "https://feeds.nbcnews.com/nbcnews/public/news"),
        ("Washington Post",   "https://feeds.washingtonpost.com/rss/national"),
        ("NYT",               "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"),
        ("USA Today",         "https://rssfeeds.usatoday.com/usatoday-NewsTopStories"),
    ],
    "Technology": [
        ("Hacker News",       "https://hnrss.org/frontpage"),
        ("Ars Technica",      "https://feeds.arstechnica.com/arstechnica/index"),
        ("The Verge",         "https://www.theverge.com/rss/index.xml"),
        ("Wired",             "https://www.wired.com/feed/rss"),
        ("TechCrunch",        "https://techcrunch.com/feed/"),
        ("MIT Tech Review",   "https://www.technologyreview.com/feed/"),
        ("Engadget",          "https://www.engadget.com/rss.xml"),
    ],
    "Science": [
        ("NASA",              "https://www.nasa.gov/rss/dyn/breaking_news.rss"),
        ("New Scientist",     "https://www.newscientist.com/feed/home/"),
        ("Science Daily",     "https://www.sciencedaily.com/rss/all.xml"),
        ("Space.com",         "https://www.space.com/feeds/all"),
        ("Live Science",      "https://www.livescience.com/feeds/all"),
    ],
    "Finance": [
        ("CNBC",              "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
        ("MarketWatch",       "https://feeds.marketwatch.com/marketwatch/topstories/"),
        ("Yahoo Finance",     "https://finance.yahoo.com/news/rssindex"),
        ("Bloomberg",         "https://feeds.bloomberg.com/markets/news.rss"),
        ("CoinDesk",          "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ],
    "Sports": [
        ("ESPN",              "https://www.espn.com/espn/rss/news"),
        ("BBC Sport",         "https://feeds.bbci.co.uk/sport/rss.xml"),
        ("Sky Sports",        "https://www.skysports.com/rss/12040"),
        ("The Athletic",      "https://theathletic.com/rss/"),
    ],
}

os.makedirs(SAVES_DIR, exist_ok=True)

# ==============================================================
# MEMORY DATABASE  (SQLite - persists across restarts)
# ==============================================================
class MemoryDB:
    """
    Persistent conversation memory backed by SQLite.

    Schema:
      sessions  (id, name, persona, created_at, updated_at)
      messages  (id, session_id, role, content, created_at)

    One "active" session is tracked at a time.
    All writes happen on the calling thread ? SQLite handles locking.
    Each new app launch continues the last session automatically.
    """

    def __init__(self, db_path: str = DB_PATH):
        self._path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._session_id: int = None
        self._init_schema()
        logger.info(f"MemoryDB opened: {db_path}")

    # ---- schema ------------------------------------------

    def _init_schema(self):
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT    NOT NULL DEFAULT 'Chat',
                    persona     TEXT    NOT NULL DEFAULT '',
                    created_at  TEXT    NOT NULL,
                    updated_at  TEXT    NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role        TEXT    NOT NULL,
                    content     TEXT    NOT NULL,
                    created_at  TEXT    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id);
            """)
            self._conn.commit()

    # ---- session management ------------------------------

    def new_session(self, name: str = None, persona: str = "") -> int:
        now  = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
        name = name or f"Chat {now[:16]}"
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sessions (name, persona, created_at, updated_at) VALUES (?,?,?,?)",
                (name, persona, now, now)
            )
            self._conn.commit()
            self._session_id = cur.lastrowid
        logger.ok(f"MemoryDB: new session #{self._session_id} '{name}'")
        return self._session_id

    def resume_last_session(self) -> int | None:
        """Load the most recent session. Returns session_id or None if no sessions exist."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id, name, persona FROM sessions ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        if row:
            self._session_id = row[0]
            logger.ok(f"MemoryDB: resumed session #{row[0]} '{row[1]}'")
            return row[0]
        return None

    def set_session(self, session_id: int):
        self._session_id = session_id
        logger.info(f"MemoryDB: switched to session #{session_id}")

    def rename_session(self, session_id: int, name: str):
        now = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET name=?, updated_at=? WHERE id=?",
                (name, now, session_id)
            )
            self._conn.commit()

    def update_persona(self, persona: str):
        if not self._session_id:
            return
        now = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET persona=?, updated_at=? WHERE id=?",
                (persona, now, self._session_id)
            )
            self._conn.commit()

    def delete_session(self, session_id: int):
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            self._conn.commit()
        logger.info(f"MemoryDB: deleted session #{session_id}")
        if self._session_id == session_id:
            self._session_id = None

    def list_sessions(self) -> list[dict]:
        """Return all sessions newest-first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, persona, created_at, updated_at, "
                "(SELECT COUNT(*) FROM messages WHERE session_id=sessions.id) as msg_count "
                "FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [
            {"id": r[0], "name": r[1], "persona": r[2],
             "created_at": r[3], "updated_at": r[4], "msg_count": r[5]}
            for r in rows
        ]

    def get_session(self, session_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, name, persona, created_at, updated_at FROM sessions WHERE id=?",
                (session_id,)
            ).fetchone()
        if row:
            return {"id": row[0], "name": row[1], "persona": row[2],
                    "created_at": row[3], "updated_at": row[4]}
        return None

    # ---- message management ------------------------------

    def append(self, role: str, content: str):
        """Append a message to the active session. Auto-creates session if needed."""
        if not self._session_id:
            self.new_session()
        now = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
                (self._session_id, role, content, now)
            )
            self._conn.execute(
                "UPDATE sessions SET updated_at=? WHERE id=?",
                (now, self._session_id)
            )
            self._conn.commit()
        logger.debug(f"MemoryDB: saved [{role}] ({len(content)} chars) to session #{self._session_id}")

    def load_messages(self, session_id: int = None) -> list[dict]:
        """Return all messages for a session as {role, content} dicts."""
        sid = session_id or self._session_id
        if not sid:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM messages WHERE session_id=? ORDER BY id ASC",
                (sid,)
            ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]

    def clear_messages(self, session_id: int = None):
        sid = session_id or self._session_id
        if not sid:
            return
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE session_id=?", (sid,))
            self._conn.commit()
        logger.info(f"MemoryDB: cleared messages in session #{sid}")

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search across all messages."""
        pattern = f"%{query}%"
        with self._lock:
            rows = self._conn.execute(
                "SELECT m.role, m.content, m.created_at, s.name "
                "FROM messages m JOIN sessions s ON m.session_id=s.id "
                "WHERE m.content LIKE ? ORDER BY m.id DESC LIMIT ?",
                (pattern, limit)
            ).fetchall()
        return [{"role": r[0], "content": r[1], "created_at": r[2], "session": r[3]}
                for r in rows]

    @property
    def session_id(self):
        return self._session_id


# Global instance
memory_db = MemoryDB()

# ==============================================================
# OBS
# ==============================================================
try:
    from obswebsocket import obsws, requests as obs_req
    OBS_AVAILABLE = True
except ImportError:
    OBS_AVAILABLE = False


def format_for_obs(text: str) -> str:
    wrapped = textwrap.fill(text.strip(), width=OBS_LINE_WIDTH)
    lines = wrapped.splitlines()
    return "\n".join(lines[-OBS_MAX_LINES:])


def obs_pages(text: str):
    wrapped = textwrap.fill(text.strip(), width=OBS_LINE_WIDTH)
    lines = wrapped.splitlines()
    return ["\n".join(lines[i:i + OBS_MAX_LINES])
            for i in range(0, max(len(lines), 1), OBS_MAX_LINES)]


class OBSWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._q = queue.Queue()
        self.ticker_mode   = False
        self.scene_on_talk = ""
        self.scene_on_idle = ""

    def _send(self, source: str, text: str):
        if not OBS_AVAILABLE:
            logger.debug(f"OBS not available, skipping send to '{source}'")
            return
        try:
            logger.debug(f"OBS -> '{source}': {repr(text[:40])}")
            ws = obsws(OBS_HOST, OBS_PORT, OBS_PASSWORD)
            ws.connect()
            ws.call(obs_req.SetInputSettings(
                inputName=source, inputSettings={"text": text}, overlay=True
            ))
            ws.disconnect()
            logger.debug(f"OBS send OK -> '{source}'")
        except Exception as e:
            logger.error(f"OBS send failed '{source}': {e}")

    def _switch_scene(self, scene: str):
        if not OBS_AVAILABLE or not scene:
            return
        try:
            logger.info(f"OBS switching scene -> '{scene}'")
            ws = obsws(OBS_HOST, OBS_PORT, OBS_PASSWORD)
            ws.connect()
            ws.call(obs_req.SetCurrentProgramScene(sceneName=scene))
            ws.disconnect()
            logger.ok(f"OBS scene switched to '{scene}'")
        except Exception as e:
            logger.error(f"OBS scene switch failed: {e}")

    def _drain(self):
        while not self._q.empty():
            try: self._q.get_nowait()
            except queue.Empty: break

    def run(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            kind = item[0]
            if kind == "update":
                self._send(item[1], item[2])
            elif kind == "paginate":
                for page in item[2]:
                    if not self._q.empty():
                        break
                    self._send(item[1], page)
                    threading.Event().wait(OBS_PAGE_DELAY)
            elif kind == "ticker":
                # Scroll text left one word at a time
                words = item[2].split()
                buf   = words[:10]
                for i in range(len(words)):
                    if not self._q.empty():
                        break
                    self._send(item[1], " ".join(buf))
                    if i + 10 < len(words):
                        buf = words[i+1:i+11]
                    threading.Event().wait(0.4)
            elif kind == "scene":
                self._switch_scene(item[1])

    def update(self, source: str, text: str):
        self._drain()
        self._q.put(("update", source, text))

    def paginate(self, source: str, text: str):
        self._drain()
        if self.ticker_mode:
            self._q.put(("ticker", source, text))
        else:
            self._q.put(("paginate", source, obs_pages(text)))

    def switch_scene(self, scene: str):
        self._q.put(("scene", scene))


obs_worker = OBSWorker()
obs_worker.start()

# ==============================================================
# TTS
# ==============================================================
class TTSWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._q = queue.Queue()
        self._engine = pyttsx3.init()

    def get_voices(self):
        return self._engine.getProperty('voices')

    def run(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            text, vol, rate, voice_id = item
            logger.info(f"TTS speaking ({len(text)} chars, vol={vol:.1f}, rate={rate})")
            self._engine.setProperty('volume', vol)
            self._engine.setProperty('rate', rate)
            if voice_id:
                self._engine.setProperty('voice', voice_id)
                logger.debug(f"TTS voice: {voice_id}")
            self._engine.say(text)
            self._engine.runAndWait()
            logger.debug("TTS finished")

    def speak(self, text, volume=0.8, rate=200, voice_id=None):
        while not self._q.empty():
            try: self._q.get_nowait()
            except queue.Empty: break
        self._q.put((text, volume, rate, voice_id))


tts_worker = TTSWorker()
tts_worker.start()

# ==============================================================
# NEWS SCRAPER
# ==============================================================
class NewsScraper:
    _enabled_sources: list = []  # set by sidebar

    @classmethod
    def set_sources(cls, sources: list):
        cls._enabled_sources = sources

    @classmethod
    def fetch(cls, max_stories=10) -> str:
        sources = cls._enabled_sources or [
            s for cat in ALL_NEWS_SOURCES.values() for s in cat
        ]
        stories = []
        for name, url in sources:
            if len(stories) >= max_stories:
                break
            try:
                r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
                soup = BeautifulSoup(r.content, "lxml-xml")
                for item in soup.find_all("item"):
                    title = item.find("title")
                    desc  = item.find("description")
                    if title:
                        t = title.get_text(strip=True)
                        d = desc.get_text(strip=True)[:120] if desc else ""
                        stories.append(f"[{name}] {t}: {d}")
                    if len(stories) >= max_stories:
                        break
            except Exception:
                continue
        if not stories:
            return "Could not fetch news at this time."
        return "TOP NEWS STORIES:\n" + "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(stories)
        )

    @staticmethod
    def is_news_request(text: str) -> bool:
        kw = ["news", "headlines", "top stories", "whats happening",
              "what's happening", "latest news", "current events", "breaking"]
        return any(k in text.lower() for k in kw)


# ==============================================================
# MODEL MANAGER
# ==============================================================
class ModelManager:
    def __init__(self):
        self.model        = None
        self.tokenizer    = None
        self.current_name = None
        self.temperature  = 0.7
        self.top_p        = 0.9
        self.max_tokens   = 512
        os.makedirs(MODELS_DIR, exist_ok=True)

    def load(self, model_name: str, use_4bit: bool = True,
             progress_cb=None, stage_cb=None):
        def prog(msg):
            logger.info(msg)
            if progress_cb: progress_cb(msg)
        def stg(s):
            logger.info(f"Stage: {s}")
            if stage_cb: stage_cb(s)

        prog(f"Loading model: {model_name}")
        prog(f"4-bit quant: {use_4bit}, CUDA: {torch.cuda.is_available()}")
        stg("tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, cache_dir=MODELS_DIR
        )
        logger.ok("Tokenizer loaded")
        prog("Tokenizer ready")
        kwargs = dict(trust_remote_code=True, device_map="auto", cache_dir=MODELS_DIR)
        if use_4bit and torch.cuda.is_available():
            prog("Applying 4-bit NF4 quantization config...")
            stg("quantization")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            prog(f"Using dtype: {dtype}")
            kwargs["torch_dtype"] = dtype
        stg("weights")
        prog("Downloading / loading weights (may take a while)...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()
        self.current_name = model_name
        stg("cuda")
        device = next(self.model.parameters()).device
        logger.ok(f"Model ready on {device}")
        prog(f"Model on {device}")
        try:
            n = sum(p.numel() for p in self.model.parameters()) / 1e6
            prog(f"Parameters: {n:.0f}M")
            logger.info(f"Model params: {n:.0f}M")
        except Exception:
            pass
        stg("ready")

    def count_tokens(self, messages: list) -> int:
        if not self.tokenizer:
            return 0
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return len(self.tokenizer.encode(text))
        except Exception:
            return 0

    def generate(self, messages, streamer):
        text   = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        kwargs = dict(
            **inputs, streamer=streamer,
            max_new_tokens=self.max_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=1.15,
            eos_token_id=self.tokenizer.eos_token_id
        )
        t = threading.Thread(target=self.model.generate, kwargs=kwargs)
        t.start(); t.join()


model_manager = ModelManager()

# ==============================================================
# THREADS
# ==============================================================
class DownloadThread(QThread):
    progress        = pyqtSignal(str)          # status text
    stage           = pyqtSignal(str)          # current stage label
    file_progress   = pyqtSignal(str, int, int) # filename, downloaded, total
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, model_name, use_4bit):
        super().__init__()
        self.model_name = model_name
        self.use_4bit   = use_4bit

    def run(self):
        try:
            # Patch huggingface_hub to intercept file download progress
            self._install_hf_hooks()
            self.stage.emit("tokenizer")
            self.progress.emit(f"Loading tokenizer for {self.model_name}...")
            model_manager.load(self.model_name, self.use_4bit,
                               progress_cb=self._on_progress,
                               stage_cb=self._on_stage)
            self.finished_signal.emit(True, "")
        except Exception as e:
            self.finished_signal.emit(False, str(e))
        finally:
            self._remove_hf_hooks()

    def _on_progress(self, msg: str):
        self.progress.emit(msg)

    def _on_stage(self, stage: str):
        self.stage.emit(stage)

    def _install_hf_hooks(self):
        """Monkey-patch huggingface_hub tqdm to capture per-file download progress."""
        try:
            import huggingface_hub.file_download as fd
            original_tqdm = fd.tqdm
            thread = self

            class _PatchedTqdm:
                def __init__(self, *a, **kw):
                    self._total = kw.get("total", 0) or 0
                    self._n     = 0
                    self._desc  = kw.get("desc", "") or ""
                    self._inner = original_tqdm(*a, **kw)

                def update(self, n=1):
                    self._n += n
                    fname = self._desc.split("/")[-1][:40] if self._desc else "file"
                    thread.file_progress.emit(fname, self._n, self._total)
                    self._inner.update(n)

                def __enter__(self):
                    self._inner.__enter__()
                    return self

                def __exit__(self, *a):
                    return self._inner.__exit__(*a)

                def set_postfix(self, *a, **kw):
                    return self._inner.set_postfix(*a, **kw)

                def close(self):
                    return self._inner.close()

            self._orig_tqdm = original_tqdm
            self._fd_module = fd
            fd.tqdm = _PatchedTqdm
        except Exception:
            pass

    def _remove_hf_hooks(self):
        try:
            self._fd_module.tqdm = self._orig_tqdm
        except Exception:
            pass


class GenerationThread(QThread):
    new_token       = pyqtSignal(str)
    finished_signal = pyqtSignal()
    error_signal    = pyqtSignal(str)

    def __init__(self, messages):
        super().__init__()
        self.messages = messages

    def run(self):
        try:
            logger.info(f"Generation start | temp={model_manager.temperature} "
                        f"top_p={model_manager.top_p} max_tok={model_manager.max_tokens}")
            t0 = datetime.datetime.now()
            token_count = 0
            streamer = TextIteratorStreamer(
                model_manager.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            gen_t = threading.Thread(
                target=model_manager.generate, args=(self.messages, streamer)
            )
            gen_t.start()
            for token in streamer:
                token_count += 1
                self.new_token.emit(token)
            gen_t.join()
            elapsed = (datetime.datetime.now() - t0).total_seconds()
            tps = token_count / elapsed if elapsed > 0 else 0
            logger.ok(f"Generation done | {token_count} tokens | {elapsed:.1f}s | {tps:.1f} tok/s")
        except Exception as e:
            logger.error(f"Generation error: {e}")
            self.error_signal.emit(str(e))
        finally:
            self.finished_signal.emit()


class MicThread(QThread):
    result      = pyqtSignal(str)
    error       = pyqtSignal(str)
    status      = pyqtSignal(str)
    wake_heard  = pyqtSignal()

    def __init__(self, wake_word_mode=False):
        super().__init__()
        self.wake_word_mode = wake_word_mode
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        rec = sr.Recognizer()
        rec.energy_threshold = 300
        try:
            with sr.Microphone() as src:
                logger.info("Mic opened, adjusting for ambient noise...")
                rec.adjust_for_ambient_noise(src, duration=0.4)
                logger.debug(f"Mic energy threshold: {rec.energy_threshold:.0f}")
                if self.wake_word_mode:
                    self.status.emit("Wake word: listening...")
                    logger.info(f"Wake word mode active, listening for: '{WAKE_WORD}'")
                    while self._running:
                        try:
                            audio = rec.listen(src, timeout=2, phrase_time_limit=4)
                            text  = rec.recognize_google(audio).lower()
                            logger.debug(f"Wake word heard fragment: '{text}'")
                            if WAKE_WORD in text:
                                logger.ok(f"Wake word triggered!")
                                self.wake_heard.emit()
                                self.status.emit("Wake word heard! Speak now...")
                                audio2 = rec.listen(src, timeout=5, phrase_time_limit=15)
                                cmd = rec.recognize_google(audio2)
                                logger.info(f"Wake command captured: '{cmd}'")
                                self.result.emit(cmd)
                                self.status.emit("Wake word: listening...")
                        except (sr.WaitTimeoutError, sr.UnknownValueError):
                            continue
                        except Exception as e:
                            logger.error(f"Wake word loop error: {e}")
                            self.error.emit(str(e))
                            break
                else:
                    self.status.emit("Listening...")
                    logger.info("Mic listening for speech...")
                    audio = rec.listen(src, timeout=8, phrase_time_limit=15)
                    self.status.emit("Processing speech...")
                    logger.info("Sending audio to Google Speech API...")
                    result = rec.recognize_google(audio)
                    logger.ok(f"Speech recognized: '{result}'")
                    self.result.emit(result)
        except sr.WaitTimeoutError:
            logger.warn("Mic: no speech detected (timeout)")
            self.error.emit("No speech detected.")
        except sr.UnknownValueError:
            logger.warn("Mic: could not understand audio")
            self.error.emit("Could not understand audio.")
        except Exception as e:
            logger.error(f"Mic error: {e}")
            self.error.emit(str(e))


class NewsFetchThread(QThread):
    finished_signal = pyqtSignal(str)

    def run(self):
        logger.info("Fetching news headlines...")
        result = NewsScraper.fetch()
        logger.ok(f"News fetch complete ({result.count(chr(10))} stories)")
        self.finished_signal.emit(result)


# ==============================================================
# WEB SCRAPER  (DuckDuckGo HTML - no API key needed)
# ==============================================================
class WebScraper:
    """
    Scrapes DuckDuckGo search results and fetches page content.
    Used for:
      - General web searches ("search the web for X")
      - Weather ("what is the temperature in Boston")
      - Any ad-hoc query the user wants grounded in live data
    """

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    # ---- keyword detection --------------------------------

    @staticmethod
    def is_web_search_request(text: str) -> bool:
        kw = [
            "search the web", "search the internet", "search online",
            "look it up", "look up", "google", "find online",
            "search for", "can you search", "web search", "internet search",
        ]
        return any(k in text.lower() for k in kw)

    @staticmethod
    def is_weather_request(text: str) -> bool:
        kw = [
            "weather", "temperature", "forecast", "how hot", "how cold",
            "raining", "sunny", "snowing", "humidity", "wind speed",
            "degrees", "temp in", "weather in", "climate in",
            "what is it like in", "outside temp",
        ]
        return any(k in text.lower() for k in kw)

    # ---- query extraction ---------------------------------

    @staticmethod
    def extract_search_query(text: str) -> str:
        """
        Strip trigger phrases and return the raw search query.
        e.g. "search the web for quantum computing" -> "quantum computing"
        e.g. "what is the temperature in Boston"   -> "what is the temperature in Boston"
        """
        lower = text.lower()
        prefixes = [
            "search the web for ", "search the internet for ",
            "search online for ", "search for ", "can you search for ",
            "can you search the web for ", "can you search the internet for ",
            "look up ", "look it up ", "google ", "find online ",
            "web search for ", "internet search for ",
        ]
        for p in prefixes:
            if lower.startswith(p):
                return text[len(p):].strip()
            if p in lower:
                idx = lower.index(p)
                return text[idx + len(p):].strip()
        # No prefix found - use the whole message as-is (good for weather queries)
        return text.strip()

    # ---- DuckDuckGo search --------------------------------

    @classmethod
    def search(cls, query: str, max_results: int = 6) -> str:
        """
        Search DuckDuckGo and return a formatted context block.
        Falls back to fetching the top result's page content if snippets are thin.
        """
        logger.info(f"Web search: '{query}'")
        try:
            url = "https://html.duckduckgo.com/html/"
            resp = requests.post(
                url,
                data={"q": query, "b": "", "kl": "us-en"},
                headers=cls.HEADERS,
                timeout=10,
            )
            soup = BeautifulSoup(resp.text, "lxml")

            results = []
            links   = []
            for r in soup.select(".result__body")[:max_results]:
                title_tag   = r.select_one(".result__title")
                snippet_tag = r.select_one(".result__snippet")
                url_tag     = r.select_one(".result__url")
                link_tag    = r.select_one("a.result__a")

                title   = title_tag.get_text(strip=True)   if title_tag   else ""
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                src_url = url_tag.get_text(strip=True)     if url_tag     else ""
                href    = link_tag["href"]                  if link_tag    else ""

                if title or snippet:
                    results.append(f"[{src_url}]\n{title}\n{snippet}")
                if href and len(links) < 2:
                    links.append(href)

            if not results:
                logger.warn("DuckDuckGo returned no results, trying fallback fetch")
                return cls._fallback_fetch(query)

            logger.ok(f"Web search returned {len(results)} results for '{query}'")

            # If snippets are very short, enrich by fetching the top page
            total_text = " ".join(results)
            if len(total_text) < 400 and links:
                logger.info(f"Snippets thin ({len(total_text)} chars), fetching top page...")
                page_text = cls._fetch_page(links[0])
                if page_text:
                    results.append(f"[Full page content]\n{page_text}")

            header = f"WEB SEARCH RESULTS for: \"{query}\"\n" + ("=" * 50)
            return header + "\n\n" + "\n\n".join(results)

        except Exception as e:
            logger.error(f"Web search error: {e}")
            return f"Web search failed for '{query}': {e}"

    @classmethod
    def _fetch_page(cls, url: str, max_chars: int = 2000) -> str:
        """Fetch a page and return cleaned plain text."""
        try:
            logger.debug(f"Fetching page: {url}")
            resp = requests.get(url, headers=cls.HEADERS, timeout=8)
            soup = BeautifulSoup(resp.text, "lxml")
            # Remove noise tags
            for tag in soup(["script", "style", "nav", "footer",
                             "header", "aside", "form", "iframe"]):
                tag.decompose()
            text = " ".join(soup.get_text(" ", strip=True).split())
            logger.ok(f"Page fetched: {len(text)} chars from {url[:60]}")
            return text[:max_chars]
        except Exception as e:
            logger.warn(f"Page fetch failed: {e}")
            return ""

    @classmethod
    def _fallback_fetch(cls, query: str) -> str:
        """Last resort: try fetching a direct Google snippet URL."""
        try:
            url = f"https://duckduckgo.com/?q={requests.utils.quote(query)}&ia=answer"
            resp = requests.get(url, headers=cls.HEADERS, timeout=8)
            soup = BeautifulSoup(resp.text, "lxml")
            answer = soup.select_one(".zci__result, .answer, [data-testid='answer']")
            if answer:
                return f"Answer: {answer.get_text(strip=True)}"
        except Exception as e:
            logger.error(f"Fallback fetch failed: {e}")
        return "No results found."


class WebSearchThread(QThread):
    finished_signal = pyqtSignal(str)

    def __init__(self, query: str):
        super().__init__()
        self.query = query

    def run(self):
        self.finished_signal.emit(WebScraper.search(self.query))


class WeatherFetchThread(QThread):
    finished_signal = pyqtSignal(str)

    def __init__(self, query: str):
        super().__init__()
        self.query = query

    def run(self):
        # Use the user's exact query as the search term - most accurate approach
        result = WebScraper.search(self.query, max_results=4)
        self.finished_signal.emit(result)


# ==============================================================
# CODE SYNTAX HIGHLIGHTER
# ==============================================================
class CodeHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rules = []

        def add(pattern, color, bold=False, italic=False):
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(color))
            if bold:   fmt.setFontWeight(QFont.Weight.Bold)
            if italic: fmt.setFontItalic(True)
            self._rules.append((re.compile(pattern), fmt))

        # Keywords
        kw = r"\b(def|class|import|from|return|if|else|elif|for|while|in|not|and|or|True|False|None|try|except|finally|with|as|pass|break|continue|lambda|yield|async|await|raise|del|global|nonlocal|is)\b"
        add(kw, "#c792ea", bold=True)
        # Strings
        add(r'"[^"\\]*(?:\\.[^"\\]*)*"', "#c3e88d")
        add(r"'[^'\\]*(?:\\.[^'\\]*)*'", "#c3e88d")
        # Comments
        add(r"#[^\n]*",      "#546e7a", italic=True)
        # Numbers
        add(r"\b\d+\.?\d*\b","#f78c6c")
        # Functions
        add(r"\b([a-zA-Z_]\w*)\s*(?=\()", "#82aaff")
        # Decorators
        add(r"@\w+",         "#ffcb6b")

    def highlightBlock(self, text):
        for pattern, fmt in self._rules:
            for m in pattern.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), fmt)


# ==============================================================
# CHAT BUBBLE
# ==============================================================
class CodeBlock(QWidget):
    """Rendered code block with syntax highlighting and copy button."""
    def __init__(self, code: str, lang: str = "", parent=None):
        super().__init__(parent)
        self.setStyleSheet("background:transparent;")
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 4, 0, 4)
        v.setSpacing(0)

        # Header bar
        hdr = QWidget()
        hdr.setStyleSheet("background:#1a1a2e;border-radius:6px 6px 0 0;")
        hdr.setFixedHeight(28)
        hr = QHBoxLayout(hdr); hr.setContentsMargins(10, 0, 6, 0)
        lbl = QLabel(lang or "code")
        lbl.setStyleSheet("color:#546e7a;font-size:10px;font-family:'Consolas';")
        hr.addWidget(lbl); hr.addStretch()
        copy_btn = QPushButton("Copy")
        copy_btn.setFixedSize(50, 20)
        copy_btn.setStyleSheet("""
            QPushButton{background:#0f3460;color:#a0c0ff;border:none;
                border-radius:3px;font-size:10px;}
            QPushButton:hover{background:#e94560;color:white;}
        """)
        copy_btn.clicked.connect(lambda: (
            QApplication.clipboard().setText(code),
            copy_btn.setText("Copied!"),
            QTimer.singleShot(1500, lambda: copy_btn.setText("Copy"))
        ))
        hr.addWidget(copy_btn)
        v.addWidget(hdr)

        # Code editor (read-only, highlighted)
        editor = QTextEdit()
        editor.setReadOnly(True)
        editor.setPlainText(code)
        editor.setFont(QFont("Consolas", 10))
        editor.setStyleSheet("""
            QTextEdit{background:#0d1117;color:#cdd9e5;
                border:none;border-radius:0 0 6px 6px;padding:8px;}
        """)
        lines = code.count("\n") + 1
        editor.setFixedHeight(min(max(lines * 19 + 20, 60), 400))
        CodeHighlighter(editor.document())
        v.addWidget(editor)


class ChatBubble(QFrame):
    def __init__(self, text: str, role: str, parent=None):
        super().__init__(parent)
        self.role     = role
        self._full    = text
        is_user       = (role == "user")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 3, 8, 3)
        outer.setSpacing(4)

        # Parse code blocks out of text
        parts = re.split(r"```(\w*)\n?(.*?)```", text, flags=re.DOTALL)
        # parts: [text, lang, code, text, lang, code, ...]

        content_w = QWidget()
        content_w.setStyleSheet("background:transparent;")
        cw_lay = QVBoxLayout(content_w)
        cw_lay.setContentsMargins(0, 0, 0, 0)
        cw_lay.setSpacing(4)

        i = 0
        has_code = False
        while i < len(parts):
            chunk = parts[i]
            if chunk.strip():
                lbl = QLabel(chunk.strip())
                lbl.setWordWrap(True)
                lbl.setMaximumWidth(560)
                lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                lbl.setFont(QFont("Consolas" if is_user else "Segoe UI", 10))
                if is_user:
                    lbl.setStyleSheet("""
                        background:qlineargradient(x1:0,y1:0,x2:1,y2:1,
                            stop:0 #e94560,stop:1 #b01030);
                        color:white;border-radius:14px 14px 2px 14px;padding:10px 14px;
                    """)
                else:
                    lbl.setStyleSheet("""
                        background:qlineargradient(x1:0,y1:0,x2:1,y2:1,
                            stop:0 #16213e,stop:1 #0f3460);
                        color:#c8d8ff;border-radius:14px 14px 14px 2px;
                        padding:10px 14px;border:1px solid #1a2a5e;
                    """)
                cw_lay.addWidget(lbl)
            i += 1
            if i + 1 < len(parts):
                lang = parts[i]; code = parts[i+1]
                if code.strip():
                    has_code = True
                    cb = CodeBlock(code.strip(), lang)
                    cb.setMaximumWidth(620)
                    cw_lay.addWidget(cb)
                i += 2

        # Copy button for whole bubble (non-code)
        if not is_user:
            btn_row = QHBoxLayout()
            btn_row.setContentsMargins(4, 0, 0, 0)
            copy_all = QPushButton("Copy response")
            copy_all.setFixedHeight(20)
            copy_all.setStyleSheet("""
                QPushButton{background:transparent;color:#404070;border:none;font-size:10px;}
                QPushButton:hover{color:#e94560;}
            """)
            copy_all.clicked.connect(lambda: (
                QApplication.clipboard().setText(self._full),
                copy_all.setText("Copied!"),
                QTimer.singleShot(1500, lambda: copy_all.setText("Copy response"))
            ))
            btn_row.addWidget(copy_all)
            btn_row.addStretch()
            cw_lay.addLayout(btn_row)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        if is_user:
            row.addStretch(); row.addWidget(content_w)
        else:
            row.addWidget(content_w); row.addStretch()
        outer.addLayout(row)

        self.setStyleSheet("ChatBubble{background:transparent;border:none;}")
        self._text_labels = [
            w for w in content_w.findChildren(QLabel) if not isinstance(w, QPushButton)
        ]

        # Fade-in animation
        self._opacity = 0.0
        self._anim = QPropertyAnimation(self, b"_op")
        self._anim.setDuration(280)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _get_op(self): return self._opacity
    def _set_op(self, v):
        self._opacity = v
        self.setStyleSheet(f"ChatBubble{{background:transparent;border:none;opacity:{v:.2f};}}")
    _op = pyqtProperty(float, _get_op, _set_op)

    def show_animated(self):
        self.show(); self._anim.start()

    def update_text(self, text: str):
        self._full = text
        # Only update plain text labels (no code blocks mid-stream)
        plain = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()
        if self._text_labels:
            self._text_labels[0].setText(plain)


# ==============================================================
# TYPING DOTS
# ==============================================================
class TypingDots(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._n = 0
        self._t = QTimer(self)
        self._t.timeout.connect(self._tick)
        self.setStyleSheet("background:#16213e;color:#5060a0;border-radius:10px;"
                           "padding:6px 14px;font-size:14px;font-family:monospace;")
        self.setFixedWidth(72)
        self.setText("o o o")
        self.hide()

    def start(self): self._t.start(450); self.show()
    def stop(self):  self._t.stop();     self.hide()
    def _tick(self):
        frames = ["o o o", "* o o", "o * o", "o o *"]
        self._n = (self._n + 1) % 4
        self.setText(frames[self._n])


# ==============================================================
# LOG PANEL  (right-side slide-out)
# ==============================================================
LEVEL_COLORS = {
    "DEBUG":   "#404060",
    "INFO":    "#5090c0",
    "WARN":    "#c0a020",
    "ERROR":   "#e94560",
    "SUCCESS": "#40c080",
}

class LogPanel(QWidget):
    """Verbose debug/activity log panel. Slides in from the right edge."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(340)
        self.setStyleSheet("background:#06060f;")

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # Header bar
        hdr = QWidget()
        hdr.setStyleSheet("background:#0a0a18;border-bottom:1px solid #12122a;")
        hdr.setFixedHeight(40)
        hr = QHBoxLayout(hdr); hr.setContentsMargins(10, 0, 10, 0)
        lbl = QLabel("Activity Log")
        lbl.setStyleSheet("color:#e94560;font-size:13px;font-weight:bold;font-family:'Segoe UI';")
        hr.addWidget(lbl); hr.addStretch()

        # Level filter checkboxes
        self._filters = {}
        for lvl in ["DEBUG", "INFO", "WARN", "ERROR", "SUCCESS"]:
            cb = QCheckBox(lvl)
            cb.setChecked(lvl != "DEBUG")   # hide DEBUG by default
            cb.setStyleSheet(f"QCheckBox{{color:{LEVEL_COLORS[lvl]};font-size:9px;}}"
                             f"QCheckBox::indicator{{width:10px;height:10px;border-radius:2px;"
                             f"border:1px solid {LEVEL_COLORS[lvl]};background:#0a0a18;}}"
                             f"QCheckBox::indicator:checked{{background:{LEVEL_COLORS[lvl]};}}")
            cb.toggled.connect(self._refilter)
            self._filters[lvl] = cb
            hr.addWidget(cb)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedSize(42, 22)
        clear_btn.setStyleSheet("QPushButton{background:#1a0a0a;color:#804040;border:none;"
                                "border-radius:3px;font-size:10px;}"
                                "QPushButton:hover{background:#e94560;color:white;}")
        clear_btn.clicked.connect(self._clear)
        hr.addWidget(clear_btn)
        v.addWidget(hdr)

        # Log text area
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 9))
        self._log.setStyleSheet("""
            QTextEdit{background:#06060f;color:#404060;border:none;padding:4px;}
            QScrollBar:vertical{background:#06060f;width:4px;}
            QScrollBar::handle:vertical{background:#1a1a30;border-radius:2px;}
        """)
        v.addWidget(self._log, 1)

        # Stats footer
        self._stats = QLabel("Ready")
        self._stats.setStyleSheet("background:#0a0a18;color:#303050;font-size:9px;"
                                  "padding:3px 10px;border-top:1px solid #12122a;")
        self._stats.setFixedHeight(22)
        v.addWidget(self._stats)

        # Internal log storage for refiltering
        self._entries = []   # list of (message, level)
        self._msg_count = 0

        # Connect global logger
        logger.log_message.connect(self._on_log)

    def _on_log(self, msg: str, level: str):
        self._entries.append((msg, level))
        self._msg_count += 1
        if self._filters.get(level, QCheckBox()).isChecked():
            self._append_line(msg, level)
        self._stats.setText(f"{self._msg_count} events  |  {len(self._entries)} stored")

    def _append_line(self, msg: str, level: str):
        color = LEVEL_COLORS.get(level, "#606080")
        cursor = self._log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.setCharFormat(fmt)
        cursor.insertText(msg + "\n")
        self._log.setTextCursor(cursor)
        # Auto-scroll to bottom
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _refilter(self):
        self._log.clear()
        for msg, level in self._entries:
            if self._filters.get(level, QCheckBox()).isChecked():
                self._append_line(msg, level)

    def _clear(self):
        self._entries.clear()
        self._log.clear()
        self._msg_count = 0
        self._stats.setText("Cleared")


# ==============================================================
# SIDEBAR PANEL  (left slide-out)
class SidebarPanel(QWidget):
    sources_changed  = pyqtSignal(list)   # list of (name, url) tuples
    persona_changed  = pyqtSignal(str)    # system prompt text
    temp_changed     = pyqtSignal(float)
    top_p_changed    = pyqtSignal(float)
    max_tok_changed  = pyqtSignal(int)
    obs_ticker_changed   = pyqtSignal(bool)
    obs_scene_on_talk    = pyqtSignal(str)
    obs_scene_on_idle    = pyqtSignal(str)
    save_requested           = pyqtSignal()
    load_requested           = pyqtSignal()
    clear_requested          = pyqtSignal()
    new_session_requested    = pyqtSignal()
    rename_session_requested = pyqtSignal()
    delete_session_requested = pyqtSignal()
    memory_search_requested  = pyqtSignal(str)
    session_switch_requested = pyqtSignal(int)
    wake_word_toggled        = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(320)
        self.setStyleSheet("""
            QWidget{background:#0a0a18;color:#a0a0c0;}
            QGroupBox{color:#404070;border:1px solid #1a1a3a;border-radius:6px;
                margin-top:8px;padding-top:8px;font-size:11px;}
            QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;}
            QCheckBox{color:#8080a0;font-size:11px;spacing:4px;}
            QCheckBox::indicator{width:12px;height:12px;border-radius:3px;
                border:1px solid #1e2a50;background:#16213e;}
            QCheckBox::indicator:checked{background:#e94560;border-color:#e94560;}
            QLabel{color:#707090;font-size:11px;}
            QSlider::groove:horizontal{background:#16213e;height:4px;border-radius:2px;}
            QSlider::handle:horizontal{background:#e94560;width:10px;height:10px;
                margin:-3px 0;border-radius:5px;}
            QComboBox{background:#16213e;color:#c0c0e0;border:1px solid #1e2a50;
                padding:4px;border-radius:4px;font-size:11px;}
            QLineEdit{background:#16213e;color:#c0c0e0;border:1px solid #1e2a50;
                padding:4px;border-radius:4px;font-size:11px;}
            QPushButton{background:#0f3460;color:#a0b0e0;border:none;
                border-radius:4px;padding:5px 10px;font-size:11px;}
            QPushButton:hover{background:#e94560;color:white;}
            QScrollBar:vertical{background:#0a0a18;width:4px;}
            QScrollBar::handle:vertical{background:#1e2a50;border-radius:2px;}
        """)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea{border:none;background:#0a0a18;}")

        content = QWidget()
        content.setStyleSheet("background:#0a0a18;")
        lay = QVBoxLayout(content)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(10)

        # ----- PERSONA -----
        grp_persona = QGroupBox("Persona")
        gp = QVBoxLayout(grp_persona)
        self.persona_combo = QComboBox()
        self.persona_combo.addItems(PERSONAS.keys())
        self.persona_combo.currentTextChanged.connect(self._on_persona)
        gp.addWidget(self.persona_combo)
        self.persona_edit = QTextEdit()
        self.persona_edit.setFixedHeight(70)
        self.persona_edit.setPlainText(PERSONAS["Assistant"])
        self.persona_edit.setStyleSheet(
            "background:#16213e;color:#c0c0e0;border:1px solid #1e2a50;"
            "border-radius:4px;font-size:11px;padding:4px;"
        )
        self.persona_edit.textChanged.connect(lambda: self.persona_changed.emit(
            self.persona_edit.toPlainText()
        ))
        gp.addWidget(self.persona_edit)
        lay.addWidget(grp_persona)

        # ----- MODEL PARAMS -----
        grp_model = QGroupBox("Model Parameters")
        gm = QVBoxLayout(grp_model)

        self.temp_lbl = QLabel("Temperature: 0.70")
        self.temp_s   = QSlider(Qt.Orientation.Horizontal)
        self.temp_s.setRange(1, 200); self.temp_s.setValue(70)
        self.temp_s.valueChanged.connect(lambda v: (
            self.temp_lbl.setText(f"Temperature: {v/100:.2f}"),
            self.temp_changed.emit(v/100)
        ))
        gm.addWidget(self.temp_lbl); gm.addWidget(self.temp_s)

        self.topp_lbl = QLabel("Top-P: 0.90")
        self.topp_s   = QSlider(Qt.Orientation.Horizontal)
        self.topp_s.setRange(1, 100); self.topp_s.setValue(90)
        self.topp_s.valueChanged.connect(lambda v: (
            self.topp_lbl.setText(f"Top-P: {v/100:.2f}"),
            self.top_p_changed.emit(v/100)
        ))
        gm.addWidget(self.topp_lbl); gm.addWidget(self.topp_s)

        self.maxtok_lbl = QLabel("Max tokens: 512")
        self.maxtok_s   = QSlider(Qt.Orientation.Horizontal)
        self.maxtok_s.setRange(64, 2048); self.maxtok_s.setValue(512)
        self.maxtok_s.valueChanged.connect(lambda v: (
            self.maxtok_lbl.setText(f"Max tokens: {v}"),
            self.max_tok_changed.emit(v)
        ))
        gm.addWidget(self.maxtok_lbl); gm.addWidget(self.maxtok_s)
        lay.addWidget(grp_model)

        # ----- MEMORY / SESSIONS -----
        grp_conv = QGroupBox("Memory & Sessions")
        gc = QVBoxLayout(grp_conv)
        gc.setSpacing(5)

        # New session button
        new_sess_btn = QPushButton("+ New Session")
        new_sess_btn.setStyleSheet("QPushButton{background:#0a2a0a;color:#60c060;border:none;"
                                   "border-radius:4px;padding:5px;font-size:11px;}"
                                   "QPushButton:hover{background:#40a040;color:white;}")
        new_sess_btn.clicked.connect(self.new_session_requested)
        gc.addWidget(new_sess_btn)

        # Session list
        gc.addWidget(QLabel("Sessions (click to switch):"))
        self.session_list = QTextEdit()
        self.session_list.setReadOnly(True)
        self.session_list.setFixedHeight(100)
        self.session_list.setStyleSheet(
            "background:#0d0d1e;color:#7080a0;border:1px solid #1e2a50;"
            "border-radius:4px;font-size:10px;padding:4px;"
        )
        gc.addWidget(self.session_list)

        # Session action buttons
        sess_row = QHBoxLayout()
        rename_btn = QPushButton("Rename")
        del_sess_btn = QPushButton("Delete")
        del_sess_btn.setStyleSheet("QPushButton{background:#3a0a0a;color:#e06060;border:none;"
                                   "border-radius:4px;padding:4px 8px;font-size:11px;}"
                                   "QPushButton:hover{background:#e94560;color:white;}")
        rename_btn.clicked.connect(self.rename_session_requested)
        del_sess_btn.clicked.connect(self.delete_session_requested)
        sess_row.addWidget(rename_btn); sess_row.addWidget(del_sess_btn)
        gc.addLayout(sess_row)

        # Memory search
        gc.addWidget(QLabel("Search memory:"))
        self.mem_search = QLineEdit()
        self.mem_search.setPlaceholderText("Search past conversations...")
        self.mem_search.returnPressed.connect(self._do_memory_search)
        gc.addWidget(self.mem_search)

        self.mem_results = QTextEdit()
        self.mem_results.setReadOnly(True)
        self.mem_results.setFixedHeight(90)
        self.mem_results.setStyleSheet(
            "background:#0d0d1e;color:#7080a0;border:1px solid #1e2a50;"
            "border-radius:4px;font-size:10px;padding:4px;"
        )
        gc.addWidget(self.mem_results)

        # Export / clear buttons
        btn_row2 = QHBoxLayout()
        export_btn = QPushButton("Export JSON")
        clear_btn  = QPushButton("Clear Chat")
        clear_btn.setStyleSheet("QPushButton{background:#3a0a0a;color:#e06060;border:none;"
                                "border-radius:4px;padding:4px 8px;font-size:11px;}"
                                "QPushButton:hover{background:#e94560;color:white;}")
        export_btn.clicked.connect(self.save_requested)
        clear_btn.clicked.connect(self.clear_requested)
        btn_row2.addWidget(export_btn); btn_row2.addWidget(clear_btn)
        gc.addLayout(btn_row2)

        lay.addWidget(grp_conv)

        # ----- WAKE WORD -----
        grp_wake = QGroupBox("Wake Word")
        gw = QVBoxLayout(grp_wake)
        self.wake_cb = QCheckBox(f'Enable wake word ("{WAKE_WORD}")')
        self.wake_cb.toggled.connect(self.wake_word_toggled)
        gw.addWidget(self.wake_cb)
        lay.addWidget(grp_wake)

        # ----- OBS -----
        grp_obs = QGroupBox("OBS Settings")
        go = QVBoxLayout(grp_obs)
        self.ticker_cb = QCheckBox("Ticker scroll mode")
        self.ticker_cb.toggled.connect(self.obs_ticker_changed)
        go.addWidget(self.ticker_cb)

        go.addWidget(QLabel("Scene when talking:"))
        self.scene_talk = QLineEdit(); self.scene_talk.setPlaceholderText("Scene name")
        self.scene_talk.textChanged.connect(self.obs_scene_on_talk)
        go.addWidget(self.scene_talk)

        go.addWidget(QLabel("Scene when idle:"))
        self.scene_idle = QLineEdit(); self.scene_idle.setPlaceholderText("Scene name")
        self.scene_idle.textChanged.connect(self.obs_scene_on_idle)
        go.addWidget(self.scene_idle)
        lay.addWidget(grp_obs)

        # ----- NEWS SOURCES -----
        grp_news = QGroupBox("News Sources")
        gn = QVBoxLayout(grp_news)
        self._news_checks = {}
        for category, sources in ALL_NEWS_SOURCES.items():
            cat_lbl = QLabel(category)
            cat_lbl.setStyleSheet("color:#e94560;font-size:10px;font-weight:bold;"
                                  "margin-top:4px;")
            gn.addWidget(cat_lbl)
            for name, url in sources:
                cb = QCheckBox(name)
                cb.setChecked(category in ("General / World",))  # default on
                cb.toggled.connect(self._on_sources_changed)
                self._news_checks[(name, url)] = cb
                gn.addWidget(cb)
        lay.addWidget(grp_news)

        lay.addStretch()
        scroll.setWidget(content)

        # Make scroll fill this panel
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        # Emit initial sources
        QTimer.singleShot(100, self._on_sources_changed)

    def _on_persona(self, name: str):
        self.persona_edit.blockSignals(True)
        self.persona_edit.setPlainText(PERSONAS.get(name, ""))
        self.persona_edit.blockSignals(False)
        self.persona_changed.emit(PERSONAS.get(name, ""))

    def _on_sources_changed(self):
        enabled = [(n, u) for (n, u), cb in self._news_checks.items() if cb.isChecked()]
        NewsScraper.set_sources(enabled)
        self.sources_changed.emit(enabled)

    def refresh_sessions(self):
        """Reload session list from DB and display in the sidebar widget."""
        sessions = memory_db.list_sessions()
        lines = []
        for s in sessions:
            marker = ">" if s["id"] == memory_db.session_id else " "
            lines.append(
                f"{marker} #{s['id']}  {s['name'][:28]}"
                f"  ({s['msg_count']} msgs)  {s['updated_at'][:16]}"
            )
        self.session_list.setPlainText("\n".join(lines) if lines else "No sessions yet.")

    def show_search_results(self, results: list):
        if not results:
            self.mem_results.setPlainText("No matches found.")
            return
        lines = []
        for r in results[:8]:
            snippet = r["content"][:80].replace("\n", " ")
            lines.append(f"[{r['role']}] {r['session']} | {snippet}")
        self.mem_results.setPlainText("\n---\n".join(lines))

    def _do_memory_search(self):
        q = self.mem_search.text().strip()
        if q:
            self.memory_search_requested.emit(q)


# ==============================================================
# MODEL DIALOG
# ==============================================================
MODEL_FAMILIES = {
    "Qwen3 (latest)": [m for m in PRESET_MODELS if "Qwen3" in m],
    "Qwen2.5 Instruct": [m for m in PRESET_MODELS if "Qwen2.5" in m and "Coder" not in m],
    "Qwen2.5 Coder": [m for m in PRESET_MODELS if "Qwen2.5-Coder" in m],
    "Microsoft Phi": [m for m in PRESET_MODELS if "phi" in m.lower() or "Phi" in m],
    "Mistral / Mixtral": [m for m in PRESET_MODELS if "mistral" in m.lower()],
    "Meta Llama": [m for m in PRESET_MODELS if "llama" in m.lower() or "Llama" in m],
    "Google Gemma": [m for m in PRESET_MODELS if "gemma" in m.lower()],
    "DeepSeek": [m for m in PRESET_MODELS if "deepseek" in m.lower()],
    "Small / Fast": [m for m in PRESET_MODELS if any(x in m for x in
                     ["TinyLlama", "stablelm", "SmolLM"])],
    "Yi / Falcon / Other": [m for m in PRESET_MODELS if any(x in m for x in
                             ["01-ai", "falcon", "openchat", "zephyr", "vicuna"])],
    "All Models": PRESET_MODELS,
}


class ModelDialog(QDialog):
    model_loaded = pyqtSignal(str)
    load_started = pyqtSignal(str, object)   # (model_name, thread)

    def __init__(self, parent=None, use_4bit=True):
        super().__init__(parent)
        self.use_4bit = use_4bit
        self.setWindowTitle("Load Model")
        self.setMinimumSize(560, 300)
        self.setStyleSheet("""
            QDialog{background:#1a1a2e;color:#e0e0e0;}
            QLabel{color:#a0a0c0;}
            QComboBox{background:#16213e;color:#e0e0e0;border:1px solid #0f3460;
                padding:6px;border-radius:4px;}
            QComboBox QAbstractItemView{background:#16213e;color:#e0e0e0;
                selection-background-color:#e94560;}
            QLineEdit{background:#16213e;color:#e0e0e0;border:1px solid #0f3460;
                padding:6px;border-radius:4px;}
            QPushButton{background:#0f3460;color:white;border:none;
                padding:8px 20px;border-radius:4px;}
            QPushButton:hover{background:#e94560;}
            QProgressBar{background:#16213e;border-radius:4px;border:none;}
            QProgressBar::chunk{background:#e94560;border-radius:4px;}
            QLabel#note{color:#506080;font-size:10px;}
        """)
        lay = QVBoxLayout(self); lay.setSpacing(10)

        # Family filter row
        fam_row = QHBoxLayout()
        fam_row.addWidget(QLabel("Family:"))
        self.fam_combo = QComboBox()
        self.fam_combo.addItems(MODEL_FAMILIES.keys())
        self.fam_combo.currentTextChanged.connect(self._on_family)
        fam_row.addWidget(self.fam_combo, 1)
        lay.addLayout(fam_row)

        # Model picker
        lay.addWidget(QLabel("Model  (or type any HuggingFace ID):"))
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        lay.addWidget(self.model_combo)

        # VRAM hint label
        self.vram_lbl = QLabel("")
        self.vram_lbl.setObjectName("note")
        lay.addWidget(self.vram_lbl)
        self.model_combo.currentTextChanged.connect(self._update_vram_hint)

        # Current model reminder
        if model_manager.current_name:
            cur_lbl = QLabel(f"Currently loaded: {model_manager.current_name}")
            cur_lbl.setObjectName("note")
            lay.addWidget(cur_lbl)

        self.status = QLabel("Ready.")
        lay.addWidget(self.status)
        self.bar = QProgressBar(); self.bar.setRange(0, 0); self.bar.hide()
        lay.addWidget(self.bar)

        row = QHBoxLayout()
        self.load_btn = QPushButton("Load / Download")
        self.load_btn.clicked.connect(self.start_load)
        cancel = QPushButton("Cancel"); cancel.clicked.connect(self.reject)
        row.addWidget(self.load_btn); row.addWidget(cancel)
        lay.addLayout(row)

        # Seed initial list
        self._on_family("Qwen3 (latest)")
        self.fam_combo.setCurrentText("Qwen3 (latest)")

    def _on_family(self, family: str):
        models = MODEL_FAMILIES.get(family, PRESET_MODELS)
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(models)
        if model_manager.current_name and model_manager.current_name in models:
            self.model_combo.setCurrentText(model_manager.current_name)
        self.model_combo.blockSignals(False)
        self._update_vram_hint(self.model_combo.currentText())

    def _update_vram_hint(self, model_id: str):
        hints = {
            "0.5B": "~1 GB VRAM (runs on CPU too)",
            "0.6B": "~1 GB VRAM (runs on CPU too)",
            "1.1B": "~1.5 GB VRAM",
            "1.5B": "~1.5 GB VRAM",
            "1.7B": "~2 GB VRAM",
            "2b":   "~2 GB VRAM",
            "3B":   "~3 GB VRAM",
            "3b":   "~3 GB VRAM",
            "4B":   "~4 GB VRAM (4-bit: ~2.5 GB)",
            "4b":   "~4 GB VRAM (4-bit: ~2.5 GB)",
            "6.7b": "~6 GB VRAM (4-bit: ~4 GB)",
            "7B":   "~8 GB VRAM (4-bit: ~5 GB)",
            "7b":   "~8 GB VRAM (4-bit: ~5 GB)",
            "8B":   "~10 GB VRAM (4-bit: ~6 GB)",
            "9b":   "~10 GB VRAM (4-bit: ~6 GB)",
            "11b":  "~12 GB VRAM (4-bit: ~7 GB)",
            "12b":  "~14 GB VRAM (4-bit: ~8 GB)",
            "13B":  "~14 GB VRAM (4-bit: ~8 GB)",
            "14B":  "~16 GB VRAM (4-bit: ~10 GB)",
            "27b":  "~28 GB VRAM (4-bit: ~16 GB)",
            "32B":  "~35 GB VRAM (4-bit: ~20 GB)",
            "34B":  "~38 GB VRAM (4-bit: ~22 GB)",
            "70B":  "~80 GB VRAM (4-bit: ~40 GB)",
            "72B":  "~80 GB VRAM (4-bit: ~42 GB)",
        }
        for key, hint in hints.items():
            if key.lower() in model_id.lower():
                self.vram_lbl.setText(f"  VRAM estimate: {hint}")
                return
        self.vram_lbl.setText("")

    def start_load(self):
        name = self.model_combo.currentText().strip()
        if not name: return
        self.load_btn.setEnabled(False); self.bar.show()
        self.status.setText(f"Loading {name}...")
        self.dl = DownloadThread(name, self.use_4bit)
        self.dl.progress.connect(self.status.setText)
        self.dl.finished_signal.connect(self.on_done)
        self.dl.start()
        # Signal main window to show the inline load panel, then close dialog
        self.load_started.emit(name, self.dl)
        self.accept()

    def on_done(self, ok, err):
        self.bar.hide()
        if ok:
            self.model_loaded.emit(model_manager.current_name)
        else:
            self.status.setText(f"Error: {err}")
            self.load_btn.setEnabled(True)


# ==============================================================
# MODEL LOAD PANEL  (inline progress overlay inside main window)
# ==============================================================
class ModelLoadPanel(QFrame):
    """
    Replaces the chat area during model loading.
    Shows stage indicators, a file-level progress bar, and a live log.
    Hidden when loading completes, restored when next load starts.
    """

    STAGES = [
        ("tokenizer",    "Tokenizer"),
        ("quantization", "Quantization"),
        ("weights",      "Model Weights"),
        ("cuda",         "CUDA / Device Map"),
        ("ready",        "Ready"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            ModelLoadPanel {
                background: #0a0a16;
                border: 1px solid #1a1a30;
                border-radius: 8px;
            }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(30, 24, 30, 24)
        root.setSpacing(14)

        # Title
        self._title = QLabel("Loading Model")
        self._title.setStyleSheet(
            "color:#e94560;font-size:16px;font-weight:bold;"
            "font-family:'Segoe UI';background:transparent;"
        )
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._title)

        # Model name
        self._model_lbl = QLabel("")
        self._model_lbl.setStyleSheet(
            "color:#5060a0;font-size:11px;font-family:'Consolas';"
            "background:transparent;"
        )
        self._model_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._model_lbl)

        # Stage row
        stage_frame = QWidget()
        stage_frame.setStyleSheet("background:transparent;")
        stage_row = QHBoxLayout(stage_frame)
        stage_row.setContentsMargins(0, 4, 0, 4)
        stage_row.setSpacing(0)
        self._stage_labels = {}
        for i, (key, label) in enumerate(self.STAGES):
            col = QVBoxLayout()
            col.setSpacing(4)
            col.setAlignment(Qt.AlignmentFlag.AlignHCenter)

            dot = QLabel("o")
            dot.setFixedSize(28, 28)
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dot.setStyleSheet(
                "background:#12122a;color:#2a2a50;border-radius:14px;"
                "font-size:13px;font-weight:bold;"
            )

            lbl = QLabel(label)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(
                "color:#2a2a50;font-size:9px;font-family:'Segoe UI';"
                "background:transparent;"
            )
            lbl.setWordWrap(True)
            lbl.setFixedWidth(72)

            col.addWidget(dot, alignment=Qt.AlignmentFlag.AlignHCenter)
            col.addWidget(lbl, alignment=Qt.AlignmentFlag.AlignHCenter)
            self._stage_labels[key] = (dot, lbl)

            stage_row.addLayout(col)
            # Connector line between dots (not after last)
            if i < len(self.STAGES) - 1:
                line = QLabel()
                line.setFixedSize(40, 2)
                line.setStyleSheet("background:#1a1a38;border-radius:1px;")
                stage_row.addWidget(line, alignment=Qt.AlignmentFlag.AlignVCenter)
                self._stage_labels[f"_line_{i}"] = line

        root.addWidget(stage_frame)

        # File progress bar
        file_row = QHBoxLayout()
        self._file_lbl = QLabel("Waiting...")
        self._file_lbl.setStyleSheet(
            "color:#404070;font-size:10px;font-family:'Consolas';"
            "background:transparent;"
        )
        file_row.addWidget(self._file_lbl, 1)
        self._pct_lbl = QLabel("")
        self._pct_lbl.setStyleSheet(
            "color:#606090;font-size:10px;background:transparent;"
        )
        self._pct_lbl.setFixedWidth(50)
        self._pct_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        file_row.addWidget(self._pct_lbl)
        root.addLayout(file_row)

        self._file_bar = QProgressBar()
        self._file_bar.setRange(0, 100)
        self._file_bar.setValue(0)
        self._file_bar.setTextVisible(False)
        self._file_bar.setFixedHeight(6)
        self._file_bar.setStyleSheet("""
            QProgressBar {
                background: #12122a;
                border: none;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #e94560, stop:1 #5050c0);
                border-radius: 3px;
            }
        """)
        root.addWidget(self._file_bar)

        # Overall pulse bar (indeterminate, shown when no per-file data)
        self._pulse_bar = QProgressBar()
        self._pulse_bar.setRange(0, 0)   # indeterminate
        self._pulse_bar.setTextVisible(False)
        self._pulse_bar.setFixedHeight(4)
        self._pulse_bar.setStyleSheet("""
            QProgressBar {background:#12122a;border:none;border-radius:2px;}
            QProgressBar::chunk {background:#2a2a70;border-radius:2px;}
        """)
        root.addWidget(self._pulse_bar)

        # Live log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Consolas", 9))
        self._log.setFixedHeight(140)
        self._log.setStyleSheet("""
            QTextEdit {
                background: #06060f;
                color: #404068;
                border: 1px solid #12122a;
                border-radius: 4px;
                padding: 4px;
            }
            QScrollBar:vertical {background:#06060f;width:4px;}
            QScrollBar::handle:vertical {background:#1a1a30;border-radius:2px;}
        """)
        root.addWidget(self._log)

        # Cancel button
        self._cancel_btn = QPushButton("Cancel Load")
        self._cancel_btn.setFixedHeight(32)
        self._cancel_btn.setStyleSheet("""
            QPushButton {
                background: #1a0a0a; color: #804040;
                border: 1px solid #3a1a1a; border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {background:#e94560;color:white;border-color:#e94560;}
        """)
        root.addWidget(self._cancel_btn)

        self._current_stage = None
        self._thread_ref    = None

    # ---- public API -------------------------------------

    def start(self, model_name: str, thread: "DownloadThread"):
        self._thread_ref = thread
        self._log.clear()
        self._title.setText("Loading Model")
        self._model_lbl.setText(model_name)
        self._file_lbl.setText("Initializing...")
        self._file_bar.setValue(0)
        self._pct_lbl.setText("")
        self._reset_stages()
        self._pulse_bar.setRange(0, 0)

        thread.progress.connect(self._on_progress)
        thread.stage.connect(self._on_stage)
        thread.file_progress.connect(self._on_file_progress)
        self._cancel_btn.clicked.connect(self._cancel)
        self.show()

    def finish(self, ok: bool, err: str = ""):
        self._pulse_bar.setRange(0, 1)
        self._pulse_bar.setValue(1)
        if ok:
            self._title.setText("Model Ready")
            self._title.setStyleSheet(
                "color:#40c080;font-size:16px;font-weight:bold;"
                "font-family:'Segoe UI';background:transparent;"
            )
            self._on_stage("ready")
            self._file_lbl.setText("All files loaded successfully")
            self._file_bar.setValue(100)
            QTimer.singleShot(1800, self.hide)
        else:
            self._title.setText("Load Failed")
            self._title.setStyleSheet(
                "color:#e94560;font-size:16px;font-weight:bold;"
                "font-family:'Segoe UI';background:transparent;"
            )
            self._append_log(f"ERROR: {err}", "#e94560")

    # ---- slots ------------------------------------------

    def _on_progress(self, msg: str):
        self._append_log(msg, "#5070a0")
        self._file_lbl.setText(msg[:80])

    def _on_stage(self, stage: str):
        if stage == self._current_stage:
            return
        self._current_stage = stage
        for i, (key, _) in enumerate(self.STAGES):
            dot, lbl = self._stage_labels[key]
            if key == stage:
                dot.setStyleSheet(
                    "background:#e94560;color:white;border-radius:14px;"
                    "font-size:13px;font-weight:bold;"
                )
                lbl.setStyleSheet(
                    "color:#e94560;font-size:9px;font-family:'Segoe UI';"
                    "background:transparent;font-weight:bold;"
                )
                dot.setText("*")
                # Light up connector lines leading to this stage
                for j in range(i):
                    line_key = f"_line_{j}"
                    if line_key in self._stage_labels:
                        self._stage_labels[line_key].setStyleSheet(
                            "background:#e94560;border-radius:1px;"
                        )
                # Mark previous stages done
                for j in range(i):
                    pk, _ = self.STAGES[j]
                    pd, pl = self._stage_labels[pk]
                    pd.setStyleSheet(
                        "background:#206040;color:#80ffa0;border-radius:14px;"
                        "font-size:13px;font-weight:bold;"
                    )
                    pd.setText("v")
                    pl.setStyleSheet(
                        "color:#406050;font-size:9px;font-family:'Segoe UI';"
                        "background:transparent;"
                    )
            elif self.STAGES.index((key, _)) > \
                    next(k for k, (s, _) in enumerate(self.STAGES) if s == stage):
                # Future stages
                dot.setStyleSheet(
                    "background:#12122a;color:#2a2a50;border-radius:14px;"
                    "font-size:13px;font-weight:bold;"
                )
                dot.setText("o")
                lbl.setStyleSheet(
                    "color:#2a2a50;font-size:9px;font-family:'Segoe UI';"
                    "background:transparent;"
                )

    def _on_file_progress(self, fname: str, downloaded: int, total: int):
        self._pulse_bar.setRange(0, 0)  # keep indeterminate during file dl
        if total > 0:
            pct = min(int(downloaded / total * 100), 100)
            self._file_bar.setValue(pct)
            self._pct_lbl.setText(f"{pct}%")
            mb_dl  = downloaded / 1e6
            mb_tot = total / 1e6
            self._file_lbl.setText(f"{fname}  {mb_dl:.1f} / {mb_tot:.1f} MB")
        else:
            self._file_lbl.setText(f"{fname}")

    def _cancel(self):
        if self._thread_ref and self._thread_ref.isRunning():
            self._thread_ref.terminate()
            self._append_log("Load cancelled by user.", "#c06040")
            self._title.setText("Cancelled")
            self._pulse_bar.setRange(0, 1)

    def _reset_stages(self):
        self._current_stage = None
        for key, _ in self.STAGES:
            dot, lbl = self._stage_labels[key]
            dot.setStyleSheet(
                "background:#12122a;color:#2a2a50;border-radius:14px;"
                "font-size:13px;font-weight:bold;"
            )
            dot.setText("o")
            lbl.setStyleSheet(
                "color:#2a2a50;font-size:9px;font-family:'Segoe UI';"
                "background:transparent;"
            )
        for i in range(len(self.STAGES) - 1):
            line_key = f"_line_{i}"
            if line_key in self._stage_labels:
                self._stage_labels[line_key].setStyleSheet(
                    "background:#1a1a38;border-radius:1px;"
                )

    def _append_log(self, msg: str, color: str = "#404068"):
        cursor = self._log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.setCharFormat(fmt)
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        cursor.insertText(f"[{ts}] {msg}\n")
        self._log.setTextCursor(cursor)
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum()
        )
STYLE = """
QMainWindow,QWidget#root{background:#0d0d1a;}
QScrollArea{background:transparent;border:none;}
QScrollBar:vertical{background:#0d0d1a;width:5px;border-radius:3px;}
QScrollBar::handle:vertical{background:#1e2a50;border-radius:3px;}
QLineEdit{background:#16213e;color:#dde8ff;border:1px solid #1e2a50;
  border-radius:22px;padding:10px 18px;font-size:13px;}
QLineEdit:focus{border-color:#e94560;}
QPushButton#send{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
  stop:0 #e94560,stop:1 #b01030);color:white;border:none;
  border-radius:22px;padding:10px 22px;font-size:13px;font-weight:bold;}
QPushButton#send:hover{background:#ff5577;}
QPushButton#send:disabled{background:#252540;color:#444;}
QPushButton#mic{background:#16213e;color:#7080c0;border:1px solid #1e2a50;
  border-radius:22px;padding:10px 14px;font-size:15px;}
QPushButton#mic:hover{background:#0f3460;color:white;}
QPushButton#mic:checked{background:#e94560;color:white;border-color:#e94560;}
QPushButton#mdl{background:#0f3460;color:#9090c0;border:1px solid #1e2a50;
  border-radius:4px;padding:4px 10px;font-size:10px;}
QPushButton#mdl:hover{background:#e94560;color:white;}
QPushButton#sidebar_toggle{background:#0f0f1e;color:#5060a0;border:none;
  border-right:1px solid #14143a;font-size:14px;padding:0;}
QPushButton#sidebar_toggle:hover{color:#e94560;}
QCheckBox{color:#707090;font-size:11px;spacing:4px;}
QCheckBox::indicator{width:13px;height:13px;border-radius:3px;
  border:1px solid #1e2a50;background:#16213e;}
QCheckBox::indicator:checked{background:#e94560;border-color:#e94560;}
QSlider::groove:horizontal{background:#16213e;height:4px;border-radius:2px;}
QSlider::handle:horizontal{background:#e94560;width:12px;height:12px;
  margin:-4px 0;border-radius:6px;}
QLabel#status{color:#404060;font-size:10px;padding:2px 12px;background:#0a0a14;}
QLabel#token_ctr{color:#303060;font-size:10px;padding:2px 12px;background:#0a0a14;}
"""


# ==============================================================
# MAIN WINDOW
# ==============================================================
class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Chat v3")
        self.resize(1060, 920)
        self.setStyleSheet(STYLE)

        self._system_prompt   = PERSONAS["Assistant"]
        self._history_msgs    = []   # excludes system prompt
        self.current_response = ""
        self._obs_chars       = 0
        self._asst_bubble     = None
        self._generating      = False
        self._wake_thread     = None
        self._selected_voice  = None

        root = QWidget(); root.setObjectName("root")
        self.setCentralWidget(root)
        main_row = QHBoxLayout(root)
        main_row.setContentsMargins(0, 0, 0, 0)
        main_row.setSpacing(0)

        # -- Sidebar toggle strip --------------------------
        self.toggle_btn = QPushButton("<<")
        self.toggle_btn.setObjectName("sidebar_toggle")
        self.toggle_btn.setFixedWidth(22)
        self.toggle_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.toggle_btn.clicked.connect(self.toggle_sidebar)
        main_row.addWidget(self.toggle_btn)

        # -- Sidebar ---------------------------------------
        self.sidebar = SidebarPanel()
        self.sidebar.persona_changed.connect(self._set_persona)
        self.sidebar.temp_changed.connect(lambda v: setattr(model_manager, 'temperature', v))
        self.sidebar.top_p_changed.connect(lambda v: setattr(model_manager, 'top_p', v))
        self.sidebar.max_tok_changed.connect(lambda v: setattr(model_manager, 'max_tokens', v))
        self.sidebar.save_requested.connect(self.save_chat)
        self.sidebar.load_requested.connect(self.load_chat)
        self.sidebar.clear_requested.connect(self.clear_chat)
        self.sidebar.new_session_requested.connect(self.new_session)
        self.sidebar.rename_session_requested.connect(self.rename_session)
        self.sidebar.delete_session_requested.connect(self.delete_session)
        self.sidebar.memory_search_requested.connect(self.memory_search)
        self.sidebar.obs_ticker_changed.connect(lambda v: setattr(obs_worker, 'ticker_mode', v))
        self.sidebar.obs_scene_on_talk.connect(lambda v: setattr(obs_worker, 'scene_on_talk', v))
        self.sidebar.obs_scene_on_idle.connect(lambda v: setattr(obs_worker, 'scene_on_idle', v))
        self.sidebar.wake_word_toggled.connect(self.toggle_wake_word)
        main_row.addWidget(self.sidebar)

        # -- Main chat column ------------------------------
        chat_col = QWidget()
        chat_col.setStyleSheet("background:#0d0d1a;")
        vbox = QVBoxLayout(chat_col)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # Header
        hdr = QWidget()
        hdr.setStyleSheet("background:#090914;border-bottom:1px solid #14143a;")
        hdr.setFixedHeight(54)
        hr = QHBoxLayout(hdr); hr.setContentsMargins(18, 0, 18, 0)
        title = QLabel("AI Chat")
        title.setStyleSheet("color:#e94560;font-size:17px;font-weight:bold;font-family:'Segoe UI';")
        hr.addWidget(title); hr.addStretch()
        self.mdl_lbl = QLabel(f"  {DEFAULT_MODEL.split('/')[-1]}")
        self.mdl_lbl.setStyleSheet("color:#404070;font-size:10px;")
        hr.addWidget(self.mdl_lbl)
        mdl_btn = QPushButton("Load Model"); mdl_btn.setObjectName("mdl")
        mdl_btn.clicked.connect(self.open_model_dialog)
        hr.addWidget(mdl_btn)

        log_btn = QPushButton("Log")
        log_btn.setObjectName("mdl")
        log_btn.setToolTip("Toggle activity log panel")
        log_btn.clicked.connect(self.toggle_log_panel)
        hr.addWidget(log_btn)
        vbox.addWidget(hdr)

        # Scroll area + Load panel share the same stretch slot via a stacked approach
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_w = QWidget(); self.chat_w.setStyleSheet("background:#0d0d1a;")
        self.cl = QVBoxLayout(self.chat_w)
        self.cl.setContentsMargins(16, 16, 16, 8); self.cl.setSpacing(4)
        self.cl.addStretch()
        self.scroll.setWidget(self.chat_w)

        # Model load panel (shown instead of chat area while loading)
        self.load_panel = ModelLoadPanel()
        self.load_panel.hide()

        # Use a QStackedWidget so they occupy the same space
        from PyQt6.QtWidgets import QStackedWidget as _SW
        self._chat_stack = _SW()
        self._chat_stack.addWidget(self.scroll)     # index 0 = chat
        self._chat_stack.addWidget(self.load_panel) # index 1 = loading
        self._chat_stack.setCurrentIndex(0)
        vbox.addWidget(self._chat_stack, 1)

        # Typing dots row
        tr = QHBoxLayout(); tr.setContentsMargins(24, 0, 0, 4)
        self.dots = TypingDots(); tr.addWidget(self.dots); tr.addStretch()
        vbox.addLayout(tr)

        # Controls bar
        ctrl = QWidget()
        ctrl.setStyleSheet("background:#090914;border-top:1px solid #14143a;")
        cv = QVBoxLayout(ctrl); cv.setContentsMargins(12, 8, 12, 8); cv.setSpacing(6)

        # Options row
        opts = QHBoxLayout(); opts.setSpacing(8)
        self.tts_cb = QCheckBox("TTS")
        self.obs_cb = QCheckBox("OBS")
        self.gpu_cb = QCheckBox("4-bit GPU")
        self.gpu_cb.setChecked(torch.cuda.is_available())
        self.vol_s = QSlider(Qt.Orientation.Horizontal)
        self.vol_s.setRange(0, 100); self.vol_s.setValue(80); self.vol_s.setFixedWidth(70)
        self.spd_s = QSlider(Qt.Orientation.Horizontal)
        self.spd_s.setRange(100, 300); self.spd_s.setValue(200); self.spd_s.setFixedWidth(70)
        for w in [self.tts_cb, QLabel("Vol"), self.vol_s, QLabel("Spd"), self.spd_s]:
            opts.addWidget(w)
        opts.addStretch()
        opts.addWidget(self.obs_cb)
        opts.addWidget(self.gpu_cb)
        cv.addLayout(opts)

        # Input row
        inp = QHBoxLayout(); inp.setSpacing(8)
        self.mic_btn = QPushButton("mic"); self.mic_btn.setObjectName("mic")
        self.mic_btn.setCheckable(True); self.mic_btn.setFixedSize(46, 46)
        self.mic_btn.clicked.connect(self.toggle_mic)
        self.input_f = QLineEdit()
        self.input_f.setPlaceholderText("Ask anything... try 'top news' for live headlines")
        self.input_f.returnPressed.connect(self.send_message)
        self.send_btn = QPushButton("Send"); self.send_btn.setObjectName("send")
        self.send_btn.setFixedHeight(46); self.send_btn.clicked.connect(self.send_message)
        inp.addWidget(self.mic_btn); inp.addWidget(self.input_f, 1); inp.addWidget(self.send_btn)
        cv.addLayout(inp)
        vbox.addWidget(ctrl)

        # Status + token counter row
        status_row = QHBoxLayout(); status_row.setContentsMargins(0, 0, 0, 0)
        self.status = QLabel("Loading model..."); self.status.setObjectName("status")
        self.status.setFixedHeight(20)
        self.token_ctr = QLabel("Tokens: 0"); self.token_ctr.setObjectName("token_ctr")
        self.token_ctr.setFixedHeight(20)
        status_row.addWidget(self.status, 1)
        status_row.addWidget(self.token_ctr)
        vbox.addLayout(status_row)

        main_row.addWidget(chat_col, 1)

        # -- Log panel (right side) ------------------------
        self.log_panel = LogPanel()
        main_row.addWidget(self.log_panel)

        self.log_toggle_btn = QPushButton(">>")
        self.log_toggle_btn.setObjectName("sidebar_toggle")
        self.log_toggle_btn.setFixedWidth(22)
        self.log_toggle_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.log_toggle_btn.clicked.connect(self.toggle_log_panel)
        main_row.addWidget(self.log_toggle_btn)

        # Boot model
        self.send_btn.setEnabled(False)
        logger.info("=== AI Chat v3 starting ===")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.ok(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logger.info(f"Default model: {DEFAULT_MODEL}")

        # Resume last session or create a fresh one
        sid = memory_db.resume_last_session()
        if sid:
            sess = memory_db.get_session(sid)
            if sess and sess["persona"]:
                self._system_prompt = sess["persona"]
            self._history_msgs = memory_db.load_messages()
            logger.info(f"Restored {len(self._history_msgs)} messages from session #{sid}")
        else:
            memory_db.new_session(persona=self._system_prompt)
            logger.info("Started fresh session")

        # Render restored history as bubbles
        QTimer.singleShot(200, self._render_history)
        self.sidebar.refresh_sessions()
        self._boot = DownloadThread(DEFAULT_MODEL, torch.cuda.is_available())
        self._boot.progress.connect(self.status.setText)
        self._boot.finished_signal.connect(self.on_model_ready)
        self._chat_stack.setCurrentIndex(1)                          # show load panel
        self.load_panel.start(DEFAULT_MODEL, self._boot)
        self._boot.start()

        # Token counter update timer
        self._tok_timer = QTimer(); self._tok_timer.timeout.connect(self._update_token_count)
        self._tok_timer.start(3000)

    # -- sidebar ------------------------------------------
    def toggle_sidebar(self):
        visible = self.sidebar.isVisible()
        self.sidebar.setVisible(not visible)
        self.toggle_btn.setText("<<" if not visible else ">>")

    def toggle_log_panel(self):
        visible = self.log_panel.isVisible()
        self.log_panel.setVisible(not visible)
        self.log_toggle_btn.setText(">>" if not visible else "<<")

    def _set_persona(self, text: str):
        self._system_prompt = text
        memory_db.update_persona(text)

    # -- utils --------------------------------------------
    def set_status(self, msg): self.status.setText(msg)

    def _full_messages(self):
        return [{"role": "system", "content": self._system_prompt}] + self._history_msgs

    def _update_token_count(self):
        if model_manager.tokenizer:
            n = model_manager.count_tokens(self._full_messages())
            self.token_ctr.setText(f"Tokens: {n:,}")

    def scroll_bottom(self):
        QTimer.singleShot(40, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()))

    def add_bubble(self, text, role) -> ChatBubble:
        b = ChatBubble(text, role)
        self.cl.insertWidget(self.cl.count() - 1, b)
        b.show_animated()
        self.scroll_bottom()
        return b

    # -- model --------------------------------------------
    def on_model_ready(self, ok, err):
        self.load_panel.finish(ok, err)
        if ok:
            short = model_manager.current_name.split("/")[-1]
            self.mdl_lbl.setText(f"  {short}")
            mode = "GPU 4-bit" if self.gpu_cb.isChecked() and torch.cuda.is_available() else "CPU"
            self.set_status(f"Ready  {short}  {mode}")
            self.send_btn.setEnabled(True)
            self.input_f.setFocus()
            QTimer.singleShot(1900, lambda: self._chat_stack.setCurrentIndex(0))
        else:
            self.set_status(f"Error: {err}")
            # Keep panel visible so user can see the error, add retry button feel

    def open_model_dialog(self):
        d = ModelDialog(self, self.gpu_cb.isChecked())

        def _on_load_start(name, thread):
            # Switch to load panel when ModelDialog kicks off a download
            self._chat_stack.setCurrentIndex(1)
            self.load_panel.start(name, thread)
            thread.finished_signal.connect(self.on_model_ready)

        d.load_started.connect(_on_load_start)
        d.model_loaded.connect(lambda n: (
            self.mdl_lbl.setText(f"  {n.split('/')[-1]}"),
            self.set_status(f"Ready  {n.split('/')[-1]}"),
            self.send_btn.setEnabled(True)
        ))
        d.exec()

    # -- memory helpers -----------------------------------
    def _render_history(self):
        """Render all in-memory history messages as chat bubbles."""
        for msg in self._history_msgs:
            if msg["role"] in ("user", "assistant"):
                self.add_bubble(msg["content"], msg["role"])
        self.scroll_bottom()
        self._update_token_count()

    # -- conversations ------------------------------------
    def new_session(self):
        """Start a fresh session ? old one stays in DB forever."""
        memory_db.new_session(persona=self._system_prompt)
        self._history_msgs = []
        self._clear_bubbles()
        self.sidebar.refresh_sessions()
        self.set_status(f"New session #{memory_db.session_id} started")
        logger.ok(f"New session #{memory_db.session_id}")

    def rename_session(self):
        from PyQt6.QtWidgets import QInputDialog
        sid = memory_db.session_id
        if not sid:
            return
        sess = memory_db.get_session(sid)
        name, ok = QInputDialog.getText(
            self, "Rename Session", "Session name:", text=sess["name"] if sess else ""
        )
        if ok and name.strip():
            memory_db.rename_session(sid, name.strip())
            self.sidebar.refresh_sessions()
            self.set_status(f"Renamed to '{name.strip()}'")

    def delete_session(self):
        from PyQt6.QtWidgets import QMessageBox
        sid = memory_db.session_id
        if not sid:
            return
        reply = QMessageBox.question(
            self, "Delete Session",
            f"Permanently delete session #{sid} and all its messages?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            memory_db.delete_session(sid)
            self._history_msgs = []
            self._clear_bubbles()
            # Auto-resume next session or create new
            nxt = memory_db.resume_last_session()
            if nxt:
                self._history_msgs = memory_db.load_messages()
                self._render_history()
            else:
                memory_db.new_session(persona=self._system_prompt)
            self.sidebar.refresh_sessions()
            self.set_status("Session deleted")

    def memory_search(self, query: str):
        results = memory_db.search(query)
        self.sidebar.show_search_results(results)
        logger.info(f"Memory search '{query}': {len(results)} results")

    def save_chat(self):
        """Export current session to JSON (DB record stays regardless)."""
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Chat", os.path.join(SAVES_DIR, f"chat_{ts}.json"),
            "JSON Files (*.json)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "session_id": memory_db.session_id,
                    "persona":    self._system_prompt,
                    "history":    self._history_msgs
                }, f, indent=2, ensure_ascii=False)
            self.set_status(f"Exported: {os.path.basename(path)}")
            logger.ok(f"Chat exported to {path}")

    def load_chat(self):
        """Import a JSON export and create a new session from it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Chat", SAVES_DIR, "JSON Files (*.json)"
        )
        if path:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._system_prompt = data.get("persona", self._system_prompt)
            msgs = data.get("history", [])
            # Create a new DB session for the imported data
            fname = os.path.splitext(os.path.basename(path))[0]
            memory_db.new_session(name=f"Import: {fname}", persona=self._system_prompt)
            for m in msgs:
                memory_db.append(m["role"], m["content"])
            self._history_msgs = memory_db.load_messages()
            self._clear_bubbles()
            self._render_history()
            self.sidebar.refresh_sessions()
            self.set_status(f"Imported: {os.path.basename(path)}")
            logger.ok(f"Imported {len(msgs)} messages from {path}")

    def clear_chat(self, keep_history=False):
        if not keep_history:
            memory_db.clear_messages()
            self._history_msgs = []
        self._clear_bubbles()
        self._update_token_count()
        self.sidebar.refresh_sessions()

    def _clear_bubbles(self):
        while self.cl.count() > 1:
            item = self.cl.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

    # -- wake word ----------------------------------------
    def toggle_wake_word(self, enabled: bool):
        if enabled:
            self._wake_thread = MicThread(wake_word_mode=True)
            self._wake_thread.result.connect(self._on_wake_command)
            self._wake_thread.status.connect(self.set_status)
            self._wake_thread.error.connect(lambda e: self.set_status(f"Wake: {e}"))
            self._wake_thread.start()
            self.set_status(f'Wake word active: say "{WAKE_WORD}"')
        else:
            if self._wake_thread:
                self._wake_thread.stop()
                self._wake_thread = None
            self.set_status("Wake word disabled.")

    def _on_wake_command(self, text: str):
        self.input_f.setText(text)
        self.send_message()

    # -- mic ----------------------------------------------
    def toggle_mic(self):
        if self.mic_btn.isChecked():
            self.mic_t = MicThread(wake_word_mode=False)
            self.mic_t.result.connect(self.on_mic_result)
            self.mic_t.error.connect(lambda e: (self.set_status(f"Mic: {e}"), self.mic_btn.setChecked(False)))
            self.mic_t.status.connect(self.set_status)
            self.mic_t.finished.connect(lambda: self.mic_btn.setChecked(False))
            self.mic_t.start()

    def on_mic_result(self, text):
        self.input_f.setText(text)
        self.send_message()

    # -- send ---------------------------------------------
    def send_message(self):
        if self._generating or not model_manager.model:
            return
        user_text = self.input_f.text().strip()
        if not user_text:
            return

        self.input_f.clear()
        self.send_btn.setEnabled(False)
        self.mic_btn.setEnabled(False)
        self._generating = True

        self.add_bubble(user_text, "user")

        if self.obs_cb.isChecked() and OBS_AVAILABLE:
            obs_worker.update("UserText", format_for_obs(f"You: {user_text}"))
            if obs_worker.scene_on_talk:
                obs_worker.switch_scene(obs_worker.scene_on_talk)

        # Build messages - detect intent and scrape live context if needed
        messages = self._full_messages()
        is_web     = WebScraper.is_web_search_request(user_text)
        is_weather = WebScraper.is_weather_request(user_text)
        is_news    = NewsScraper.is_news_request(user_text)

        if is_web:
            query = WebScraper.extract_search_query(user_text)
            logger.info(f"Web search triggered, query: '{query}'")
            self.set_status(f"Searching: {query[:50]}...")
            self._web_t = WebSearchThread(query)
            self._web_t.finished_signal.connect(
                lambda ctx: self._start_generation(user_text, ctx)
            )
            self._web_t.start()
        elif is_weather:
            # Use the user's exact message as the search query - no API, just scrape
            query = user_text.strip()
            logger.info(f"Weather search triggered, query: '{query}'")
            self.set_status("Searching for weather...")
            self._weather_t = WeatherFetchThread(query)
            self._weather_t.finished_signal.connect(
                lambda ctx: self._start_generation(user_text, ctx)
            )
            self._weather_t.start()
        elif is_news:
            logger.info("News request detected, fetching headlines...")
            self.set_status("Fetching headlines...")
            self._news_t = NewsFetchThread()
            self._news_t.finished_signal.connect(
                lambda ctx: self._start_generation(user_text, ctx)
            )
            self._news_t.start()
        else:
            self._start_generation(user_text, None)

    def _start_generation(self, user_text: str, news_ctx):
        messages = self._full_messages()
        if news_ctx:
            messages.append({"role": "system", "content": news_ctx})
            logger.info(f"Injected context ({len(news_ctx)} chars) into prompt")
        messages.append({"role": "user", "content": user_text})
        self._history_msgs.append({"role": "user", "content": user_text})
        memory_db.append("user", user_text)                         # <-- persist to SQLite
        logger.debug(f"Prompt has {len(messages)} messages")

        self.current_response = ""
        self._obs_chars       = 0
        self._asst_bubble     = None

        self.dots.start()
        self.set_status("Generating...")

        self.gen_t = GenerationThread(messages)
        self.gen_t.new_token.connect(self.on_token)
        self.gen_t.finished_signal.connect(self.on_done)
        self.gen_t.error_signal.connect(lambda e: self.set_status(f"Error: {e}"))
        self.gen_t.start()

    def on_token(self, token: str):
        self.current_response += token
        self._obs_chars += len(token)
        if self._asst_bubble is None:
            self.dots.stop()
            self._asst_bubble = self.add_bubble(token, "assistant")
        else:
            self._asst_bubble.update_text(self.current_response)
            self.scroll_bottom()
        if self.obs_cb.isChecked() and OBS_AVAILABLE and self._obs_chars >= 50:
            obs_worker.update("ChatText", format_for_obs(self.current_response))
            self._obs_chars = 0

    def on_done(self):
        self.dots.stop()
        clean = self.current_response.strip()

        # Re-render bubble with full text (picks up code blocks properly)
        if self._asst_bubble:
            idx = self.cl.indexOf(self._asst_bubble)
            if idx >= 0:
                self._asst_bubble.deleteLater()
                new_b = ChatBubble(clean, "assistant")
                self.cl.insertWidget(idx, new_b)
                new_b.show_animated()

        self._history_msgs.append({"role": "assistant", "content": clean})
        memory_db.append("assistant", clean)                        # <-- persist to SQLite
        self.sidebar.refresh_sessions()                             # update msg count display

        if self.obs_cb.isChecked() and OBS_AVAILABLE:
            obs_worker.paginate("ChatText", clean)
            if obs_worker.scene_on_idle:
                obs_worker.switch_scene(obs_worker.scene_on_idle)

        if self.tts_cb.isChecked():
            tts_worker.speak(clean, self.vol_s.value()/100, self.spd_s.value(), self._selected_voice)

        mode  = "GPU 4-bit" if self.gpu_cb.isChecked() and torch.cuda.is_available() else "CPU"
        short = model_manager.current_name.split("/")[-1] if model_manager.current_name else "?"
        self.set_status(f"Done  {mode}  {short}")
        self._update_token_count()
        self._generating = False
        self.send_btn.setEnabled(True)
        self.mic_btn.setEnabled(True)
        self.input_f.setFocus()


# ==============================================================
# ENTRY POINT
# ==============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,          QColor("#0d0d1a"))
    pal.setColor(QPalette.ColorRole.WindowText,      QColor("#e0e0ff"))
    pal.setColor(QPalette.ColorRole.Base,            QColor("#16213e"))
    pal.setColor(QPalette.ColorRole.Text,            QColor("#e0e0ff"))
    pal.setColor(QPalette.ColorRole.Button,          QColor("#16213e"))
    pal.setColor(QPalette.ColorRole.ButtonText,      QColor("#e0e0ff"))
    pal.setColor(QPalette.ColorRole.Highlight,       QColor("#e94560"))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(pal)

    win = ChatWindow()
    win.show()
    sys.exit(app.exec())
