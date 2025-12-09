import os
from pathlib import Path
import torch

os.environ["TRANSFORMERS_NO_TF"] = "1"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

BASE = Path("models")
LLM_DIR = BASE / "llms"
TRANS_DIR = BASE / "translators"
CACHE_FILE = BASE / "cache.json"

RAG_DIR = BASE / "rag"
RAG_INDEX_FILE = RAG_DIR / "rag.index"
RAG_META_FILE = RAG_DIR / "metadata.json"

# IndicTrans2 language codes (source_script, target_script)
# Using single bidirectional model: ai4bharat/indictrans2-en-indic-dist-200M
LANG_MAP = {
    # Indo-Aryan (Devanagari + others)
    "hi": ("hin_Deva", "eng_Latn"),      # Hindi
    "bn": ("ben_Beng", "eng_Latn"),      # Bengali
    "mr": ("mar_Deva", "eng_Latn"),      # Marathi
    "gu": ("guj_Gujr", "eng_Latn"),      # Gujarati
    "pa": ("pan_Guru", "eng_Latn"),      # Punjabi (Gurmukhi)
    "ur": ("urd_Arab", "eng_Latn"),      # Urdu
    "as": ("asm_Beng", "eng_Latn"),      # Assamese
    "bho": ("bho_Deva", "eng_Latn"),     # Bhojpuri
    "mag": ("mag_Deva", "eng_Latn"),     # Magahi
    "mai": ("mai_Deva", "eng_Latn"),     # Maithili
    "hne": ("hne_Deva", "eng_Latn"),     # Chhattisgarhi
    "or": ("ory_Orya", "eng_Latn"),      # Odia
    "ks_ar": ("kas_Arab", "eng_Latn"),   # Kashmiri (Arabic)
    "ks_de": ("kas_Deva", "eng_Latn"),   # Kashmiri (Devanagari)
    "sd": ("snd_Arab", "eng_Latn"),      # Sindhi (Arabic)
    "sa": ("san_Deva", "eng_Latn"),      # Sanskrit

    # Munda
    "sat": ("sat_Olck", "eng_Latn"),     # Santali (Ol Chiki)

    # Tibeto-Burman
    "mni": ("mni_Beng", "eng_Latn"),     # Manipuri / Meitei (Bengali script)

    # Dravidian
    "ta": ("tam_Taml", "eng_Latn"),      # Tamil
    "te": ("tel_Telu", "eng_Latn"),      # Telugu
    "kn": ("kan_Knda", "eng_Latn"),      # Kannada
    "ml": ("mal_Mlym", "eng_Latn"),      # Malayalam
}




# Default IndicTrans2 model (supports both directions)
DEFAULT_TRANSLATOR_MODEL = "ai4bharat/indictrans2-en-indic-dist-200M"

def ensure_dirs():
    BASE.mkdir(exist_ok=True)
    LLM_DIR.mkdir(parents=True, exist_ok=True)
    TRANS_DIR.mkdir(parents=True, exist_ok=True)
    RAG_DIR.mkdir(exist_ok=True)

# Local llama binaries directory (release build). If set, use these binaries.
# Example: set env LLAMA_BIN_DIR to something like models/llama-bin or tools/llama
LLAMA_BIN_DIR = Path("C:/Users/karivarkey/Documents/code/mainproj/llama-cli")
