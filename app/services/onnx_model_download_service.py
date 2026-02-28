from __future__ import annotations

import html
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests

from app.config import ONNX_MODELS_DIR, ONNX_TOKENIZER_DIR

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1tN4wqRMMCWfdy-nXOCjaMTv-H5Paj8Zi?usp=drive_link"
DRIVE_FOLDER_ID = "1tN4wqRMMCWfdy-nXOCjaMTv-H5Paj8Zi"

DEFAULT_MODEL_FILES = [
    "m2m100_encoder_w8a32_SAFE.onnx",
    "m2m100_decoder_w8a32.onnx",
    "m2m100_lm_head.onnx",
]
DEFAULT_TOKENIZER_MODEL = "facebook/m2m100_418M"

# Simple in-memory cache to avoid re-scraping Drive for every request
_catalog_cache: Dict[str, object] = {
    "ts": 0.0,
    "files": [],
}
CATALOG_TTL_SECONDS = 300


def _embedded_folder_list_url() -> str:
    return f"https://drive.google.com/embeddedfolderview?id={DRIVE_FOLDER_ID}#list"


def _normalize_name(name: str) -> str:
    return " ".join(name.strip().split())


def _parse_drive_embedded_listing(content: str) -> List[Dict[str, str]]:
    # Example anchor: <a ... href="/file/d/<id>/view?...">file_name.onnx</a>
    anchor_re = re.compile(
        r'<a[^>]*href="([^\"]*/file/d/([A-Za-z0-9_-]+)[^\"]*)"[^>]*>(.*?)</a>',
        re.S,
    )

    out: List[Dict[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for href, file_id, inner_html in anchor_re.findall(content):
        label = re.sub(r"<[^<]+?>", "", inner_html)
        label = _normalize_name(html.unescape(label))
        if not label:
            continue

        full_link = href if href.startswith("http") else f"https://drive.google.com{href}"
        key = (file_id, label)
        if key in seen:
            continue
        seen.add(key)

        out.append({
            "id": file_id,
            "name": label,
            "view_url": full_link,
        })

    return out


def _fetch_drive_catalog() -> List[Dict[str, str]]:
    resp = requests.get(_embedded_folder_list_url(), timeout=30)
    resp.raise_for_status()
    files = _parse_drive_embedded_listing(resp.text)
    if not files:
        raise RuntimeError("No files found in Drive folder listing")
    return files


def get_onnx_catalog(force_refresh: bool = False) -> Dict[str, object]:
    now = time.time()
    cached_ts = float(_catalog_cache.get("ts") or 0.0)
    cached_files = _catalog_cache.get("files") or []

    if (not force_refresh) and cached_files and (now - cached_ts < CATALOG_TTL_SECONDS):
        files = cached_files
    else:
        files = _fetch_drive_catalog()
        _catalog_cache["ts"] = now
        _catalog_cache["files"] = files

    sorted_files = sorted(files, key=lambda item: item["name"].lower())
    return {
        "folder_url": DRIVE_FOLDER_URL,
        "folder_id": DRIVE_FOLDER_ID,
        "default_files": DEFAULT_MODEL_FILES,
        "files": sorted_files,
    }


def _destination_for_model(filename: str) -> Path:
    name = filename.lower()
    if name.startswith("m2m100_encoder"):
        return ONNX_MODELS_DIR / "encoder" / filename
    if name.startswith("m2m100_decoder"):
        return ONNX_MODELS_DIR / "decoder" / filename
    if name.startswith("m2m100_lm_head"):
        return ONNX_MODELS_DIR / "lm_head" / filename
    return ONNX_MODELS_DIR / filename


def is_onnx_tokenizer_ready() -> bool:
    required = [
        ONNX_TOKENIZER_DIR / "tokenizer_config.json",
        ONNX_TOKENIZER_DIR / "sentencepiece.bpe.model",
    ]
    return all(path.exists() for path in required)


def ensure_onnx_tokenizer(force_download: bool = False) -> Dict[str, object]:
    if is_onnx_tokenizer_ready() and not force_download:
        return {
            "ready": True,
            "path": str(ONNX_TOKENIZER_DIR),
            "downloaded": False,
            "model": DEFAULT_TOKENIZER_MODEL,
        }

    try:
        from transformers import AutoTokenizer, M2M100Tokenizer
    except Exception as e:
        raise RuntimeError(f"transformers is required to prepare tokenizer: {e}") from e

    ONNX_TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_MODEL)
        except Exception:
            tokenizer = M2M100Tokenizer.from_pretrained(DEFAULT_TOKENIZER_MODEL)
        tokenizer.save_pretrained(str(ONNX_TOKENIZER_DIR))
    except Exception as e:
        raise RuntimeError(
            f"Failed to download/save tokenizer '{DEFAULT_TOKENIZER_MODEL}' to {ONNX_TOKENIZER_DIR}: {e}"
        ) from e

    if not is_onnx_tokenizer_ready():
        raise RuntimeError(f"Tokenizer files are incomplete at {ONNX_TOKENIZER_DIR}")

    return {
        "ready": True,
        "path": str(ONNX_TOKENIZER_DIR),
        "downloaded": True,
        "model": DEFAULT_TOKENIZER_MODEL,
    }


def _download_file_from_drive(file_id: str, destination: Path) -> None:
    # Handles Drive's confirmation flow for larger files.
    session = requests.Session()
    base_url = "https://drive.google.com/uc?export=download"
    fallback_url = "https://drive.usercontent.google.com/download"

    def _looks_like_html(resp: requests.Response) -> bool:
        content_type = (resp.headers.get("Content-Type") or "").lower()
        if "text/html" in content_type:
            return True
        try:
            prefix = resp.content[:512].lower()
            return b"<!doctype html" in prefix or b"<html" in prefix
        except Exception:
            return False

    def _extract_confirm_params(page_html: str) -> Dict[str, str]:
        params: Dict[str, str] = {}
        for key in ("id", "confirm", "uuid"):
            m = re.search(rf'name="{key}"\s+value="([^\"]+)"', page_html)
            if m:
                params[key] = m.group(1)

        if "confirm" not in params:
            m = re.search(r"confirm=([0-9A-Za-z_\-]+)", page_html)
            if m:
                params["confirm"] = m.group(1)

        if "id" not in params:
            params["id"] = file_id
        return params

    response = session.get(base_url, params={"id": file_id}, stream=True, timeout=60)
    response.raise_for_status()

    confirm_token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirm_token = value
            break

    if confirm_token:
        response = session.get(
            base_url,
            params={"id": file_id, "confirm": confirm_token},
            stream=True,
            timeout=60,
        )
        response.raise_for_status()

    if _looks_like_html(response):
        html_text = response.text
        confirm_params = _extract_confirm_params(html_text)
        if confirm_params.get("confirm"):
            response = session.get(base_url, params=confirm_params, stream=True, timeout=60)
            response.raise_for_status()

    if _looks_like_html(response):
        response = session.get(
            fallback_url,
            params={"id": file_id, "export": "download", "confirm": "t"},
            stream=True,
            timeout=60,
        )
        response.raise_for_status()

    if _looks_like_html(response):
        raise RuntimeError(
            "Google Drive returned an HTML page instead of model binary (possible quota/permission/confirmation issue)."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


def download_onnx_models(selected_files: List[str] | None = None, include_tokenizer: bool = True) -> Dict[str, object]:
    catalog = get_onnx_catalog(force_refresh=False)
    available = {item["name"]: item for item in catalog["files"]}

    requested = selected_files or DEFAULT_MODEL_FILES
    requested = [_normalize_name(name) for name in requested if str(name).strip()]

    if not requested:
        raise RuntimeError("No model files requested")

    unknown = [name for name in requested if name not in available]
    if unknown:
        raise RuntimeError(f"Requested files not found in Drive folder: {', '.join(unknown)}")

    # Auto-include sidecar external tensor files when present (e.g., *.onnx.data)
    expanded_requested: List[str] = list(requested)
    for filename in requested:
        if filename.endswith(".onnx"):
            sidecar = f"{filename}.data"
            if sidecar in available and sidecar not in expanded_requested:
                expanded_requested.append(sidecar)

    downloaded: List[Dict[str, str]] = []
    for filename in expanded_requested:
        file_id = available[filename]["id"]
        destination = _destination_for_model(filename)
        _download_file_from_drive(file_id, destination)
        downloaded.append({
            "name": filename,
            "id": file_id,
            "saved_to": str(destination),
        })

    tokenizer_info = None
    if include_tokenizer:
        tokenizer_info = ensure_onnx_tokenizer(force_download=False)

    return {
        "folder_url": DRIVE_FOLDER_URL,
        "requested_files": expanded_requested,
        "downloaded": downloaded,
        "tokenizer": tokenizer_info,
    }
