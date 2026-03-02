import json
import re
from flask import request, jsonify, Response, stream_with_context
from . import bp
import app.config as app_config
from app.config import (
    LANG_MAP, LANG_ALIASES, NLLB_LANG_MAP, 
    USE_ONNX_TRANSLATOR, ONNX_LANG_MAP
)
from app.services.translator_service import (
    translate, detect_supported_language, 
    unload_translator, preload_translator, local_translator_path
)
from app.services.onnx_translator_service import (
    translate_onnx, get_onnx_status, 
    unload_onnx_translator, preload_onnx_translator, ensure_onnx_models
)
from app.services.onnx_model_download_service import (
    get_onnx_catalog, download_onnx_models, 
    ensure_onnx_tokenizer, list_downloaded_onnx_model_files, 
    ensure_default_onnx_models
)

ACTIVE_TRANSLATOR = "onnx" if USE_ONNX_TRANSLATOR else "nllb"

def _get_effective_active_translator() -> str:
    global ACTIVE_TRANSLATOR
    if ACTIVE_TRANSLATOR == "onnx":
        try:
            if get_onnx_status().get("available"):
                return "onnx"
        except Exception:
            pass
        return "nllb"
    return "nllb"

@bp.get("/translator_status")
def ep_translator_status():
    onnx_status = get_onnx_status()
    active = _get_effective_active_translator()
    nllb_path = local_translator_path(app_config.NLLB_MODEL)
    nllb_downloaded = nllb_path.exists() and any(nllb_path.iterdir())
    onnx_downloaded_models = list_downloaded_onnx_model_files()
    nllb_downloaded_models = [app_config.NLLB_MODEL] if nllb_downloaded else []

    return jsonify({
        "active_translator": active,
        "onnx": onnx_status,
        "nllb": {
            "available": True,
            "model": app_config.NLLB_MODEL,
            "downloaded": nllb_downloaded,
            "local_path": str(nllb_path),
        },
        "downloaded_models": {
            "onnx": onnx_downloaded_models,
            "nllb": nllb_downloaded_models,
            "all": [
                *[f"onnx:{name}" for name in onnx_downloaded_models],
                *[f"nllb:{name}" for name in nllb_downloaded_models],
            ],
        },
    })

@bp.post("/toggle_translator")
def ep_toggle_translator():
    global ACTIVE_TRANSLATOR
    body = request.get_json() or {}
    use_onnx = body.get("use_onnx", _get_effective_active_translator() == "onnx")
    onnx_status = get_onnx_status()
    if use_onnx and not onnx_status["available"]:
        return jsonify({"error": "ONNX models not available", "details": onnx_status}), 503
    
    if use_onnx:
        unload_translator()
        ACTIVE_TRANSLATOR = "onnx"
    else:
        unload_onnx_translator()
        ACTIVE_TRANSLATOR = "nllb"
    
    return jsonify({"ok": True, "active_translator": ACTIVE_TRANSLATOR})

@bp.post("/translate")
def ep_translate():
    body = request.get_json() or {}
    text = body.get("text")
    target = (body.get("target") or "en").lower()
    stream = bool(body.get("stream", True))
    max_tokens = int(body.get("max_new_tokens", 256))
    use_onnx = body.get("use_onnx", USE_ONNX_TRANSLATOR)

    if not text:
        return jsonify({"error": "text required"}), 400

    src_lang_key = detect_supported_language(text)
    if not src_lang_key:
        return jsonify({"error": "could not auto-detect a supported language"}), 400

    if use_onnx:
        target_map = ONNX_LANG_MAP
        translate_fn = translate_onnx
        backend = "onnx"
        src_code = src_lang_key
        if src_lang_key not in ONNX_LANG_MAP:
            return jsonify({"error": f"Language '{src_lang_key}' not supported by ONNX"}), 400
    else:
        target_map = NLLB_LANG_MAP
        translate_fn = translate
        backend = "nllb"
        src_code, _ = LANG_MAP[src_lang_key]

    target_key = LANG_ALIASES.get(target, target)
    target_code = target_map.get(target_key, target_key)

    sentence_end_re = re.compile(r"(.+?[.!?](?:\"|'|”)?)(\s+|$)", re.S)
    def iter_sentences(blob: str):
        buffer = blob
        while True:
            match = sentence_end_re.search(buffer)
            if not match: break
            sent = match.group(1).strip()
            buffer = buffer[match.end():]
            if sent: yield sent
        if buffer.strip(): yield buffer.strip()

    if not stream:
        translated_sentences = []
        for sent in iter_sentences(text):
            translated = translate_fn(sent, src_code, target_code, max_tokens)
            translated_sentences.append({"source": sent, "translated": translated})
        combined = " ".join(item["translated"] for item in translated_sentences)
        return jsonify({
            "translated_text": combined,
            "detected_lang": src_lang_key,
            "backend": backend
        })

    def event_stream():
        yield f"data: {json.dumps({'type': 'meta', 'backend': backend})}\n\n"
        for idx, sent in enumerate(iter_sentences(text), start=1):
            translated = translate_fn(sent, src_code, target_code, max_tokens)
            yield f"data: {json.dumps({'type': 'sentence', 'index': idx, 'translated': translated})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(stream_with_context(event_stream()), content_type="text/event-stream")

# Catalog and Download Aliases
@bp.get("/onnx_models/catalog")
@bp.get("/api/onnx_models/catalog")
def ep_onnx_models_catalog():
    refresh = str(request.args.get("refresh", "false")).lower() in ("1", "true", "yes")
    return jsonify({"ok": True, **get_onnx_catalog(force_refresh=refresh)})

@bp.post("/onnx_models/download")
@bp.post("/api/onnx_models/download")
def ep_onnx_models_download():
    body = request.get_json() or {}
    files = body.get("files")
    return jsonify({"ok": True, **download_onnx_models(selected_files=files)})