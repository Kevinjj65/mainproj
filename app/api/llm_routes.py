from flask import request, jsonify
from . import bp
from app.services.llm_service import (
    download_gguf, 
    load_llm_from_gguf, 
    get_current_name, 
    unload_llm, 
    SERVER_URL,
    list_all_llms
)
from app.services.translator_service import unload_translator

@bp.post("/download_llm")
def ep_download_llm():
    body = request.get_json() or {}
    url = body.get("url")
    name = body.get("name")
    if not url:
        return jsonify({"error": "url required"}), 400
    if not name:
        return jsonify({"error": "name required"}), 400
    path = download_gguf(url, name)
    return jsonify({"ok": True, "path": str(path)})

@bp.get("/list_llms")
def ep_list_llms():
    return jsonify({
        "downloaded_llms": list_all_llms(),
        "loaded_llm": get_current_name(),
        "server_url": SERVER_URL,
    })

@bp.post("/load_llm")
def ep_load_llm():
    body = request.get_json() or {}
    name = body.get("name")
    ctx_size = int(body.get("ctx_size", 4096))
    n_gpu_layers = int(body.get("n_gpu_layers", -1))
    if not name:
        return jsonify({"error": "name required"}), 400
    load_llm_from_gguf(name, n_ctx=ctx_size, n_gpu_layers=n_gpu_layers)
    return jsonify({"ok": True, "loaded": name, "server_url": SERVER_URL})

@bp.get("/current_llm")
def ep_current_llm():
    return jsonify({
        "ok": True,
        "loaded_llm": get_current_name(),
        "server_url": SERVER_URL
    })

@bp.post("/unload_llm")
def ep_unload_llm():
    unload_llm()
    translator_unloaded = unload_translator()
    return jsonify({
        "ok": True,
        "message": "LLM server stopped and translator unloaded",
        "translator_unloaded": bool(translator_unloaded)
    })