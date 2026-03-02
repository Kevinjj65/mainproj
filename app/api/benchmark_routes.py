from flask import request, jsonify
from . import bp
import app.config as app_config
from app.config import LANG_MAP, LANG_ALIASES, USE_ONNX_TRANSLATOR
from app.services.benchmark_service import (
    benchmark_pipeline, benchmark_resource_usage, 
    benchmark_llm_metrics, benchmark_translator_metrics, 
    benchmark_rag_metrics
)
from app.services.benchmark_cache_service import benchmark_query_cache

@bp.get("/benchmark")
def ep_benchmark():
    text = request.args.get("text", "കേരളത്തിൽ മഴ കനത്തിരിക്കുന്നു.")
    lang = LANG_ALIASES.get(request.args.get("lang", "ml"), "ml")
    src_lang, en_lang = LANG_MAP[lang]
    return jsonify({"results": benchmark_pipeline(text, src_lang, en_lang)})

@bp.post("/benchmark/resource")
def ep_benchmark_resource():
    body = request.get_json() or {}
    return jsonify({"results": benchmark_resource_usage(**body)})

@bp.post("/llm_metrics")
def ep_llm_metrics():
    body = request.get_json() or {}
    return jsonify({"results": benchmark_llm_metrics(**body)})

@bp.post("/translator_metrics")
def ep_translator_metrics():
    body = request.get_json() or {}
    body["use_onnx"] = body.get("use_onnx", USE_ONNX_TRANSLATOR)
    return jsonify({"results": benchmark_translator_metrics(**body)})

@bp.post("/rag_metrics")
def ep_rag_metrics():
    body = request.get_json() or {}
    return jsonify({"results": benchmark_rag_metrics(**body)})

@bp.post("/benchmark/cache")
def ep_benchmark_cache():
    body = request.get_json() or {}
    return jsonify({"results": benchmark_query_cache(**body)})