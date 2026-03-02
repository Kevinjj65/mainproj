import time
import json
import psutil
import os
from pathlib import Path
from flask import request, jsonify, Response, stream_with_context
from . import bp
import app.config as app_config
from app.services.query_cache_service import QueryCache

QUERY_CACHE_FILE = Path(getattr(app_config, "QUERY_CACHE_FILE", Path("models") / "query_cache.json"))
QUERY_CACHE_SIMILARITY_THRESHOLD = float(getattr(app_config, "QUERY_CACHE_SIMILARITY_THRESHOLD", 0.80))
QUERY_CACHE_MAX_ENTRIES = int(getattr(app_config, "QUERY_CACHE_MAX_ENTRIES", 1000))
QUERY_CACHE_ENABLED = bool(getattr(app_config, "QUERY_CACHE_ENABLED", True))

query_cache = None

def get_query_cache():
    global query_cache
    if query_cache is None and QUERY_CACHE_ENABLED:
        query_cache = QueryCache(
            cache_file=QUERY_CACHE_FILE,
            similarity_threshold=QUERY_CACHE_SIMILARITY_THRESHOLD,
            max_entries=QUERY_CACHE_MAX_ENTRIES,
        )
    return query_cache

@bp.get("/health")
def ep_health():
    return jsonify({"status": "alive"})

@bp.get("/system/metrics")
def ep_system_metrics():
    interval_ms = int(request.args.get("interval_ms", "1000"))
    process = psutil.Process(os.getpid())
    
    def event_stream():
        while True:
            vm = psutil.virtual_memory()
            payload = {
                "type": "metrics",
                "cpu_percent": psutil.cpu_percent(),
                "ram": {"percent": vm.percent, "used_bytes": vm.used},
                "process": {"pid": process.pid, "rss_bytes": process.memory_info().rss}
            }
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(interval_ms / 1000.0)

    return Response(stream_with_context(event_stream()), content_type="text/event-stream")

@bp.get("/query_cache/stats")
def ep_query_cache_stats():
    q = get_query_cache()
    return jsonify(q.stats() if q else {"enabled": False})

@bp.post("/query_cache/clear")
def ep_query_cache_clear():
    q = get_query_cache()
    if q: q.clear()
    return jsonify({"ok": True})