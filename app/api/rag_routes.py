import os
import tempfile
from flask import request, jsonify
from werkzeug.utils import secure_filename
from . import bp
from app.services.rag_service import (
    rag_add, rag_remove, rag_list, rag_clear, 
    rag_retrieve, add_pdf_to_rag
)
from app.services.rag_backend import available_backends, load_backend, get_active_backend_name

@bp.post("/rag/add")
def ep_rag_add():
    body = request.get_json(silent=True) or {}
    text = body.get("text") or request.form.get("text")
    if not text:
        return jsonify({"error": "text required"}), 400
    doc_id = rag_add(text.strip())
    return jsonify({"ok": True, "id": doc_id})

@bp.post("/rag/remove")
def ep_rag_remove():
    body = request.get_json() or {}
    doc_id = body.get("id")
    if not doc_id or not rag_remove(doc_id):
        return jsonify({"error": "invalid id"}), 400
    return jsonify({"ok": True})

@bp.get("/rag/list")
def ep_rag_list():
    docs = rag_list()
    return jsonify({"ok": True, "documents": docs, "count": len(docs)})

@bp.post("/rag/clear")
def ep_rag_clear():
    rag_clear()
    return jsonify({"ok": True, "message": "All RAG documents cleared"})

@bp.post("/rag/add_pdf")
def ep_rag_add_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "file required"}), 400
    f = request.files['file']
    filename = secure_filename(f.filename)
    tmp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    tmp_path = os.path.join(tmp_dir, filename)
    try:
        f.save(tmp_path)
        result = add_pdf_to_rag(tmp_path)
        return jsonify({"ok": True, "result": result})
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        if os.path.exists(tmp_dir): os.rmdir(tmp_dir)

@bp.get("/rag/backends")
def ep_rag_backends():
    return jsonify({"available": available_backends(), "active": get_active_backend_name()})

@bp.post("/rag/swap_backend")
def ep_rag_swap_backend():
    body = request.get_json() or {}
    name = body.get("backend")
    load_backend(name)
    return jsonify({"ok": True, "active": name})