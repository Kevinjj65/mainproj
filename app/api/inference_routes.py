import json
import time
import re
from flask import request, jsonify, Response, stream_with_context
from . import bp
from .utils import _clean_generation, _clean_rag_context, _ensure_complete_sentence
from .translate_routes import _get_effective_active_translator
from app.config import LANG_MAP, LANG_ALIASES, ONNX_LANG_MAP
from app.services.llm_service import llm_generate, llm_generate_stream
from app.services.translator_service import translate, detect_supported_language
from app.services.onnx_translator_service import translate_onnx
from app.services.rag_service import rag_retrieve, get_embed_model
from .system_routes import get_query_cache

@bp.post("/infer")
def ep_infer():
    body = request.get_json() or {}
    text = body.get("text")
    lang = body.get("lang", "auto").lower()
    max_tokens = int(body.get("max_new_tokens", 128))
    stream = bool(body.get("stream", True))

    if not text:
        return jsonify({"error": "text required"}), 400

    detected_lang = detect_supported_language(text) if lang == "auto" else lang
    lang = LANG_ALIASES.get(detected_lang, detected_lang) or "en"
    
    src_lang, _ = LANG_MAP.get(lang, ("eng_Latn", "en"))
    is_source_english = (src_lang == "eng_Latn")
    
    backend = _get_effective_active_translator()
    if backend == "onnx" and lang not in ONNX_LANG_MAP:
        backend = "nllb"

    def _translate_pipe(t, s, tg):
        return translate_onnx(t, s, tg) if backend == "onnx" else translate(t, s, tg)

    # 1. To English
    _t0 = time.perf_counter()
    english_text = text if is_source_english else _translate_pipe(text, lang if backend=="onnx" else src_lang, "en")
    translate_in_s = time.perf_counter() - _t0

    # 2. RAG & Cache
    qcache = get_query_cache()
    cache_hit, rag_docs = False, []
    if qcache:
        cached = qcache.find_similar_query(get_embed_model().encode([english_text])[0].tolist())
        if cached:
            rag_docs, _ = cached
            cache_hit = True

    if not cache_hit:
        rag_docs = rag_retrieve(english_text)
        if qcache: qcache.add_query(english_text, get_embed_model().encode([english_text])[0].tolist(), rag_docs)

    context = _clean_rag_context(rag_docs)
    if not context:
        # Fallback Logic
        fallback_en = "I'm sorry, I don't have information on that topic."
        answer_native = fallback_en if is_source_english else _translate_pipe(fallback_en, "en", lang if backend=="onnx" else src_lang)
        return jsonify({"final_output": answer_native, "out_of_bounds": True})

    # 3. Prompt
    final_prompt = f"Context:\n{context}\n\nQuestion: {english_text}\nAnswer (one sentence):"

    if not stream:
        llm_out = _ensure_complete_sentence(_clean_generation(llm_generate(final_prompt, max_new_tokens=max_tokens)))
        final_out = llm_out if is_source_english else _translate_pipe(llm_out, "en", lang if backend=="onnx" else src_lang)
        return jsonify({"final_output": _ensure_complete_sentence(final_out)})

    def event_stream():
        sentence_end_re = re.compile(r"(.+?[.!?](?:\"|'|”)?)(\s+|$)", re.S)
        buffer = ""
        yield f"data: {json.dumps({'type': 'meta', 'english_in': english_text})}\n\n"
        for chunk in llm_generate_stream(final_prompt, max_new_tokens=max_tokens):
            buffer += chunk
            while True:
                m = sentence_end_re.search(buffer)
                if not m: break
                sent_clean = _clean_generation(m.group(1))
                buffer = buffer[m.end():]
                translated = sent_clean if is_source_english else _translate_pipe(sent_clean, "en", lang if backend=="onnx" else src_lang)
                yield f"data: {json.dumps({'type': 'sentence', 'translated': _ensure_complete_sentence(translated)})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(stream_with_context(event_stream()), content_type="text/event-stream")

@bp.post("/infer_raw")
def ep_infer_raw():
    body = request.get_json() or {}
    prompt = body.get("prompt")
    use_rag = body.get("use_rag", True)
    rag_docs = rag_retrieve(prompt) if use_rag else []
    final_prompt = f"Context: {rag_docs}\n\nQuestion: {prompt}" if rag_docs else prompt
    return jsonify({"output": llm_generate(final_prompt)})