import re

def _clean_generation(text: str) -> str:
    """Sanitize model output by removing noise and context echoes."""
    cleaned = text
    # drop fenced blocks and stray backticks
    cleaned = re.sub(r"```.*?```", " ", cleaned, flags=re.S)
    cleaned = re.sub(r"`+", "", cleaned)
    # remove common prefixes
    cleaned = re.sub(r"^\s*Answer\s*:\s*", "", cleaned, flags=re.I)
    # strip out any residual context echoes
    cleaned = re.sub(r"CONTEXT:.*", "", cleaned, flags=re.S|re.I)
    cleaned = re.sub(r"Document \d+:.*", "", cleaned, flags=re.S|re.I)
    # collapse whitespace and trim
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()

def _clean_rag_context(docs: list) -> str:
    """
    Clean RAG documents: filter tables, remove incomplete lines, extract main text.
    Returns cleaned, readable context for LLM prompt.
    """
    if not docs:
        return ""
    
    cleaned_docs = []
    for doc in docs:
        text = doc.get("text") if isinstance(doc, dict) else str(doc)
        if not text:
            continue
        
        # Remove table-like formatting (lines with many |, -, ✓, ✗, etc.)
        lines = text.split('\n')
        filtered_lines = []
        for line in lines:
            if re.match(r'^[\s\|:=\-\+]+$', line):
                continue
            if line.count('|') > 3 or line.count('—') > 2:
                continue
            if re.match(r'^\s*\d+\s+[A-Z]\s*$', line):
                continue
            line_clean = line.strip()
            if line_clean and len(line_clean) > 10:
                filtered_lines.append(line_clean)
        
        cleaned_text = " ".join(filtered_lines)
        if len(cleaned_text) > 500:
            cleaned_text = cleaned_text[:500]
            for punct in '.!?।':
                idx = cleaned_text.rfind(punct)
                if idx > 200:
                    cleaned_text = cleaned_text[:idx+1]
                    break
        
        if cleaned_text and len(cleaned_text) > 20:
            cleaned_docs.append(cleaned_text)
    
    return "\n".join(cleaned_docs)

def _ensure_complete_sentence(text: str) -> str:
    """
    Ensure text ends with a sentence terminator. If cut mid-word, backtrack to last space.
    """
    if not text:
        return text
    
    text = text.strip()
    if not text:
        return text
    
    if text[-1] in '.!?।':
        return text
    
    if text[-1].isalnum():
        last_space = text.rfind(' ')
        if last_space > len(text) / 2:
            text = text[:last_space]
            if text and text[-1] not in '.!?।':
                text = text.rstrip() + '.'
            return text
    
    return text + '.'