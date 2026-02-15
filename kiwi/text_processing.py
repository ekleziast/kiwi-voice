#!/usr/bin/env python3
"""Pure text-processing helpers for Kiwi Voice TTS pipeline."""

import re

from kiwi.utils import kiwi_log


def remove_duplicate_prefixes(text: str) -> str:
    """Remove duplicate prefix in a string (e.g. 'ПриПривет' -> 'Привет')."""
    if len(text) < 4:
        return text

    for length in range(2, len(text) // 2 + 1):
        prefix = text[:length]
        if text.startswith(prefix + prefix):
            pos = 0
            while text[pos:pos + length] == prefix:
                pos += length
            return text[pos - length:]

    return text


def normalize_tts_text(text: str) -> str:
    """Normalize text for TTS: strip markdown, control chars, add pauses."""
    if not text:
        return ""

    paragraph_marker = "__KIWI_PARAGRAPH_PAUSE__"

    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[\u200B-\u200D\u2060\uFEFF]', '', text)  # zero-width chars
    text = re.sub(r'[\x00-\x08\x0B-\x1F\x7F]', ' ', text)    # control chars except \n/\t

    # Remove formatting and noisy technical artifacts
    text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
    text = re.sub(r'`([^`]*)`', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[<>{}\[\]|\\^~]+', ' ', text)
    text = re.sub(r'\bи\s*т\.?\s*д\.?\b', 'и так далее', text, flags=re.IGNORECASE)
    text = re.sub(r'\bи\s*т\.?\s*п\.?\b', 'и тому подобное', text, flags=re.IGNORECASE)

    # Paragraph break => longer pause, single newline => short pause
    text = re.sub(r'\n\s*\n+', f' {paragraph_marker} ', text)
    text = re.sub(r'\n+', ', ', text)

    text = re.sub(r'[ \t\f\v]+', ' ', text).strip()
    text = re.sub(rf'\s*{re.escape(paragraph_marker)}\s*', '. ', text)

    text = re.sub(r'([.!?]){2,}', r'\1', text)
    text = re.sub(r'\s+([,.;:!?])', r'\1', text)
    text = re.sub(r'([,.;:!?])([^\s])', r'\1 \2', text)
    text = re.sub(r'\s{2,}', ' ', text).strip(' ,')
    # Remove leading punctuation left after emoji-only prefixes.
    text = re.sub(r'^[\s\.,;:!?-]+', '', text)

    return text


def clean_chunk_for_tts(chunk: str) -> str:
    """Aggressively clean a TTS chunk: strip JSON artefacts, emoji, markdown, duplicates."""
    if not chunk:
        return ""

    original = chunk

    # 1. Remove JSON delta content patterns
    if "'type':" in chunk or '"type":' in chunk:
        matches = re.findall(r"'text':\s*'([^']*?)'", chunk)
        if matches:
            chunk = "".join(matches)
        else:
            matches = re.findall(r'"text":\s*"([^"]*?)"', chunk)
            if matches:
                chunk = "".join(matches)

    # 2. Strip leftover dict fragments
    chunk = re.sub(r"\{'type':\s*'text',\s*'text':\s*'", "", chunk)
    chunk = re.sub(r"'\}", "", chunk)
    chunk = re.sub(r'\{"type":\s*"text",\s*"text":\s*"', "", chunk)
    chunk = re.sub(r'"\}', "", chunk)

    # 3. Remove emoji
    chunk = re.sub(
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
        r'\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF'
        r'\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF'
        r'\U00002702-\U000027B0\U000024C2-\U0001F251]+',
        '', chunk,
    )

    # 4. Remove markdown
    chunk = re.sub(r'\*\*|\*|__|_|`', '', chunk)

    # 5. Fix prefix duplication
    chunk = remove_duplicate_prefixes(chunk)

    # 6. Normalize whitespace
    chunk = normalize_tts_text(chunk)

    if original != chunk:
        kiwi_log("SPEAK_CHUNK", f"Cleaned: '{original[:50]}...' -> '{chunk[:50]}...'", level="INFO")

    return chunk


def split_text_into_chunks(text: str, max_chunk_size: int = 150) -> list:
    """Split text into sentence-aware chunks for streaming TTS."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            parts = sentence.split(', ')
            for part in parts:
                if len(current_chunk) + len(part) + 2 <= max_chunk_size:
                    current_chunk += part + ", " if part != parts[-1] else part
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip().rstrip(','))
                    current_chunk = part
        else:
            if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]
