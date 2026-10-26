"""Text helpers: tokenising, sentence timing and quote location.

Every timestamp the app shows comes from here. ASR turns carry real start and
end times; when a turn holds several sentences, each sentence's time is
estimated from its share of the turn's words and marked ``estimated``.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, asdict

WORD = re.compile(r"[\w%']+", re.UNICODE)
SENTENCE = re.compile(r"[^.!?]+(?:[.!?]+|$)")

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "was",
    "it", "its", "this", "that", "be", "by", "as", "at", "our", "your", "you", "we", "i", "my",
    "before", "after", "from", "so", "but", "not", "does", "do", "gets", "get", "goes", "each",
    "every", "any", "ang", "ng", "sa", "na", "ito", "yung", "mga", "po", "ko", "mo", "siya",
}

# Common Tagalog function words. A line with two or more is treated as Taglish,
# and a rewrite of a Taglish line has to keep at least one.
TAGALOG_MARKERS = {
    "ang", "ng", "sa", "na", "ito", "siya", "po", "talaga", "mo", "ko", "ka", "kasi", "yung",
    "mga", "para", "bago", "dito", "lang", "naman", "grabe", "sobrang", "sulit", "uy", "nyo",
    "n'yo", "ba", "din", "rin", "pa", "kayo", "natin", "namin", "tayo", "ay", "hindi", "may",
    "wala", "walang", "agad", "lahat", "kami", "ninyo", "niya", "pag", "kapag", "eh",
}


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = text.replace("’", "'").replace("‘", "'")
    return text


def tokens(text: str) -> list[str]:
    return WORD.findall(normalise(text))


def stem(word: str) -> str:
    """A small suffix stripper: "flagged", "flags" and "flag" all become "flag"."""
    for suffix in ("ings", "ing", "ers", "er", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiou%0123456789":
        word = word[:-1]
    return word


def content_terms(text: str) -> list[str]:
    return [stem(word) for word in tokens(text) if word not in STOPWORDS and len(word) > 1]


def is_taglish(text: str) -> bool:
    words = set(tokens(text))
    return len(words & TAGALOG_MARKERS) >= 2


def has_tagalog(text: str) -> bool:
    return bool(set(tokens(text)) & TAGALOG_MARKERS)


@dataclass
class Sentence:
    index: int
    start: float
    end: float
    text: str
    estimated: bool

    def as_dict(self) -> dict:
        data = asdict(self)
        data["start"] = round(self.start, 2)
        data["end"] = round(self.end, 2)
        return data


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE.findall(text or "") if part.strip()]


def sentences_from_turns(turns: list[dict]) -> list[Sentence]:
    """Turn ASR turns ``{start, end, text}`` into timed sentences.

    A turn with one sentence keeps the ASR times exactly. A turn with several
    sentences spreads its duration by word count; those times are estimates.
    """
    out: list[Sentence] = []
    for turn in turns:
        parts = split_sentences(turn.get("text", ""))
        if not parts:
            continue
        start, end = float(turn.get("start", 0.0)), float(turn.get("end", 0.0))
        turn_estimated = bool(turn.get("estimated", False))
        if len(parts) == 1:
            out.append(Sentence(len(out), start, end, parts[0], turn_estimated))
            continue
        total = sum(max(1, len(tokens(p))) for p in parts)
        cursor = start
        for part in parts:
            span = (end - start) * max(1, len(tokens(part))) / total
            out.append(Sentence(len(out), cursor, cursor + span, part, True))
            cursor += span
    return out


def containment(quote: str, text: str) -> float:
    """Share of the quote's words found in ``text`` (0 to 1)."""
    q = tokens(quote)
    if not q:
        return 0.0
    pool = tokens(text)
    remaining = list(pool)
    hits = 0
    for word in q:
        if word in remaining:
            remaining.remove(word)
            hits += 1
    return hits / len(q)


def locate(quote: str, sentences: list[Sentence], threshold: float = 0.8) -> tuple[list[Sentence], float]:
    """Find the sentence (or two adjacent sentences) that contain ``quote``."""
    candidates = []
    for i, sentence in enumerate(sentences):
        candidates.append((containment(quote, sentence.text), 1, [sentence]))
        if i + 1 < len(sentences):
            pair = sentences[i : i + 2]
            candidates.append((containment(quote, pair[0].text + " " + pair[1].text) - 0.05, 2, list(pair)))
    if not candidates:
        return [], 0.0
    score, _, matched = max(candidates, key=lambda c: (c[0], -c[1]))
    if score < threshold:
        return [], score
    return matched, min(1.0, score)


def quote_in(quote: str, text: str, threshold: float = 0.9) -> bool:
    """True when the quote's words appear in ``text`` (allows small ASR/punctuation drift)."""
    if not quote or not quote.strip():
        return False
    if normalise(quote).strip(" .\"'") in normalise(text):
        return True
    return containment(quote, text) >= threshold and len(tokens(quote)) >= 3


def quote_span(quote: str, matched: list[Sentence]) -> tuple[float, float, bool]:
    """Start and end time of ``quote`` inside the matched sentences.

    Word times are spread evenly across each sentence, so a quote that covers
    only part of a sentence gets an estimated time (marked ``True``). A quote
    that covers whole sentences keeps their times and their own flag.
    """
    words = []
    for sentence in matched:
        sentence_words = tokens(sentence.text)
        step = (sentence.end - sentence.start) / max(1, len(sentence_words))
        for k, word in enumerate(sentence_words):
            words.append((word, sentence.start + k * step, sentence.start + (k + 1) * step))
    quote_words = tokens(quote)
    whole = (round(matched[0].start, 2), round(matched[-1].end, 2), any(s.estimated for s in matched)) if matched else (0.0, 0.0, True)
    if not words or not quote_words or len(quote_words) >= len(words):
        return whole
    size = len(quote_words)
    best_index, best_score = 0, -1
    for i in range(len(words) - size + 1):
        window = [w[0] for w in words[i : i + size]]
        score = sum(1 for word in quote_words if word in window)
        if score > best_score:
            best_index, best_score = i, score
    if best_index == 0 and size == len(words):
        return whole
    return round(words[best_index][1], 2), round(words[best_index + size - 1][2], 2), True
