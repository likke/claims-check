"""The brand's source pack: parsing, chunking, retrieval and URL fetching.

A source pack is pasted text plus URLs, one URL per line. Pasted text is
indexed straight away. URLs stay pending until ``fetch_source`` reads them.
Retrieval is BM25 over sentence-sized chunks, so a verdict can cite the exact
chunk that supports it.
"""
from __future__ import annotations

import html
import ipaddress
import math
import re
import socket
from collections import Counter
from dataclasses import dataclass, field
from urllib.parse import urlparse

from .text import content_terms, split_sentences

URL_LINE = re.compile(r"^\s*((?:https?|fixture)://\S+)\s*$", re.I)
MAX_FETCH_BYTES = 400_000


@dataclass
class Chunk:
    id: str
    text: str
    origin: str  # "pasted", a URL, or "Rule pack: <name>"

    def as_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "origin": self.origin}


@dataclass
class SourceIndex:
    chunks: list[Chunk] = field(default_factory=list)
    pending_urls: list[str] = field(default_factory=list)
    fetched_urls: dict[str, str] = field(default_factory=dict)  # url -> "ok" or error text

    @classmethod
    def from_pack(cls, text: str, rules: list[str] | None = None) -> "SourceIndex":
        index = cls()
        pasted: list[str] = []
        for line in (text or "").splitlines():
            match = URL_LINE.match(line)
            if match:
                if match.group(1) not in index.pending_urls:
                    index.pending_urls.append(match.group(1))
            else:
                pasted.append(line)
        index.add_document("pasted", "\n".join(pasted), prefix="S")
        for rule in rules or []:
            index.add_document(rule.split("|", 1)[0], rule.split("|", 1)[1], prefix="R")
        return index

    def add_document(self, origin: str, text: str, prefix: str = "S") -> list[Chunk]:
        added = []
        for paragraph in re.split(r"\n\s*\n|\n", text or ""):
            for sentence in split_sentences(paragraph):
                if len(sentence) < 3:
                    continue
                number = sum(1 for c in self.chunks if c.id.startswith(prefix)) + 1
                chunk = Chunk(f"{prefix}{number}", sentence, origin)
                self.chunks.append(chunk)
                added.append(chunk)
        return added

    def get(self, chunk_id: str) -> Chunk | None:
        return next((c for c in self.chunks if c.id == chunk_id), None)

    def search(self, query: str, k: int = 4, include_rules: bool = True) -> list[tuple[Chunk, float]]:
        pool = [c for c in self.chunks if include_rules or not c.id.startswith("R")]
        if not pool:
            return []
        docs = [content_terms(c.text) for c in pool]
        avg = sum(len(d) for d in docs) / len(docs) or 1.0
        df = Counter(term for d in docs for term in set(d))
        query_terms = content_terms(query)
        scored = []
        for chunk, doc in zip(pool, docs):
            tf = Counter(doc)
            score = 0.0
            for term in query_terms:
                if term not in tf:
                    continue
                idf = math.log(1 + (len(pool) - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * tf[term] * 2.2 / (tf[term] + 1.2 * (0.25 + 0.75 * len(doc) / avg))
            scored.append((chunk, round(score, 3)))
        scored.sort(key=lambda item: -item[1])
        hits = [item for item in scored if item[1] > 0][:k]
        return hits or scored[: min(k, 2)]


FIXTURE_DOCS = {
    "fixture://ledgerly-ai/pricing": (
        "Ledgerly AI plans start at 1,490 pesos per month. "
        "A 14-day free trial is available on the Starter plan. "
        "Ledgerly AI is a fictional product used for this demo."
    ),
}


def _public_host(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
            return False
    return True


def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr)>", "\n", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    text = html.unescape(raw)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if len(line) > 2)


def fetch_url(url: str, client=None) -> str:
    """Fetch a public URL as plain text. ``fixture://`` URLs read bundled fictional docs."""
    if url.startswith("fixture://"):
        if url not in FIXTURE_DOCS:
            raise ValueError(f"unknown fixture document {url}")
        return FIXTURE_DOCS[url]
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("only http and https URLs can be fetched")
    if not _public_host(parsed.hostname):
        raise ValueError("refusing to fetch a private or unresolvable host")
    import httpx

    owns = client is None
    client = client or httpx.Client(timeout=10, follow_redirects=True, headers={"User-Agent": "DashoClaimCheck/0.2"})
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            body = b""
            for part in response.iter_bytes():
                body += part
                if len(body) > MAX_FETCH_BYTES:
                    break
            kind = response.headers.get("content-type", "")
    finally:
        if owns:
            client.close()
    text = body.decode("utf-8", errors="replace")
    return html_to_text(text) if "html" in kind or "<html" in text[:500].lower() else text
