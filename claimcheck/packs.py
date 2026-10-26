"""Rule packs. A pack adds rule chunks to the source index, hard rules that
code enforces after the model's verdict, and required disclaimers."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class HardRule:
    id: str
    pattern: re.Pattern
    reason: str
    rule_chunk: str  # id of the rule chunk the verdict cites


@dataclass(frozen=True)
class Disclaimer:
    id: str
    label: str
    patterns: tuple[re.Pattern, ...]
    applies_to: str  # "supplement"


@dataclass(frozen=True)
class Pack:
    key: str
    name: str
    guidance: str
    rules: tuple[str, ...]  # "origin|text"
    hard_rules: tuple[HardRule, ...] = field(default_factory=tuple)
    disclaimers: tuple[Disclaimer, ...] = field(default_factory=tuple)


TECH = Pack(
    key="tech",
    name="Tech claims",
    guidance=(
        "Software and AI product claims. A capability, number, integration or outcome must match the "
        "product documentation. Absolute words (guarantees, 100%, zero, never, always, all, lahat) need "
        "explicit support. Opinions and personal experience are not product claims."
    ),
    rules=(
        "Rule pack: Tech claims|An absolute claim such as guarantees, 100 percent, zero or all needs a source that states the same absolute.",
    ),
)

THERAPEUTIC = re.compile(
    r"\b(cures?|cured|treats?|treatment for|heals?|healed|prevents? (?:disease|illness|cancer|acne|diabetes)"
    r"|gamot sa|lunas sa|nagpapagaling|gumagamot|pampagaling|pinapagaling|nakakagaling)\b",
    re.I,
)

FDA_BEAUTY = Pack(
    key="fda_beauty",
    name="FDA beauty",
    guidance=(
        "Philippine FDA rules for cosmetics and food supplements. They may not claim to diagnose, treat, "
        "cure, heal or prevent any disease. Food supplements must carry the line 'No approved therapeutic "
        "claims' (Tagalog: 'Mahalagang paalala: ... hindi gamot at hindi dapat gamiting panggamot')."
    ),
    rules=(
        "Rule pack: FDA beauty|Cosmetics and food supplements may not claim to diagnose, treat, cure, heal or prevent any disease.",
        "Rule pack: FDA beauty|Food supplement advertising must state No approved therapeutic claims, spoken or on screen.",
    ),
    hard_rules=(
        HardRule(
            id="FDA-1",
            pattern=THERAPEUTIC,
            reason="FDA beauty pack: cosmetics and supplements may not claim to treat, cure, heal or prevent disease.",
            rule_chunk="R1",
        ),
    ),
    disclaimers=(
        Disclaimer(
            id="no_therapeutic_claims",
            label="No approved therapeutic claims",
            patterns=(
                re.compile(r"no approved therapeutic claims?", re.I),
                re.compile(r"walang aprubadong therapeutic", re.I),
                re.compile(r"hindi gamot at hindi dapat gamiting panggamot", re.I),
                re.compile(r"mahalagang paalala", re.I),
            ),
            applies_to="supplement",
        ),
    ),
)

PACKS = {TECH.key: TECH, FDA_BEAUTY.key: FDA_BEAUTY}


def get_pack(name: str | None) -> Pack:
    key = (name or "tech").strip().lower().replace(" ", "_").replace("-", "_")
    if key in ("fda", "fda_beauty", "fdabeauty"):
        return FDA_BEAUTY
    if key in ("tech", "tech_claims"):
        return TECH
    return PACKS.get(key, TECH)


def hard_rule_hits(pack: Pack, line: str) -> list[HardRule]:
    return [rule for rule in pack.hard_rules if rule.pattern.search(line or "")]
