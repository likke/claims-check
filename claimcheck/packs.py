"""Source packs used by the review agent."""
from dataclasses import dataclass

TECH_SOURCE = "Ledgerly AI helps teams organize invoices and flag unusual charges for human review. It does not guarantee fraud prevention or regulatory compliance."
FDA_BEAUTY_SOURCE = "FDA beauty claims must not promise to diagnose, treat, cure, or prevent disease. Cosmetic benefits should be truthful and supported by evidence."


@dataclass(frozen=True)
class SourcePack:
    name: str
    guidance: str


TECH_PACK = SourcePack("Tech claims", TECH_SOURCE)
FDA_BEAUTY_PACK = SourcePack("FDA beauty", FDA_BEAUTY_SOURCE)


def pack_guidance(name: str) -> str:
    if name.lower() == "fda beauty":
        return FDA_BEAUTY_SOURCE
    return TECH_SOURCE
