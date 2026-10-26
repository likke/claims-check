"""Compatibility entry point for Dasho Claim Check."""
from claimcheck.agent import extract_claims, run_review
from claimcheck.media import extract_media
from claimcheck.packs import TECH_SOURCE as FIXTURE_SOURCE
from claimcheck.providers import FixtureProvider, MetaProvider, OpenAICompatibleProvider, get_provider
from claimcheck.tools import check_coverage, check_disclosure, fetch_source, judge_claim, rewrite_line, search_source
from claimcheck.web import REVIEWS, app
