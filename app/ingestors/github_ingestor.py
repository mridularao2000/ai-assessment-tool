"""Fetches GitHub repository content for grading evidence.

fetch_repo_content() always pulls the repo's README.md (highest-signal,
always-present file) plus any additional file paths the student explicitly
named in their submission's text_content (e.g. "my implementation is in
src/components/CheckoutWizard.jsx").

The actual fetch goes through llm.fetch_github_content() — the shared,
budget-capped web_fetch path already used by every other tool-enabled call
site in this codebase (AnthropicLLMAdapter._call under the hood). This
module never makes a direct requests/httpx call: it only resolves GitHub
URL shapes into fetchable raw URLs and decides what to fetch.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

from app.exceptions import IngestionError
from app.interfaces.llm import GithubFetchRequest, LLMError, LLMInterface, llm_log_context

# Recognized GitHub hosts a submitted URL may use. raw.githubusercontent.com
# is accepted as input (a student may paste a raw file link) even though
# every OUTPUT url this module builds targets github.com's /raw/HEAD/
# redirect instead (see _raw_url).
_GITHUB_HOSTS = {"github.com", "www.github.com", "raw.githubusercontent.com"}

# Best-effort scan for explicit file-path references in the student's own
# explanation — e.g. "see src/components/CheckoutWizard.jsx for the retry
# logic". Requires at least one path separator and a recognizable
# source/doc extension so it doesn't fire on ordinary prose; this is a
# targeted heuristic, not a general source-tree walk — only paths the
# student actually named get fetched.
_REFERENCED_PATH_RE = re.compile(
    r"(?<![\w/])"
    r"[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)+"
    r"\.(?:py|js|jsx|ts|tsx|java|go|rb|rs|c|h|cpp|hpp|cs|php"
    r"|html|css|scss|json|ya?ml|md|sql|sh)"
    r"(?![\w/])"
)


def _normalize_repo_url(url: str) -> tuple[str, str]:
    """Parse (owner, repo) out of any common GitHub URL shape: a plain
    browser URL, a .git clone URL, a deep /tree/<branch>/... or /blob/...
    link, or a raw.githubusercontent.com URL.

    Raises:
        IngestionError: if the URL isn't a recognizable GitHub repo URL.
    """
    if not url or not url.strip():
        raise IngestionError("No GitHub URL was provided.")

    parsed = urlparse(url.strip())
    host = parsed.netloc.lower()
    if host not in _GITHUB_HOSTS:
        raise IngestionError(f"Not a recognized GitHub URL: {url!r}")

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise IngestionError(f"Could not parse owner/repo from URL: {url!r}")

    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[: -len(".git")]
    return owner, repo


def _raw_url(owner: str, repo: str, path: str) -> str:
    """Build a directly-fetchable raw URL for a path in the repo's default
    branch. github.com's /raw/HEAD/<path> redirects to the raw content of
    whatever the default branch actually is (main, master, ...) without a
    separate API call to look it up first."""
    return f"https://github.com/{owner}/{repo}/raw/HEAD/{path.lstrip('/')}"


def _extract_referenced_paths(text_content: str | None) -> list[str]:
    """Return the distinct file paths the student explicitly named in
    text_content, in first-seen order. Empty when text_content is absent
    or names no recognizable path."""
    if not text_content:
        return []
    seen: dict[str, None] = {}
    for match in _REFERENCED_PATH_RE.finditer(text_content):
        seen.setdefault(match.group(0), None)
    return list(seen)


def fetch_repo_content(
    url: str,
    llm: LLMInterface,
    text_content: str | None = None,
) -> str:
    """Fetch a GitHub repo's README plus any file paths the student
    explicitly referenced in text_content, and return them as a single
    string with each file's content clearly labeled by path — grading
    evidence, ready to embed in a GradingRequest.

    text_content is used only to find paths worth fetching alongside the
    README; it is not itself included in the returned string (the caller
    — GradingService — is responsible for combining the student's written
    explanation with this fetched evidence).

    Raises:
        IngestionError: url isn't a recognizable GitHub repo URL, or the
            underlying fetch genuinely failed (network error, private
            repo, 404, tool budget exhausted).
    """
    owner, repo = _normalize_repo_url(url)

    targets = {"README.md": _raw_url(owner, repo, "README.md")}
    for path in _extract_referenced_paths(text_content):
        targets.setdefault(path, _raw_url(owner, repo, path))

    try:
        with llm_log_context(f"github_ingestor repo={owner}/{repo}"):
            return llm.fetch_github_content(GithubFetchRequest(targets=targets))
    except LLMError as exc:
        raise IngestionError(f"Failed to fetch GitHub repo {url!r}: {exc}") from exc
