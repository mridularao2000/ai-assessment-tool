"""Unit tests for app.ingestors.github_ingestor.

fetch_repo_content() never makes a direct HTTP call — it resolves GitHub
URL shapes to fetchable raw URLs and delegates the actual fetch to an
injected LLMInterface (llm.fetch_github_content), which in production
goes through AnthropicLLMAdapter's shared, budget-capped web_fetch path.
Here that boundary is a plain stub — no network, no Anthropic SDK call.
"""
from __future__ import annotations

import pytest

from app.exceptions import IngestionError
from app.ingestors.github_ingestor import fetch_repo_content
from app.interfaces.llm import GithubFetchRequest, LLMUnavailableError, LLMValidationError


class _StubLLM:
    """Records the GithubFetchRequest it received; returns a canned
    response or raises a canned exception, standing in for the real
    web_fetch-backed AnthropicLLMAdapter.fetch_github_content."""

    def __init__(self, response: str = "", exc: Exception | None = None) -> None:
        self._response = response
        self._exc = exc
        self.last_request: GithubFetchRequest | None = None

    def fetch_github_content(self, request: GithubFetchRequest) -> str:
        self.last_request = request
        if self._exc is not None:
            raise self._exc
        return self._response


class TestFetchRepoContent:

    def test_fetches_readme(self):
        llm = _StubLLM(response="=== README.md ===\n# My Project\nDescription here.")
        result = fetch_repo_content("https://github.com/octocat/hello-world", llm)

        assert llm.last_request is not None
        assert "README.md" in llm.last_request.targets
        assert (
            llm.last_request.targets["README.md"]
            == "https://github.com/octocat/hello-world/raw/HEAD/README.md"
        )
        assert "README.md" in result
        assert "My Project" in result

    def test_fetches_referenced_paths_from_text_content(self):
        llm = _StubLLM(response="=== README.md ===\ncontent")
        text_content = (
            "My checkout retry logic is in src/components/CheckoutWizard.jsx, "
            "and the API client is at src/api/client.ts."
        )
        fetch_repo_content(
            "https://github.com/octocat/hello-world", llm, text_content=text_content
        )

        assert llm.last_request is not None
        targets = llm.last_request.targets
        assert "README.md" in targets
        assert "src/components/CheckoutWizard.jsx" in targets
        assert "src/api/client.ts" in targets
        assert (
            targets["src/components/CheckoutWizard.jsx"]
            == "https://github.com/octocat/hello-world/raw/HEAD/src/components/CheckoutWizard.jsx"
        )

    def test_no_referenced_paths_when_text_content_absent(self):
        llm = _StubLLM(response="=== README.md ===\ncontent")
        fetch_repo_content("https://github.com/octocat/hello-world", llm)
        assert list(llm.last_request.targets.keys()) == ["README.md"]

    def test_no_referenced_paths_when_text_has_no_file_paths(self):
        llm = _StubLLM(response="=== README.md ===\ncontent")
        fetch_repo_content(
            "https://github.com/octocat/hello-world", llm,
            text_content="I think the event loop handles this correctly.",
        )
        assert list(llm.last_request.targets.keys()) == ["README.md"]

    def test_duplicate_referenced_paths_fetched_once(self):
        llm = _StubLLM(response="=== README.md ===\ncontent")
        fetch_repo_content(
            "https://github.com/octocat/hello-world", llm,
            text_content="See src/foo.py for details. Also src/foo.py has the retry loop.",
        )
        assert list(llm.last_request.targets.keys()) == ["README.md", "src/foo.py"]

    def test_raises_ingestion_error_on_llm_unavailable(self):
        """A genuine network-level failure (simulated: the underlying
        Anthropic call raising LLMUnavailableError, as it would on a 404,
        private repo, or connection error) must surface as IngestionError,
        not the raw LLM exception and not ImportError."""
        llm = _StubLLM(exc=LLMUnavailableError("Claude API unreachable: 404 Not Found"))
        with pytest.raises(IngestionError) as exc_info:
            fetch_repo_content("https://github.com/octocat/nonexistent-repo", llm)
        assert not isinstance(exc_info.value, ImportError)

    def test_raises_ingestion_error_on_tool_budget_exhaustion(self):
        llm = _StubLLM(exc=LLMValidationError("No text block in response content"))
        with pytest.raises(IngestionError):
            fetch_repo_content("https://github.com/octocat/hello-world", llm)

    def test_import_succeeds_cleanly(self):
        """The whole point of Fix 1: this module must exist and import
        without error — no more ImportError swallowed by a bare except."""
        import app.ingestors.github_ingestor  # noqa: F401

    @pytest.mark.parametrize("url", [
        "https://github.com/octocat/hello-world",
        "https://github.com/octocat/hello-world.git",
        "https://github.com/octocat/hello-world/",
        "https://github.com/octocat/hello-world/tree/main",
        "https://www.github.com/octocat/hello-world",
    ])
    def test_normalizes_common_url_shapes_to_same_repo(self, url):
        llm = _StubLLM(response="=== README.md ===\ncontent")
        fetch_repo_content(url, llm)
        assert (
            llm.last_request.targets["README.md"]
            == "https://github.com/octocat/hello-world/raw/HEAD/README.md"
        )

    def test_raw_githubusercontent_url_accepted(self):
        llm = _StubLLM(response="=== README.md ===\ncontent")
        fetch_repo_content(
            "https://raw.githubusercontent.com/octocat/hello-world/main/README.md", llm
        )
        assert (
            llm.last_request.targets["README.md"]
            == "https://github.com/octocat/hello-world/raw/HEAD/README.md"
        )

    def test_non_github_url_raises_ingestion_error(self):
        llm = _StubLLM(response="should never be reached")
        with pytest.raises(IngestionError):
            fetch_repo_content("https://gitlab.com/octocat/hello-world", llm)
        assert llm.last_request is None  # never even attempted a fetch

    def test_empty_url_raises_ingestion_error(self):
        llm = _StubLLM(response="should never be reached")
        with pytest.raises(IngestionError):
            fetch_repo_content("", llm)

    def test_url_without_repo_path_raises_ingestion_error(self):
        llm = _StubLLM(response="should never be reached")
        with pytest.raises(IngestionError):
            fetch_repo_content("https://github.com/octocat", llm)
