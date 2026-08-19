"""Unit tests for the ZenodoClient."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from gwmock.repository.zenodo import ZenodoClient, get_deposition_id_from_doi


@pytest.fixture
def zenodo_client():
    """Fixture to create a ZenodoClient instance."""
    return ZenodoClient(access_token="fake_token", sandbox=True)  # noqa: S106


@pytest.fixture
def mock_response():
    """Fixture to create a mock response."""
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"id": "123", "links": {"bucket": "https://sandbox.zenodo.org/api/files/abc"}}
    return response


class TestZenodoClient:
    """Test suite for ZenodoClient."""

    def test_init(self):
        """Test client initialization."""
        client = ZenodoClient("token", sandbox=True)
        assert client.access_token == "token"  # noqa: S105
        assert client.sandbox is True
        assert client.base_url == "https://sandbox.zenodo.org/api/"
        assert client.headers == {"Authorization": "Bearer token"}

        client_prod = ZenodoClient("token", sandbox=False)
        assert client_prod.base_url == "https://zenodo.org/api/"

    @patch("gwmock.repository.zenodo.requests.request")
    def test_request_success(self, mock_request, zenodo_client, mock_response):
        """Test successful _request call."""
        mock_request.return_value = mock_response
        result = zenodo_client._request("GET", "test_url", headers={}, timeout=60).json()
        assert result == {"id": "123", "links": {"bucket": "https://sandbox.zenodo.org/api/files/abc"}}
        mock_request.assert_called_once()

    @patch("gwmock.utils.retry.time.sleep")  # the retry backoff, which is asserted separately
    @patch("gwmock.repository.zenodo.requests.request")
    def test_request_failure(self, mock_request, mock_sleep, zenodo_client):
        """Test _request with HTTP error."""
        mock_request.side_effect = requests.HTTPError("404")
        with pytest.raises(requests.HTTPError):
            zenodo_client._request("GET", "test_url", headers={}, timeout=60)

    @patch.object(ZenodoClient, "_request")
    def test_create_deposition(self, mock_request_method, zenodo_client, mock_response):
        """Test create_deposition."""
        mock_request_method.return_value = mock_response
        result = zenodo_client.create_deposition(metadata={"title": "Test"})
        assert result == {"id": "123", "links": {"bucket": "https://sandbox.zenodo.org/api/files/abc"}}
        mock_request_method.assert_called_with(
            "POST",
            "https://sandbox.zenodo.org/api/deposit/depositions",
            headers={"Content-Type": "application/json", "Authorization": "Bearer fake_token"},
            timeout=60,
            json={"metadata": {"title": "Test"}},
        )

    @patch.object(ZenodoClient, "_request")
    @patch.object(ZenodoClient, "get_deposition")
    def test_upload_file(self, mock_get_deposition, mock_request_method, zenodo_client, tmp_path, mock_response):
        """Test upload_file."""
        mock_get_deposition.return_value = {"links": {"bucket": "https://sandbox.zenodo.org/api/files/abc"}}
        mock_request_method.return_value = mock_response

        file_path = tmp_path / "test.txt"
        file_path.write_text("test content")

        result = zenodo_client.upload_file("dep123", file_path)
        assert result == {"id": "123", "links": {"bucket": "https://sandbox.zenodo.org/api/files/abc"}}
        mock_request_method.assert_called_once()

    @patch.object(ZenodoClient, "_request")
    def test_update_metadata(self, mock_request_method, zenodo_client, mock_response):
        """Test update_metadata."""
        mock_request_method.return_value = mock_response
        result = zenodo_client.update_metadata("dep123", {"title": "Updated"})
        assert result == {"id": "123", "links": {"bucket": "https://sandbox.zenodo.org/api/files/abc"}}
        mock_request_method.assert_called_with(
            "PUT",
            "https://sandbox.zenodo.org/api/deposit/depositions/dep123",
            headers={"Content-Type": "application/json", "Authorization": "Bearer fake_token"},
            timeout=60,
            data=json.dumps({"metadata": {"title": "Updated"}}),
        )

    @patch.object(ZenodoClient, "_request")
    def test_publish_deposition(self, mock_request_method, zenodo_client, mock_response):
        """Test publish_deposition."""
        mock_response.json.return_value = {"doi": "10.5281/zenodo.123"}
        mock_request_method.return_value = mock_response
        result = zenodo_client.publish_deposition("dep123")
        assert result == {"doi": "10.5281/zenodo.123"}
        mock_request_method.assert_called_with(
            "POST",
            "https://sandbox.zenodo.org/api/deposit/depositions/dep123/actions/publish",
            headers={"Authorization": "Bearer fake_token"},
            timeout=300,
        )

    @patch.object(ZenodoClient, "_request")
    def test_get_deposition(self, mock_request_method, zenodo_client, mock_response):
        """Test get_deposition."""
        mock_response.json.return_value = {"id": "123"}
        mock_request_method.return_value = mock_response
        result = zenodo_client.get_deposition("dep123")
        assert result == {"id": "123"}
        mock_request_method.assert_called_with(
            "GET",
            "https://sandbox.zenodo.org/api/deposit/depositions/dep123",
            headers={"Content-Type": "application/json", "Authorization": "Bearer fake_token"},
            timeout=60,
        )

    @patch.object(ZenodoClient, "_request")
    def test_download_file(self, mock_request_method, zenodo_client, tmp_path):
        """Test download_file."""
        # Mock response object with iter_content for streaming
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_request_method.return_value = mock_response

        output_path = tmp_path / "downloaded.txt"
        _result = zenodo_client.download_file("dep123", "file.txt", output_path)

        # Verify the file was created with correct content
        assert output_path.exists()
        assert output_path.read_bytes() == b"chunk1chunk2"
        # Verify _request was called
        mock_request_method.assert_called_once()

    @patch.object(ZenodoClient, "_request")
    def test_list_depositions(self, mock_request_method, zenodo_client):
        """Test list_depositions."""
        mock_response = MagicMock()
        mock_response.json.return_value = [{"id": "123"}]
        mock_request_method.return_value = mock_response
        result = zenodo_client.list_depositions(status="draft")
        assert result == [{"id": "123"}]
        mock_request_method.assert_called_with(
            "GET",
            "https://sandbox.zenodo.org/api/deposit/depositions",
            headers={"Content-Type": "application/json", "Authorization": "Bearer fake_token"},
            timeout=60,
            params={"status": "draft"},
        )

    @patch.object(ZenodoClient, "_request")
    def test_delete_deposition(self, mock_request_method, zenodo_client):
        """Test delete_deposition."""
        mock_request_method.return_value = {"message": "deleted"}
        result = zenodo_client.delete_deposition("dep123")
        assert result == {"message": "deleted"}
        mock_request_method.assert_called_with(
            "DELETE",
            "https://sandbox.zenodo.org/api/deposit/depositions/dep123",
            headers={"Authorization": "Bearer fake_token"},
            timeout=60,
        )


class TestTheDoiParser:
    """``get_deposition_id_from_doi`` decides *which Zenodo* a later call talks to.

    Getting the sandbox flag backwards points a download at the wrong host, and getting the id
    wrong points it at someone else's record, so both halves of the returned tuple are pinned.
    """

    def test_a_production_doi_gives_the_id_and_production(self):
        assert get_deposition_id_from_doi("10.5281/zenodo.1234567") == ("1234567", False)

    def test_a_sandbox_doi_gives_the_id_and_sandbox(self):
        assert get_deposition_id_from_doi("10.5072/zenodo.7654321") == ("7654321", True)

    def test_the_prefix_is_not_read_as_the_id(self):
        """The prefix contains a dot of its own (``10.5281/zenodo``), so the id is the field after
        the *last* one. Splitting on every dot leaves "10" as the prefix, which matches neither
        host and makes every real DOI invalid."""
        assert get_deposition_id_from_doi("10.5281/zenodo.1234567")[0] == "1234567"

    def test_the_prefix_decides_the_host_rather_than_the_id(self):
        """The same numeric id exists on both hosts, so only the prefix can tell them apart."""
        assert get_deposition_id_from_doi("10.5281/zenodo.42")[1] is False
        assert get_deposition_id_from_doi("10.5072/zenodo.42")[1] is True

    def test_the_sandbox_flag_is_a_boolean(self):
        """Callers pass it straight into ``ZenodoClient(sandbox=...)``, where anything falsy-but-
        not-False silently selects production."""
        assert get_deposition_id_from_doi("10.5072/zenodo.42")[1] is True
        assert get_deposition_id_from_doi("10.5281/zenodo.42")[1] is False

    def test_an_unknown_prefix_is_refused_and_the_doi_is_quoted(self):
        with pytest.raises(ValueError, match=r"Invalid Zenodo DOI: 10\.1234/other\.1"):
            get_deposition_id_from_doi("10.1234/other.1")

    def test_the_prefix_is_matched_case_sensitively(self):
        """A DOI is case-insensitive in the registry but this comparison is not, so an upper-case
        spelling must be refused rather than silently treated as production."""
        with pytest.raises(ValueError, match="Invalid Zenodo DOI"):
            get_deposition_id_from_doi("10.5281/ZENODO.42")

    def test_a_doi_with_no_dot_after_the_prefix_is_not_a_doi(self):
        with pytest.raises(ValueError, match="Invalid Zenodo DOI"):
            get_deposition_id_from_doi("10.5281/zenodo")

    @pytest.mark.parametrize("doi", ["10", "zenodo", "", "10.5281/zenodo.", ".", "10.5281/zenodo.."])
    def test_a_malformed_doi_is_refused_rather_than_half_read(self, doi):
        """Both halves have to be present and non-empty. A string with no dot used to raise
        `IndexError` from the tuple unpacking, and a trailing dot returned an *empty* deposition id
        against a matching prefix -- which the caller then puts in a URL and asks Zenodo for
        nothing."""
        with pytest.raises(ValueError, match="Invalid Zenodo DOI"):
            get_deposition_id_from_doi(doi)


class TestWhatEachCallSends:
    """The method, URL, headers and timeout of every request.

    Every test above mocks ``_request`` and asserts only on the decoded JSON, so a mutation to the
    verb, the URL or the headers is invisible: a ``DELETE`` sent as a ``PUT``, or a publish sent to
    the deposition instead of its publish action, would pass. These pin the request itself.
    """

    @staticmethod
    def _call(mock_request_method):
        return mock_request_method.call_args

    @patch.object(ZenodoClient, "_request")
    def test_create_posts_to_the_depositions_collection(self, request, zenodo_client, mock_response):
        request.return_value = mock_response
        zenodo_client.create_deposition()
        assert self._call(request).args[0] == "POST"
        assert self._call(request).args[1] == "https://sandbox.zenodo.org/api/deposit/depositions"

    @patch.object(ZenodoClient, "_request")
    def test_create_sends_a_json_body_and_says_so(self, request, zenodo_client, mock_response):
        request.return_value = mock_response
        zenodo_client.create_deposition(metadata={"title": "t"})
        kwargs = self._call(request).kwargs
        assert kwargs["json"] == {"metadata": {"title": "t"}}
        assert kwargs["headers"]["Content-Type"] == "application/json"
        assert kwargs["headers"]["Authorization"] == "Bearer fake_token"

    @patch.object(ZenodoClient, "_request")
    def test_create_without_metadata_sends_an_empty_body(self, request, zenodo_client, mock_response):
        """Not ``{"metadata": None}``: Zenodo rejects a null metadata block."""
        request.return_value = mock_response
        zenodo_client.create_deposition()
        assert self._call(request).kwargs["json"] == {}

    @patch.object(ZenodoClient, "_request")
    def test_create_passes_its_timeout(self, request, zenodo_client, mock_response):
        request.return_value = mock_response
        zenodo_client.create_deposition(timeout=7)
        assert self._call(request).kwargs["timeout"] == 7

    @patch.object(ZenodoClient, "_request")
    @patch.object(ZenodoClient, "get_deposition")
    def test_upload_puts_the_file_into_the_bucket_under_its_own_name(
        self, get_deposition, request, zenodo_client, tmp_path, mock_response
    ):
        get_deposition.return_value = {"links": {"bucket": "https://sandbox.zenodo.org/api/files/abc"}}
        request.return_value = mock_response
        path = tmp_path / "frame.gwf"
        path.write_bytes(b"data")

        zenodo_client.upload_file("123", path)

        assert self._call(request).args[0] == "PUT"
        assert self._call(request).args[1] == "https://sandbox.zenodo.org/api/files/abc/frame.gwf"
        get_deposition.assert_called_once_with("123")

    @patch.object(ZenodoClient, "_request")
    @patch.object(ZenodoClient, "get_deposition")
    def test_upload_streams_the_open_file_rather_than_its_bytes(
        self, get_deposition, request, zenodo_client, tmp_path, mock_response
    ):
        """A deposited frame file can be gigabytes; the handle is passed so requests streams it."""
        get_deposition.return_value = {"links": {"bucket": "https://sandbox.zenodo.org/api/files/abc"}}
        request.return_value = mock_response
        path = tmp_path / "frame.gwf"
        path.write_bytes(b"data")

        zenodo_client.upload_file("123", path)

        assert hasattr(self._call(request).kwargs["data"], "read")

    @patch.object(ZenodoClient, "_request")
    @patch.object(ZenodoClient, "get_deposition")
    def test_upload_does_not_claim_the_body_is_json(
        self, get_deposition, request, zenodo_client, tmp_path, mock_response
    ):
        get_deposition.return_value = {"links": {"bucket": "https://sandbox.zenodo.org/api/files/abc"}}
        request.return_value = mock_response
        path = tmp_path / "frame.gwf"
        path.write_bytes(b"data")

        zenodo_client.upload_file("123", path)

        assert "Content-Type" not in self._call(request).kwargs["headers"]

    @patch.object(ZenodoClient, "_request")
    def test_update_metadata_puts_to_the_deposition(self, request, zenodo_client, mock_response):
        request.return_value = mock_response
        zenodo_client.update_metadata("123", {"title": "t"})
        assert self._call(request).args[0] == "PUT"
        assert self._call(request).args[1] == "https://sandbox.zenodo.org/api/deposit/depositions/123"

    @patch.object(ZenodoClient, "_request")
    def test_update_metadata_sends_the_metadata_wrapped_and_serialised(self, request, zenodo_client, mock_response):
        request.return_value = mock_response
        zenodo_client.update_metadata("123", {"title": "t"})
        assert json.loads(self._call(request).kwargs["data"]) == {"metadata": {"title": "t"}}

    @patch.object(ZenodoClient, "_request")
    def test_publish_posts_to_the_publish_action(self, request, zenodo_client, mock_response):
        """The action URL is the difference between publishing and overwriting the draft."""
        request.return_value = mock_response
        zenodo_client.publish_deposition("123")
        assert self._call(request).args[0] == "POST"
        assert self._call(request).args[1].endswith("/deposit/depositions/123/actions/publish")

    @patch.object(ZenodoClient, "_request")
    def test_get_deposition_gets_the_deposition(self, request, zenodo_client, mock_response):
        request.return_value = mock_response
        zenodo_client.get_deposition("123")
        assert self._call(request).args[0] == "GET"
        assert self._call(request).args[1] == "https://sandbox.zenodo.org/api/deposit/depositions/123"

    @patch.object(ZenodoClient, "_request")
    def test_list_depositions_filters_by_status(self, request, zenodo_client):
        request.return_value = MagicMock(json=MagicMock(return_value=[]))
        zenodo_client.list_depositions(status="draft")
        assert self._call(request).kwargs["params"] == {"status": "draft"}

    @patch.object(ZenodoClient, "_request")
    def test_list_depositions_defaults_to_published(self, request, zenodo_client):
        request.return_value = MagicMock(json=MagicMock(return_value=[]))
        zenodo_client.list_depositions()
        assert self._call(request).kwargs["params"] == {"status": "published"}

    @patch.object(ZenodoClient, "_request")
    def test_delete_uses_the_delete_verb(self, request, zenodo_client):
        """A mutation to the verb here would send a PUT to the same URL, which edits rather than
        removes -- and the return value is the response either way."""
        request.return_value = MagicMock()
        zenodo_client.delete_deposition("123")
        assert self._call(request).args[0] == "DELETE"
        assert self._call(request).args[1] == "https://sandbox.zenodo.org/api/deposit/depositions/123"

    @patch.object(ZenodoClient, "_request")
    def test_delete_returns_the_response_rather_than_its_json(self, request, zenodo_client):
        """Zenodo answers a delete with 204 and no body, so ``.json()`` would raise."""
        response = MagicMock()
        request.return_value = response
        assert zenodo_client.delete_deposition("123") is response


class TestDownloading:
    @patch.object(ZenodoClient, "_request")
    def test_it_reads_the_published_record_by_default(self, request, zenodo_client, tmp_path):
        request.return_value = MagicMock(iter_content=MagicMock(return_value=[b"x"]))
        zenodo_client.download_file("123", "frame.gwf", tmp_path / "frame.gwf")
        assert request.call_args.args[1] == "https://sandbox.zenodo.org/api/records/123/files/frame.gwf"

    @patch.object(ZenodoClient, "_request")
    def test_a_draft_is_read_from_the_draft_of_the_record(self, request, zenodo_client, tmp_path):
        request.return_value = MagicMock(iter_content=MagicMock(return_value=[b"x"]))
        zenodo_client.download_file("123", "frame.gwf", tmp_path / "frame.gwf", is_draft=True)
        assert request.call_args.args[1] == "https://sandbox.zenodo.org/api/records/123/draft/files/frame.gwf"

    @patch.object(ZenodoClient, "_request")
    def test_the_body_is_streamed(self, request, zenodo_client, tmp_path):
        request.return_value = MagicMock(iter_content=MagicMock(return_value=[b"x"]))
        zenodo_client.download_file("123", "frame.gwf", tmp_path / "frame.gwf")
        assert request.call_args.kwargs["stream"] is True

    @patch.object(ZenodoClient, "_request")
    def test_the_file_appears_whole_or_not_at_all(self, request, zenodo_client, tmp_path):
        """Written to ``.tmp`` and renamed, so an interrupted download cannot be mistaken for a
        complete file by the next run."""
        seen: list[list[str]] = []

        def chunks(chunk_size):
            seen.append([p.name for p in sorted(tmp_path.iterdir())])
            return [b"ab", b"cd"]

        request.return_value = MagicMock(iter_content=chunks)
        output = tmp_path / "frame.gwf"

        zenodo_client.download_file("123", "frame.gwf", output)

        assert seen == [["frame.tmp"]], "the partial download was not written to a temporary name"
        assert output.read_bytes() == b"abcd"
        assert not (tmp_path / "frame.tmp").exists()


class TestTheTimeoutForBigTransfers:
    """``auto_timeout`` is the difference between a large upload finishing and being cut off.

    The arithmetic is ten seconds per megabyte with a floor, and every operator in it survived
    the first mutation run: the size was measured in bytes, in megabytes-squared, and scaled by a
    tenth, and no test noticed.
    """

    @staticmethod
    def _upload(zenodo_client, request, get_deposition, tmp_path, size_bytes, **kwargs):
        get_deposition.return_value = {"links": {"bucket": "https://example.invalid/bucket"}}
        request.return_value = MagicMock(json=MagicMock(return_value={}))
        path = tmp_path / "frame.gwf"
        path.write_bytes(b"\0" * size_bytes)
        zenodo_client.upload_file("123", path, **kwargs)
        return request.call_args.kwargs["timeout"]

    @patch.object(ZenodoClient, "get_deposition")
    @patch.object(ZenodoClient, "_request")
    def test_ten_seconds_are_allowed_per_megabyte(self, request, get_deposition, zenodo_client, tmp_path):
        """A 3 MiB file with a 1 s floor: 3 MiB -> 30 s, which no other reading of the expression
        produces."""
        assert self._upload(zenodo_client, request, get_deposition, tmp_path, 3 * 1024 * 1024, timeout=1) == 30

    @patch.object(ZenodoClient, "get_deposition")
    @patch.object(ZenodoClient, "_request")
    def test_the_size_is_read_in_mebibytes(self, request, get_deposition, zenodo_client, tmp_path):
        """1 MiB is 1, not 1048576 and not 1/1048576, so the derived timeout is 10 s."""
        assert self._upload(zenodo_client, request, get_deposition, tmp_path, 1024 * 1024, timeout=1) == 10

    @patch.object(ZenodoClient, "get_deposition")
    @patch.object(ZenodoClient, "_request")
    def test_the_given_timeout_is_a_floor_not_a_ceiling(self, request, get_deposition, zenodo_client, tmp_path):
        """A small file keeps the caller's timeout rather than being cut down to a fraction of it."""
        assert self._upload(zenodo_client, request, get_deposition, tmp_path, 1024, timeout=300) == 300

    @patch.object(ZenodoClient, "get_deposition")
    @patch.object(ZenodoClient, "_request")
    def test_the_default_floor_is_five_minutes(self, request, get_deposition, zenodo_client, tmp_path):
        assert self._upload(zenodo_client, request, get_deposition, tmp_path, 1024) == 300

    @patch.object(ZenodoClient, "get_deposition")
    @patch.object(ZenodoClient, "_request")
    def test_it_can_be_switched_off(self, request, get_deposition, zenodo_client, tmp_path):
        timeout = self._upload(
            zenodo_client, request, get_deposition, tmp_path, 3 * 1024 * 1024, timeout=1, auto_timeout=False
        )
        assert timeout == 1

    @patch.object(ZenodoClient, "get_deposition")
    @patch.object(ZenodoClient, "_request")
    def test_auto_timeout_is_on_by_default(self, request, get_deposition, zenodo_client, tmp_path):
        assert self._upload(zenodo_client, request, get_deposition, tmp_path, 3 * 1024 * 1024, timeout=1) == 30

    @patch.object(ZenodoClient, "_request")
    def test_a_declared_download_size_raises_the_timeout(self, request, zenodo_client, tmp_path):
        request.return_value = MagicMock(iter_content=MagicMock(return_value=[b"x"]))
        zenodo_client.download_file("1", "f.gwf", tmp_path / "f.gwf", file_size_in_mb=100)
        assert request.call_args.kwargs["timeout"] == 1000

    @patch.object(ZenodoClient, "_request")
    def test_a_small_declared_download_size_keeps_the_floor(self, request, zenodo_client, tmp_path):
        request.return_value = MagicMock(iter_content=MagicMock(return_value=[b"x"]))
        zenodo_client.download_file("1", "f.gwf", tmp_path / "f.gwf", file_size_in_mb=1)
        assert request.call_args.kwargs["timeout"] == 300

    @patch.object(ZenodoClient, "_request")
    def test_an_unknown_download_size_keeps_the_given_timeout(self, request, zenodo_client, tmp_path):
        request.return_value = MagicMock(iter_content=MagicMock(return_value=[b"x"]))
        zenodo_client.download_file("1", "f.gwf", tmp_path / "f.gwf", timeout=17)
        assert request.call_args.kwargs["timeout"] == 17


class TestTheDefaultHost:
    def test_the_client_talks_to_production_unless_asked_otherwise(self):
        """The default has to be production: a default of sandbox would upload real data to a
        host that deletes it, and every call would still look like it worked."""
        assert ZenodoClient("token").base_url == "https://zenodo.org/api/"
        assert ZenodoClient("token").sandbox is False


class TestTheRetryPolicyAroundEveryRequest:
    """``_request`` is wrapped in ``retry_on_failure()``, so its defaults are this client's policy.

    Asserting the schedule rather than only "it eventually raises" is what makes the delays
    testable without waiting for them -- and the reason the failure test above took seven seconds
    is that nothing did.
    """

    @patch("gwmock.utils.retry.time.sleep")
    @patch("gwmock.repository.zenodo.requests.request")
    def test_a_transient_failure_is_retried_and_the_result_returned(self, request, sleep, zenodo_client):
        response = MagicMock()
        response.raise_for_status.return_value = None
        request.side_effect = [requests.ConnectionError("reset"), response]

        assert zenodo_client._request("GET", "url", headers={}) is response
        assert request.call_count == 2

    @patch("gwmock.utils.retry.time.sleep")
    @patch("gwmock.repository.zenodo.requests.request")
    def test_it_gives_up_after_four_attempts(self, request, sleep, zenodo_client):
        request.side_effect = requests.HTTPError("500")
        with pytest.raises(requests.HTTPError):
            zenodo_client._request("GET", "url", headers={})
        assert request.call_count == 4

    @patch("gwmock.utils.retry.time.sleep")
    @patch("gwmock.repository.zenodo.requests.request")
    def test_the_delay_doubles_between_attempts(self, request, sleep, zenodo_client):
        """One second, then two, then four: a backoff that did not grow would hammer a rate-limited
        API, and one that grew from zero would not wait at all."""
        request.side_effect = requests.HTTPError("500")
        with pytest.raises(requests.HTTPError):
            zenodo_client._request("GET", "url", headers={})
        assert [call.args[0] for call in sleep.call_args_list] == [1.0, 2.0, 4.0]

    @patch("gwmock.utils.retry.time.sleep")
    @patch("gwmock.repository.zenodo.requests.request")
    def test_a_first_attempt_that_works_does_not_wait(self, request, sleep, zenodo_client):
        response = MagicMock()
        response.raise_for_status.return_value = None
        request.return_value = response
        zenodo_client._request("GET", "url", headers={})
        sleep.assert_not_called()

    @patch("gwmock.utils.retry.time.sleep")
    @patch("gwmock.repository.zenodo.requests.request")
    def test_the_request_carries_the_default_timeout(self, request, sleep, zenodo_client):
        response = MagicMock()
        response.raise_for_status.return_value = None
        request.return_value = response
        zenodo_client._request("GET", "url", headers={})
        assert request.call_args.kwargs["timeout"] == 60
