import httpx
import pytest

from plot_llm_value.api import APIError, fetch_models


def page(number=1, more=False, version=4.3, rows=None):
    return {
        "tier": "free",
        "intelligence_index_version": version,
        "pagination": {
            "page": number,
            "page_size": 200,
            "total_pages": 2 if more or number == 2 else 1,
            "has_more": more,
        },
        "data": rows if rows is not None else [{"id": str(number)}],
    }


def test_follows_pages_and_fetches_again_without_cache():
    requests = []

    def respond(request):
        requests.append(request)
        n = int(request.url.params["page"])
        return httpx.Response(200, json=page(n, more=n == 1))

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        first = fetch_models("test-key", client=client)
        second = fetch_models("test-key", client=client)
    assert first["data"] == [{"id": "1"}, {"id": "2"}]
    assert second["data"] == first["data"]
    assert first["source"]["intelligence_index_version"] == 4.3
    assert first["source"]["retrieved_at"].endswith("+00:00")
    assert len(requests) == 4
    assert all(r.headers["x-api-key"] == "test-key" for r in requests)
    assert all(r.url.host == "artificialanalysis.ai" for r in requests)
    assert all(r.url.path == "/api/v2/language/models/free" for r in requests)


@pytest.mark.parametrize(
    "status, message", [(401, "API key"), (403, "access"), (429, "Retry-After: 17")]
)
def test_http_errors_are_actionable_and_do_not_leak_key(status, message):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(status, headers={"Retry-After": "17"}, text="SECRET")
        )
    ) as client:
        with pytest.raises(APIError, match=message) as error:
            fetch_models("SECRET", client=client)
    assert "SECRET" not in str(error.value)


def test_transient_server_error_retries_then_succeeds():
    attempts, delays = [], []

    def respond(request):
        attempts.append(1)
        return httpx.Response(503) if len(attempts) < 3 else httpx.Response(200, json=page())

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = fetch_models("key", client=client, sleep=delays.append)
    assert result["data"] == [{"id": "1"}]
    assert delays == [0.5, 1.0]


@pytest.mark.parametrize(
    "body", [[], {}, {"data": []}, page(rows="bad"), page(more="yes"), page(version=None)]
)
def test_rejects_malformed_envelope(body):
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body))
    ) as client:
        with pytest.raises(APIError):
            fetch_models("key", client=client)


def test_rejects_invalid_json():
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text="<html>"))
    ) as client:
        with pytest.raises(APIError, match="JSON"):
            fetch_models("key", client=client)


@pytest.mark.parametrize("second", [page(2, version=4.4), page(1), page(2, rows=[{"id": "1"}])])
def test_rejects_inconsistent_pages_instead_of_partial_comparison(second):
    def respond(request):
        return httpx.Response(
            200, json=page(1, True) if request.url.params["page"] == "1" else second
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(APIError):
            fetch_models("key", client=client)


def test_missing_key_fails_before_network():
    with pytest.raises(APIError, match="ARTIFICIAL_ANALYSIS_API_KEY"):
        fetch_models("")
