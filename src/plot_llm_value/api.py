"""Official AA Free API transport; one fresh complete snapshot per call."""

import math
import time
from datetime import datetime, timezone

import httpx

ENDPOINT = "https://artificialanalysis.ai/api/v2/language/models/free"


class APIError(Exception):
    """Actionable transport or upstream schema failure."""


def _page(client, key, number, timeout, sleep):
    for attempt in range(3):
        try:
            response = client.get(
                ENDPOINT, params={"page": number}, headers={"x-api-key": key}, timeout=timeout
            )
        except httpx.TransportError:
            if attempt < 2:
                sleep(0.5 * 2**attempt)
                continue
            raise APIError("Network request failed after 3 attempts; check connectivity.") from None
        if response.status_code >= 500 and attempt < 2:
            sleep(0.5 * 2**attempt)
            continue
        if response.status_code != 200:
            messages = {
                401: "Invalid API key; check ARTIFICIAL_ANALYSIS_API_KEY.",
                403: "API access denied for this key.",
                429: "Rate limit exhausted. Retry-After: "
                + response.headers.get("Retry-After", "not supplied"),
            }
            raise APIError(
                messages.get(
                    response.status_code,
                    f"AA returned HTTP {response.status_code}; try again later.",
                )
            )
        try:
            return response.json()
        except ValueError:
            raise APIError("AA returned invalid JSON.") from None


def fetch_models(key, *, client=None, timeout=30, sleep=time.sleep):
    if not key or not key.strip():
        raise APIError("Set ARTIFICIAL_ANALYSIS_API_KEY (or AA_API_KEY).")
    if not math.isfinite(timeout) or timeout <= 0:
        raise APIError("Timeout must be positive and finite.")
    if client is None:
        with httpx.Client(follow_redirects=False) as owned:
            return fetch_models(key, client=owned, timeout=timeout, sleep=sleep)
    rows, seen, version = [], set(), None
    for number in range(1, 1001):
        body = _page(client, key, number, timeout, sleep)
        try:
            data, pagination, current_version = (
                body["data"],
                body["pagination"],
                body["intelligence_index_version"],
            )
            valid = (
                isinstance(data, list)
                and isinstance(pagination, dict)
                and type(pagination["page"]) is int
                and pagination["page"] == number
                and type(pagination["has_more"]) is bool
                and type(current_version) in (int, float)
                and math.isfinite(current_version)
                and current_version > 0
            )
            if not valid:
                raise ValueError
            if version is not None and version != current_version:
                raise APIError(
                    "Index version changed during pagination; rerun for a consistent snapshot."
                )
            version = current_version
            for row in data:
                row_id = row["id"]
                if not isinstance(row_id, str) or not row_id or row_id in seen:
                    raise APIError("Duplicate or invalid model ID in API pages; rerun.")
                seen.add(row_id)
            rows.extend(data)
            if not pagination["has_more"]:
                return {
                    "data": rows,
                    "source": {
                        "name": "Artificial Analysis",
                        "url": "https://artificialanalysis.ai/",
                        "endpoint": ENDPOINT,
                        "retrieved_at": datetime.now(timezone.utc).isoformat(),
                        "intelligence_index_version": version,
                    },
                }
            if not data:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            raise APIError(f"Invalid AA response schema on page {number}.") from None
    raise APIError("AA pagination exceeded 1000 pages; refusing an incomplete snapshot.")
