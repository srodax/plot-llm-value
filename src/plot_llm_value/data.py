"""Normalize official rows without inventing scores, efforts, or costs."""

import math
import re

from .api import APIError

METRICS = {
    "intelligence": "Intelligence Index",
    "coding": "Coding Index",
    "agentic": "Agentic Index",
}
EFFORTS = {
    "none": 0,
    "non-reasoning": 0,
    "minimal": 1,
    "low": 2,
    "medium": 3,
    "high": 4,
    "xhigh": 5,
    "max": 6,
    "ultra": 7,
    "thinking": 4,
    "reasoning": 4,
}


class SelectionError(Exception):
    """Invalid selectors or no comparable data."""


def split_effort(name):
    match = re.search(r"\s+\(([^()]*)\)$", name)
    if match:
        suffix = match[1].strip().casefold()
        if (
            re.fullmatch(
                r"(?:non-reasoning(?:,\s*(?:low|medium|high))?|none|minimal|low|"
                r"medium|high|xhigh|max|ultra|thinking|reasoning)(?: with fallback)?",
                suffix,
            )
            or suffix == "with fallback"
            or re.fullmatch(r"\d+(?:\.\d+)?k?(?: tokens)?", suffix)
        ):
            return name[: match.start()], suffix
    return name, None


def effort_order(effort):
    if effort is None:
        return (-1, "")
    base = effort.removesuffix(" with fallback")
    if base.startswith("non-reasoning"):
        qualifier = base.partition(",")[2].strip()
        return ({"low": 0.1, "medium": 0.2, "high": 0.3}.get(qualifier, 0), base)
    if base in EFFORTS:
        return (EFFORTS[base], base)
    if re.fullmatch(r"\d+(?:\.\d+)?k?(?: tokens)?", base):
        number = base.removesuffix(" tokens")
        return (float(number.rstrip("k")) * (1000 if number.endswith("k") else 1), base)
    return (99, base)


def _number(value, *, nonnegative=False):
    if type(value) in (int, float) and math.isfinite(value):
        if not nonnegative or value >= 0:
            return value
    return None


def _object(value):
    return value if isinstance(value, dict) else {}


def normalize(raw):
    result = []
    for row in raw:
        try:
            creator = row["model_creator"]
            for value in (row["id"], row["name"], row["slug"], creator["id"], creator["name"]):
                if not isinstance(value, str) or not value.strip():
                    raise ValueError
            family, effort = split_effort(row["name"])
            evaluations = _object(row.get("evaluations"))
            cost = _object(
                _object(row.get("artificial_analysis_intelligence_index_cost")).get("cost_per_task")
            ).get("total_cost")
            result.append(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "slug": row["slug"],
                    "family": family,
                    "effort": effort,
                    "provider": creator["name"],
                    "provider_id": creator["id"],
                    "provider_slug": creator.get("slug")
                    or re.sub(r"[^a-z0-9]+", "-", creator["name"].casefold()).strip("-"),
                    "indices": {
                        key: _number(evaluations.get(f"artificial_analysis_{key}_index"))
                        for key in METRICS
                    },
                    "cost_per_task_usd": _number(cost, nonnegative=True),
                    "url": "https://artificialanalysis.ai/models/" + row["slug"],
                }
            )
        except (KeyError, TypeError, ValueError):
            raise APIError("Invalid model identity in AA response.") from None
    return result


def select(rows, *, providers=None, models=None, variants=None):
    modes = [(providers, "provider"), (models, "model"), (variants, "variant")]
    active = [(values, mode) for values, mode in modes if values]
    if len(active) > 1:
        raise SelectionError("Choose one selection mode: --provider, --model, or --variant.")
    if not active:
        return rows
    values, mode = active[0]
    selected = set()
    for value in values:
        query = value.strip().casefold()
        fields = {
            "provider": ("provider", "provider_id", "provider_slug"),
            "model": ("family",),
            "variant": ("name", "slug", "id"),
        }[mode]
        matches = [row for row in rows if any(row[f].casefold() == query for f in fields)]
        if not matches:
            raise SelectionError(f"Unknown {mode}: {value!r}. Run models to discover exact names.")
        identities = {
            (r["provider_id"], r["family"].casefold())
            if mode == "model"
            else r["provider_id"]
            if mode == "provider"
            else r["id"]
            for r in matches
        }
        if len(identities) != 1:
            raise SelectionError(f"{mode} {value!r} is ambiguous; use exact variant IDs instead.")
        selected.update(row["id"] for row in matches)
    return [row for row in rows if row["id"] in selected]


def compare(rows, metric="intelligence"):
    metric = "agentic" if metric == "agent" else metric
    if metric not in METRICS:
        raise SelectionError(f"Unknown metric {metric!r}; choose intelligence, coding, or agentic.")
    result = []
    for row in rows:
        score, cost = row["indices"][metric], row["cost_per_task_usd"]
        reasons = ([] if score is not None else ["missing score"]) + (
            [] if cost is not None else ["missing cost"]
        )
        result.append(
            {
                **row,
                "score": score,
                "comparable": not reasons,
                "exclusion_reasons": reasons,
                "pareto_optimal": None,
            }
        )
    comparable = [r for r in result if r["comparable"]]
    for row in comparable:
        row["pareto_optimal"] = not any(
            other["cost_per_task_usd"] <= row["cost_per_task_usd"]
            and other["score"] >= row["score"]
            and (
                other["cost_per_task_usd"] < row["cost_per_task_usd"]
                or other["score"] > row["score"]
            )
            for other in comparable
        )
    return {
        "schema_version": 1,
        "metric": {"key": metric, "label": METRICS[metric], "higher_is_better": True},
        "cost": {
            "field": "artificial_analysis_intelligence_index_cost.cost_per_task.total_cost",
            "unit": "USD",
            "workload": "Artificial Analysis Intelligence Index task",
            "lower_is_better": True,
        },
        "rows": result,
        "selected_count": len(result),
        "comparable_count": len(comparable),
        "excluded_count": len(result) - len(comparable),
    }
