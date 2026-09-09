import pytest


@pytest.fixture
def raw_models():
    """Synthetic values, in the official Free API shape; not a live snapshot."""
    rows = []
    for i, (name, provider, score, cost) in enumerate(
        [
            ("Alpha (high)", "OpenAI", 45, 2.0),
            ("Alpha (low)", "OpenAI", 35, 0.5),
            ("Alpha (medium)", "OpenAI", 42, 1.0),
            ("Beta (max with fallback)", "Anthropic", 48, 3.0),
            ("Beta (low with fallback)", "Anthropic", 38, 0.8),
            ("Gamma (Non-reasoning, high)", "Anthropic", 20, None),
            ("Gamma (max)", "Anthropic", None, 1.0),
            ("Free Model", "Other", 10, 0),
        ]
    ):
        rows.append(
            {
                "id": f"id-{i}",
                "name": name,
                "slug": f"model-{i}",
                "release_date": "2026-01-01",
                "model_creator": {"id": provider.lower(), "name": provider},
                "evaluations": {
                    "artificial_analysis_intelligence_index": score,
                    "artificial_analysis_coding_index": score - 5 if score is not None else None,
                    "artificial_analysis_agentic_index": score + 1 if score is not None else None,
                },
                "artificial_analysis_intelligence_index_cost": {
                    "total_cost": 999,
                    "cost_per_task": {"total_cost": cost},
                },
                "pricing": {"price_1m_input_tokens": 1, "price_1m_output_tokens": 5},
                "performance": {
                    "median_output_tokens_per_second": 100,
                    "median_time_to_first_token_seconds": 1,
                    "median_time_to_first_answer_token_seconds": 2,
                    "median_end_to_end_response_time_seconds": 5,
                },
            }
        )
    return rows
