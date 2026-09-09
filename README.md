# plot-llm-value

Compare language models using fresh official Artificial Analysis data: capability
on the y-axis and **measured USD per Intelligence Index task** on the x-axis.
Python + Matplotlib; designed for agents that execute a command and read its output.

```sh
uv sync
uv run plot-llm-value --help
```

Or install the command globally in an isolated environment:

```sh
uv tool install .
plot-llm-value --help
```

Set `ARTIFICIAL_ANALYSIS_API_KEY` (or `AA_API_KEY`) using a key from
[Artificial Analysis](https://artificialanalysis.ai/data-api). Alternatively,
store it in an ignored `.env` and pass `--env-file .env` explicitly.

```sh
# Default: a fresh plot, logarithmic cost axis, Intelligence Index.
uv run plot-llm-value --provider anthropic openai

# Discover exact family names, variants, slugs and stable IDs.
uv run plot-llm-value models --provider openai

# All available efforts of a release; substitute a family from models output.
uv run plot-llm-value --model 'gpt-oss-20B' --metric coding

# Exact settings only, without generating a plot.
uv run plot-llm-value data --variant 'gpt-oss-20B (high)'

# Agentic Index, linear cost axis, persistent export.
uv run plot-llm-value --provider anthropic --metric agentic --linear-x --output compare.svg
```

`--help` is the complete operating guide: selectors, API-key precedence, error
codes, output schemas and the image-display workflow. No separate agent skill is
needed. Providers mean model creators (OpenAI, Anthropic, etc.), not inference hosts.

Plots include model names beside their groups, effort labels, model colors,
provider marker shapes, and legends. Labels are placed in rendered coordinates
against points, series and other labels; leader lines are clipped around text.
The canvas grows for larger selections. Dashed connections mark gaps in known
effort ranks; missing intermediate measurements are never interpolated.

The default output is a JSON result containing the absolute path to a temporary
PNG. This is more useful to shell-based agents than binary stdout: they can open
that path with their image tool and display **only the plot** in chat, as the help
instructs. Display depends on the host agent; the CLI cannot control chat rendering.
Temporary files persist until OS cleanup. Use `--output` for PNG, SVG or PDF you
want to keep.

`data` produces structured JSON with all three indices, selected score, measured
cost, explicit exclusions, source/version/retrieval metadata and Pareto membership.
Missing values are null, never zero. Zero cost remains valid in JSON and linear
plots, but cannot be plotted on a logarithmic axis. Even for Coding or Agentic
Index, the cost workload remains **Intelligence Index tasks**, not a measured
coding-specific or agent-specific workload.

Every data-producing command requests all pages from the official
[AA Free API](https://artificialanalysis.ai/data-api/docs), with bounded retries
for network/server errors. There is no cached-data fallback. API quotas apply per
request/page. Attribution is included in plots and JSON.

Development:

```sh
uv sync
uv run pytest
uv run ruff check src tests
uv build
```

Tests use synthetic API-shaped fixtures and verify pagination, selectors,
missing/zero values, HTTP failures, export files, label geometry and CLI output.
Live authenticated validation requires your AA key.
