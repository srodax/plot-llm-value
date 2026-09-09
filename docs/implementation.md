# Implementation design and red/green plan

Use the official `/api/v2/language/models/free` endpoint, including pagination
and index version. Preserve missing values. Never substitute token pricing for
`artificial_analysis_intelligence_index_cost.cost_per_task.total_cost`.

The CLI help is the public specification; no separate skill wrapper. Default
plots use an ephemeral PNG path in JSON rather than binary stdout: agents can
inspect structured output and invoke their native image tool. Explicit output
paths support PNG/SVG/PDF. Data mode writes JSON only.

Implementation units and order:

- [x] API: `api.py`, tests of fresh paginated retrieval, credentials, version
  consistency, rate limits, invalid JSON/schema, duplicate IDs and retries.
- [x] Comparison: `data.py`, tests of effort grouping (preserve release names),
  all three selector modes, metric choice, null/zero values and Pareto frontier.
- [x] Plot: `plot.py`, tests of connected effort order, colors/provider markers,
  log/linear coordinates, automatic label geometry and exported artifacts.
  Use display-space candidate placement scored against labels, points and
  line segments, with leader lines and an expanded layout for dense inputs.
- [x] CLI: `cli.py`, tests of help, JSON-only stdout, exit codes, ephemeral
  artifacts, explicit files, dotenv loading and realistic help-only workflow.
- [x] Verify package installation, inspect rendered representative plots,
  review changes and run full suite. Release via commit and push to origin.

For each unit write behavior tests, run to demonstrate missing behavior, implement,
then rerun green. The reference conversation specifies a compact cream-background
plot with a bottom legend and small effort annotations. Keep this look while
replacing hand-tuned label offsets with automatic geometry.

Sources inspected 2026-09-09:
- https://artificialanalysis.ai/data-api/docs (current API schema)
- https://artificialanalysis.ai/methodology (cost definition)
- User's shared conversation (reference plotting code, recovered via direct GET)

Live validation needs an AA API key; tests use explicitly synthetic API-shaped
fixtures. Never present fixture scores as current model evaluations.


Verification completed:
- 72 tests pass, including HTTP-boundary fixtures, all selector modes,
  three metrics, log/linear exports, dense label and legend/footer geometry.
- Ruff passes; wheel and sdist build. A clean temporary wheel installation
  prints the exact help and renders a plot using the synthetic HTTP harness.
- A separate agent completed discovery, a two-family Coding plot and exact
  variant JSON extraction using only CLI help.
- Independent review identified the crowded legend/footer overlap; a failing
  regression test reproduced it before the measured-layout fix.
- Official endpoint reachability verified: unauthenticated HTTP 401.
  Authenticated live validation was not performed because no AA key was supplied.
