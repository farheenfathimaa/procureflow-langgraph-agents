# Evaluation results

Generated: **2026-11-05 04:44 UTC**

These numbers were measured by running `make eval` against the bundled
27-question procurement set with the real RAGAS harness, local embedding and
cross-encoder models, and a Groq free-tier judge. Metrics that failed or
returned no finite values are explicitly marked **not measured**.

## Run configuration

| Setting | Value |
| --- | --- |
| Questions evaluated | 27 |
| Judge calls completed | 70 / 108 |
| Retrieval | hybrid BM25 + pgvector-compatible cosine (in-memory index), RRF, cross-encoder |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Top-k contexts | 4 |
| Judge LLM | `openai/gpt-oss-120b` (Groq free tier) |
| Metrics with finite values | 4 / 4 |

## RAGAS metrics

| Metric | Score | Samples scored |
| --- | ---: | ---: |
| Faithfulness | 0.8917 | not recorded (mean over completed judge calls) |
| Answer relevancy | 0.8642 | not recorded (mean over completed judge calls) |
| Context precision | 0.9875 | not recorded (mean over completed judge calls) |
| Context recall | 0.9545 | not recorded (mean over completed judge calls) |

Each score is the mean of the finite per-sample scores RAGAS returned. On the
free tier, individual judge calls can be rejected by rate limits; 38 of 108
calls hit the Groq tokens-per-minute/tokens-per-day limits in this run. Failed
calls were never replaced with invented values. The harness now records
per-metric `Samples scored` coverage on subsequent runs.

## Notes

- Answers were generated from retrieved contexts with citations; the judge was
  a separate free-tier Groq model invocation, not the application's answer model.
- Scores on a 20-document synthetic corpus are a smoke test of the pipeline, not
  a claim of production retrieval quality.
- The evaluation intentionally uses the same hybrid retrieval implementation as
  the API, with an in-memory store so it can run without Docker/Postgres.
