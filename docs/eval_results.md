# Evaluation results

Generated: **2026-11-12 18:22 UTC**

These numbers were measured by running `make eval` against the bundled
27-question procurement set. Metrics that failed or returned no finite
values are explicitly marked **not measured**.

## Run configuration

| Setting | Value |
| --- | --- |
| Questions evaluated | 27 |
| Retrieval | hybrid BM25 + pgvector-compatible cosine (in-memory index), RRF, cross-encoder |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Top-k contexts | 4 |
| Judge LLM | `openai/gpt-oss-120b` (Groq free tier) |
| Metrics with finite values | 4 / 4 |

## RAGAS metrics

| Metric | Score | Samples scored |
| --- | ---: | ---: |
| Faithfulness | 0.9368 | 19/27 |
| Answer relevancy | 0.8712 | 20/27 |
| Context precision | 0.9815 | 18/27 |
| Context recall | 1.0000 | 8/27 |

## Notes

- Answers were generated from retrieved contexts with citations; the judge was
  a separate free-tier Groq model invocation, not the application's answer model.
- Each score is the mean of the finite per-sample scores RAGAS returned. On the
  free tier, individual judge calls can be rejected by rate limits; the
  `Samples scored` column records how many of the questions contributed. Failed
  calls are never replaced with invented values.
- Scores on a 20-document synthetic corpus are a smoke test of the pipeline, not
  a claim of production retrieval quality.
- The evaluation intentionally uses the same hybrid retrieval implementation as
  the API, with an in-memory store so it can run without Docker/Postgres.
