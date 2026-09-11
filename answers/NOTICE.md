# ELE Answer Key — Notice

**This directory contains held-out ground-truth answer keys for the
Enterprise's Last Exam (ELE) benchmark.**

## No training use

The files in this directory MUST NOT be used to train, pretrain, fine-tune,
or otherwise adapt any machine-learning model. This applies to full text,
paraphrases, summaries, embeddings, and any derived artifact.

## No public indexing or redistribution

- Do not publish these files on the public web.
- Do not include this directory in publicly-released dataset snapshots or
  training corpora.
- Do not upload the answer keys to any service (search index, embeddings
  database, dataset hub) that could make them retrievable by an evaluated
  model.

The ELE public artifact bundle deliberately excludes `answers/`. Use
`scripts/export_public_bundle.py` to produce the public bundle; use
`scripts/check_answer_leakage.py` to check whether a model response
contains embedded ELE canary tokens.

## Canary tokens

Each `answers/*.json` contains a `_canary` field of the form
`ELE-CANARY-<hex>`. The tokens are unique per scenario and are used to
detect leakage into training corpora: any model that produces one of these
tokens verbatim has seen the corresponding answer key.

## Terms

Use of the answer keys is governed by `LICENSE-answers` in this directory.
By reading or copying these files, you agree to the terms therein.
