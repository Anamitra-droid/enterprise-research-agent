---
id: api-guide
title: LedgerLeaf API Developer Guide
classification: public
department: engineering
last_updated: 2026-05-15
---
# LedgerLeaf API Developer Guide

## Authentication
All API requests are authenticated with secret API keys sent in the Authorization header. Test-mode keys start with `ll_test_` and live keys start with `ll_live_`.

## Rate limits
The API rate limit is 100 requests per second per live API key, with bursts of up to 200 requests. In test mode the rate limit is 25 requests per second. Requests over the limit receive HTTP 429 with a Retry-After header.

## Idempotency
POST requests accept an Idempotency-Key header. Idempotency keys are stored for 24 hours.

## Webhooks
Webhook deliveries that fail are retried with exponential backoff for up to 3 days (16 attempts). Each webhook is signed with an HMAC-SHA256 signature in the LedgerLeaf-Signature header.

## Versioning
The API is versioned by date. Breaking changes are announced at least 6 months in advance.
