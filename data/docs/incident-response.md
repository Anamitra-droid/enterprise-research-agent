---
id: incident-response
title: Incident Response Runbook
classification: internal
department: engineering
last_updated: 2026-04-02
---
# Incident Response Runbook

## Severity levels
- **SEV1**: payments are unavailable or data is at risk. The on-call engineer must be paged within 5 minutes, the status page updated within 15 minutes and executives notified within 30 minutes.
- **SEV2**: significant degradation for a subset of merchants. Status page update within 30 minutes.
- **SEV3**: minor issue with a workaround.

## Post-incident
A blameless postmortem is required for every SEV1 and SEV2 incident within 5 business days.

## Regulatory notification
If personal data is affected, Legal must be informed immediately so that regulators can be notified within 72 hours.
