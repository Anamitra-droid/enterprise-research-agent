---
id: security-pci-policy
title: Information Security and PCI DSS Policy
classification: internal
department: security
last_updated: 2026-02-15
---
# Information Security and PCI DSS Policy

## PCI DSS compliance
LedgerLeaf is a PCI DSS v4.0.1 Level 1 Service Provider. Compliance is validated by an annual on-site assessment by a Qualified Security Assessor (QSA) and quarterly external ASV vulnerability scans.

## Encryption
Cardholder data is encrypted at rest with AES-256 and in transit with TLS 1.2 or higher. Encryption keys are stored in an HSM and rotated every 90 days.

## Access control
Multi-factor authentication (MFA) is mandatory for all employees and contractors. Access to production systems is reviewed quarterly and follows least privilege.

## Testing
External penetration tests are performed twice a year and after any significant infrastructure change.
