"""Context sovereignty: consent, egress control, audit, encryption, injection gating.

These are the parts of CI whose correctness can be *guaranteed* by code rather than
hoped for from a model, so they are written to be testable in isolation:

- crypto.py     per-record keys; deleting a record destroys its key (crypto-shredding)
- consent.py    grants per (source, purpose, provider); revocation is immediate
- egress.py     the only path by which personal context reaches a model provider
- audit.py      hash-chained, append-only log of every egress decision and deletion
- injection.py  untrusted text can be stored as data but cannot become an instruction
"""
