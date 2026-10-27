# PTFM Architectural Anti-Patterns

The following design choices are prohibited within the PTFM Core.

---

## 1. Hardcoded Domain Knowledge

Never place legal, medical, financial, or other domain-specific logic inside `ptfm/core`.

All domain logic must reside inside plugins.

---

## 2. Raw Neural Confidence

Never use raw LLM or NLI probabilities directly.

All confidence values must pass through the conformal calibration layer.

---

## 3. Full-Paragraph Verification

Never verify entire paragraphs.

Always:

Response
→ Atomic Claims
→ Individual Verification

---

## 4. Sequential Hybrid Retrieval

Dense, sparse, and graph retrieval must execute concurrently using asynchronous execution.

---

## 5. Hardcoded Thresholds

Never write rules such as:

```python
if score > 0.8:
```

All thresholds must originate from the calibration layer.

---

## 6. Plugin Coupling

The PTFM Core must never import domain-specific plugins directly.

Only abstract interfaces may be referenced.

---

## 7. Blocking I/O

Never perform synchronous database or network operations inside async functions.

---

## 8. Global Mutable State

Avoid global variables and mutable singleton objects.

Use dependency injection and immutable configuration.

---

## 9. Duplicate Verification Logic

Verification algorithms must exist in only one location.

Plugins may extend behavior but not duplicate the core verification pipeline.

---

## 10. Missing Type Safety

All public functions and schemas must include explicit type hints and Pydantic validation.

---

## 11. Missing Error Handling

Retrieval, verification, calibration, and plugin execution must raise explicit PTFM exceptions rather than generic exceptions.

---

## 12. Evidence-Free Generation

The Decision Engine must never generate a response unless all accepted claims belong to the Verifiable Truth Space (VTS).
