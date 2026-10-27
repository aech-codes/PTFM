# PTFM Mathematical & Statistical Specification

## Overview

This document defines the mathematical foundations of the **Probably Truthful Foundation Model (PTFM)**. PTFM combines hybrid retrieval, neural verification, symbolic reasoning, semantic uncertainty estimation, and conformal prediction to construct a **Verifiable Truth Space (VTS)** with formal statistical guarantees.

The formulations in this document are **domain-agnostic** and apply to every PTFM plugin, including Legal, Finance, Healthcare, Scientific Literature, and Enterprise Knowledge Systems.

---

# Mathematical Notation

| Symbol | Description |
|---------|-------------|
| \(q\) | User query |
| \(R\) | Foundation model response |
| \(C=\{c_1,\dots,c_m\}\) | Set of atomic claims extracted from \(R\) |
| \(c_i\) | Individual atomic claim |
| \(E_D\) | Evidence retrieved from domain \(D\) |
| \(e_j\) | Individual evidence item |
| \(V(c,E)\) | Calibrated verification confidence |
| \(\hat C(c)\) | Calibrated confidence of claim \(c\) |
| \(P(c,E)\) | Symbolic verification result |
| \(S(q)\) | Semantic entropy |
| \(r(E)\) | Retrieval reliability |
| \(\alpha\) | Target conformal error rate |
| \(\hat q\) | Conformal threshold |
| \(\tau\) | User-defined confidence threshold |
| \(\tau_{low}\) | Retry threshold |
| \(C_{response}\) | Final response confidence |
| \(\Gamma(E)\) | Verifiable Truth Space |

---

# 1. Atomic Claim Extraction

Given a generated response

\[
R=\{s_1,s_2,\ldots,s_n\}
\]

an extraction function

\[
f:R\rightarrow C
\]

produces

\[
C=\{c_1,c_2,\ldots,c_m\}
\]

where each claim satisfies

- atomic
- independently verifiable
- context preserving
- semantically complete

Each claim is verified independently.

---

# 2. Verifiable Truth Space (VTS)

Let

- \(C\) denote the extracted claim set.
- \(E_D\) denote retrieved evidence.

The **Verifiable Truth Space** is

\[
\Gamma(E_D)=
\left\{
c\in C
\mid
V(c,E_D)\ge\tau
\land
P(c,E_D)=1
\right\}
\]

where

\[
V(c,E_D)\in[0,1]
\]

is the calibrated verification confidence and

\[
P(c,E_D)\in\{0,1\}
\]

is the symbolic verification indicator.

Only claims inside

\[
\Gamma(E_D)
\]

are eligible for inclusion in the final response.

---

# 3. Retrieval Reliability

Suppose retrieval returns

\[
E=\{e_1,\ldots,e_k\}
\]

Each evidence item has

- retrieval relevance

\[
r_i
\]

and

- source credibility

\[
s_i
\]

The retrieval reliability is

\[
r(E)
=
\frac1k
\sum_{i=1}^{k}
r_i s_i
\]

subject to

\[
0\le r(E)\le1.
\]

---

# 4. Neural Claim Verification

For each claim

\[
c_i
\]

an NLI model predicts

- entailment

\[
p_e
\]

- contradiction

\[
p_c
\]

- neutral

\[
p_n
\]

with

\[
p_e+p_c+p_n=1.
\]

The raw verification score is

\[
V_{raw}=p_e-p_c.
\]

Raw neural scores are **never** passed directly to the decision engine.

Instead,

\[
V_{cal}=g(V_{raw})
\]

where

\[
g(\cdot)
\]

is the conformal calibration function.

---

# 5. Symbolic Verification

Symbolic verification is

\[
P(c,E)
=
\begin{cases}
1,&\text{claim satisfies all symbolic constraints}\\
0,&\text{otherwise}
\end{cases}
\]

Examples include

- ontology rules
- knowledge graph constraints
- deterministic domain rules
- temporal consistency
- numerical consistency

A claim must satisfy

\[
P(c,E)=1
\]

before entering the Verifiable Truth Space.

---

# 6. Semantic Entropy

Generate

\[
N
\]

independent responses

\[
x_1,\ldots,x_N.
\]

Cluster semantically equivalent outputs into

\[
C_1,\ldots,C_K.
\]

Semantic entropy is

\[
S(q)
=
-
\sum_{k=1}^{K}
p(C_k|q)
\log p(C_k|q)
\]

where

\[
p(C_k|q)
\]

is the probability of semantic cluster

\[
C_k.
\]

Higher entropy indicates greater uncertainty.

---

# 7. Nonconformity Score

Each verified claim is assigned a nonconformity score

\[
\alpha(c)
=
1-\hat C(c)
\]

where

\[
\hat C(c)
\]

is the calibrated confidence.

Properties

- Lower score → higher conformity
- Higher score → greater hallucination risk

The calibration set is

\[
\{\alpha_1,\alpha_2,\ldots,\alpha_n\}.
\]

---

# 8. Conformal Prediction

Given calibration scores

\[
\alpha_1,\ldots,\alpha_n
\]

the conformal threshold is

\[
\hat q
=
Q
\left(
\frac{\lceil(n+1)(1-\alpha)\rceil}{n}
\right)
\]

where

- \(Q(\cdot)\) is the empirical quantile,
- \(\alpha\) is the desired error rate.

Coverage guarantee:

\[
P
(
c\in\Gamma(E)
\mid
c\text{ is true}
)
\ge
1-\alpha.
\]

---

# 9. Response Confidence Aggregation

Suppose

\[
\hat C_1,\hat C_2,\ldots,\hat C_m
\]

are calibrated confidence scores of all verified claims.

The response confidence is

\[
C_{response}
=
\left(
\prod_{i=1}^{m}
\hat C_i
\right)^{1/m}
\]

This geometric mean prevents highly confident claims from masking low-confidence or unverifiable claims.

Properties

- Penalizes any weak claim.
- Rewards uniformly verified responses.
- Robust against isolated hallucinations.

---

# 10. Hallucination Risk

Hallucination risk is estimated using

- semantic entropy
- retrieval reliability
- calibrated verification confidence

One possible operational estimate is

\[
H
=
S(q)
\left(
1-r(E)
\right).
\]

subject to

\[
0\le H\le1.
\]

Higher values indicate greater uncertainty.

---

# 11. Decision Function

The decision engine computes

\[
C_{response}
=
\left(
\prod_i
\hat C_i
\right)^{1/m}
\]

The decision policy is

### Generate

\[
C_{response}
\ge
\max(\tau,\hat q)
\]

All claims belong to the Verifiable Truth Space.

---

### Retry

\[
\tau_{low}
\le
C_{response}
<
\tau
\]

A second retrieval round is executed using

- expanded search
- query decomposition
- claim-targeted retrieval

Retry is bounded by

\[
retry_{count}<N_{max}.
\]

---

### Refuse

\[
C_{response}
<
\tau_{low}
\]

or

\[
retry_{count}
\ge
N_{max}.
\]

The refusal response includes diagnostics explaining which claims could not be verified.

---

# 12. Mathematical Pipeline

The verification pipeline is defined as

\[
q
\rightarrow
Retrieval
\rightarrow
Evidence
\rightarrow
Claim\ Extraction
\rightarrow
Verification
\rightarrow
Calibration
\rightarrow
Nonconformity
\rightarrow
Conformal\ Threshold
\rightarrow
Response\ Confidence
\rightarrow
Decision.
\]

---

# 13. Optimization Objective

PTFM minimizes

\[
\mathcal L
=
\lambda_1L_{verify}
+
\lambda_2L_{entropy}
+
\lambda_3L_{retrieval}
+
\lambda_4L_{calibration}
\]

subject to

\[
\lambda_i\ge0.
\]

where

- \(L_{verify}\) minimizes claim verification error.
- \(L_{entropy}\) penalizes semantic uncertainty.
- \(L_{retrieval}\) improves evidence quality.
- \(L_{calibration}\) minimizes calibration error.

---

# 14. Statistical Guarantees

PTFM provides the following guarantees:

1. **Evidence-backed verification** through hybrid retrieval.

2. **Deterministic validation** through symbolic reasoning.

3. **Distribution-free calibration** using conformal prediction.

4. **Finite-sample coverage**

\[
P(c\in\Gamma(E))
\ge
1-\alpha.
\]

5. **Domain independence**, as all mathematical formulations are independent of domain-specific knowledge.

---

# Summary

| Component | Mathematical Model |
|------------|--------------------|
| Claim Extraction | Atomic decomposition |
| Retrieval | Hybrid relevance scoring |
| Verification | Neural NLI + Symbolic reasoning |
| Uncertainty | Semantic Entropy |
| Calibration | Conformal Prediction |
| Nonconformity | \(1-\hat C\) |
| Response Confidence | Geometric Mean |
| Decision | Threshold-gated policy |
| Trust Space | Verifiable Truth Space |
| Statistical Guarantee | Finite-sample conformal coverage |

---

## Relationship to Formal Theorems

This specification defines the **operational mathematics** of the PTFM framework.

Formal properties such as

- VTS Closure
- VTS Monotonicity
- VTS–Hallucination Duality
- VTS Impossibility
- Hallucination Risk Model (HRM)
- Formal Proof Sketches

are specified separately in **`03_ptfm_theorems.md`**, ensuring a clear separation between implementation mathematics and theoretical guarantees.