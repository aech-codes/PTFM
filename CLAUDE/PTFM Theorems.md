# PTFM Formal Properties and Theoretical Guarantees

## Overview

This document presents the theoretical foundations of the **Probably Truthful Foundation Model (PTFM)**. While `02_ptfm_math_spec.md` defines the operational mathematics used by the implementation, this document provides the formal properties that justify the architecture and explain the behavior of the Verifiable Truth Space (VTS).

These propositions assume:

- Continuous calibrated confidence functions.
- Sound symbolic verification.
- Exchangeability assumptions required by conformal prediction.
- Correct implementation of the hybrid retrieval pipeline.

---

# Mathematical Notation

Let

- \(S\) denote the semantic space of all atomic claims.
- \(I\) denote the retrieved evidence set.
- \(VTS(I,\tau)\subseteq S\) denote the Verifiable Truth Space.
- \(\hat C(c,I)\) denote the calibrated confidence of claim \(c\).
- \(P(c,I)\) denote symbolic verification.
- \(\tau\) denote the verification threshold.

The Verifiable Truth Space is

\[
VTS(I,\tau)
=
\{c\in S:
\hat C(c,I)\ge\tau
\land
P(c,I)=1
\}.
\]

---

# Proposition 1 — Closure of the Verifiable Truth Space

## Statement

Assume

\[
\hat C:S\times I\rightarrow[0,1]
\]

is continuous with respect to the semantic metric

\[
d_{sem}.
\]

Then

\[
VTS(I,\tau)
\]

is a closed subset of

\[
(S,d_{sem}).
\]

---

## Intuition

Small semantic changes to a claim cannot cause arbitrary jumps in verification confidence.

Therefore nearby claims exhibit similar verification behavior.

---

## Proof Sketch

The set

\[
\{c:\hat C(c,I)\ge\tau\}
\]

is the inverse image of the closed interval

\[
[\tau,1]
\]

under a continuous function.

Since symbolic verification defines another closed constraint,

their intersection remains closed.

---

## Practical Implication

The verification engine behaves smoothly under small perturbations of user queries or generated claims.

---

# Proposition 2 — Monotonicity with Respect to Evidence

## Statement

Let

\[
I\subseteq I'.
\]

Assume retrieval only adds valid evidence and never removes existing evidence.

Then

\[
VTS(I,\tau)
\subseteq
VTS(I',\tau).
\]

---

## Interpretation

Adding trustworthy evidence cannot reduce the number of verifiable claims.

The truth space may expand but should not shrink.

---

## Proof Sketch

Every claim verified using

\[
I
\]

remains verifiable under

\[
I'
\]

because all previous supporting evidence remains available.

Additional evidence may verify previously unsupported claims.

---

## Consequence

Improving retrieval quality can only increase verification coverage.

---

# Proposition 3 — Hallucination Decomposition

## Statement

For a generated response

\[
x,
\]

the hallucination probability satisfies

\[
P(H|x)
=
P(f\notin VTS|x)
+
P(f\in VTS,\text{incorrect}|x).
\]

---

## Interpretation

Hallucinations arise from two sources:

1. Claims outside the Verifiable Truth Space.

2. Incorrect claims that nevertheless entered the VTS due to imperfect verification.

---

## Consequence

Reducing hallucination requires improving

- retrieval quality,
- verification accuracy,
- calibration quality.

Improving only one component cannot eliminate hallucinations completely.

---

# Proposition 4 — Finite Evidence Limitation

## Statement

Assume retrieval operates over a finite evidence collection

\[
I.
\]

Then there exist true claims

\[
c
\]

such that

\[
c
\notin
VTS(I,\tau).
\]

---

## Interpretation

Finite evidence implies incomplete verification coverage.

Some true claims remain unverifiable.

---

## Practical Consequence

A trustworthy system must sometimes refuse to answer instead of generating unsupported responses.

---

# Proposition 5 — Soundness

## Statement

Assume

- symbolic verification is sound,
- calibration is valid,
- retrieved evidence is correct.

Then every accepted claim satisfies

\[
P(c\text{ is true})
\ge
1-\alpha.
\]

---

## Interpretation

PTFM guarantees bounded verification risk under the assumptions of the calibration framework.

---

# Proposition 6 — Retrieval Consistency

Suppose

\[
r(E_1)
<
r(E_2).
\]

Then

\[
E[V(c,E_1)]
\le
E[V(c,E_2)].
\]

Higher-quality retrieval is expected to increase verification confidence.

---

# Proposition 7 — Entropy Monotonicity

Suppose semantic entropy satisfies

\[
S_1<S_2.
\]

Then

\[
P(H|S_1)
<
P(H|S_2).
\]

Higher semantic entropy corresponds to greater hallucination risk.

---

# Proposition 8 — Confidence Monotonicity

Let

\[
\hat C_1,\ldots,\hat C_m
\]

be calibrated claim confidences.

The response confidence is

\[
C_{response}
=
\left(
\prod_i
\hat C_i
\right)^{1/m}.
\]

If any

\[
\hat C_i
\]

decreases while all others remain fixed,

then

\[
C_{response}
\]

cannot increase.

---

## Consequence

Every claim contributes to the overall confidence.

A single weak claim lowers trust in the complete response.

---

# Proposition 9 — Decision Consistency

Let

\[
\tau_{low}
<
\tau.
\]

The decision function

\[
D(C_{response})
\]

is

\[
Generate,
\quad
Retry,
\quad
Refuse.
\]

These decision regions are mutually exclusive and collectively exhaustive.

Therefore every response belongs to exactly one decision state.

---

# Hallucination Risk Model

Define

- hallucination indicator

\[
y_{hall}
\]

- generation indicator

\[
\mathbf1_{generate}
\]

- refusal indicator

\[
\mathbf1_{refuse}.
\]

The Hallucination Risk Model (HRM) loss is

\[
\ell_H
=
y_{hall}
\cdot
conf
\cdot
\mathbf1_{generate}
+
(1-y_{hall})
\cdot
c_{abs}
\cdot
\mathbf1_{refuse}.
\]

This objective penalizes

- confident hallucinations,
- unnecessary refusals,

while encouraging calibrated decisions.

---

# Decision Guarantees

The decision engine follows

### Generate

\[
C_{response}
\ge
\max(\tau,\hat q).
\]

---

### Retry

\[
\tau_{low}
\le
C_{response}
<
\tau.
\]

---

### Refuse

\[
C_{response}
<
\tau_{low}
\]

or

retrieval budget exhausted.

---

# Research Contributions

The theoretical framework establishes:

- A formal definition of the Verifiable Truth Space.
- Topological stability of verification under continuity assumptions.
- Monotonicity of verification with increasing evidence.
- A decomposition of hallucination sources.
- A principled justification for calibrated refusal.
- Statistical guarantees through conformal prediction.
- A response confidence function based on the geometric mean of calibrated claim confidences.

---

# Future Extensions

The current theory can be extended to include:

- Formal completeness bounds.
- Graph-based verification guarantees.
- Bayesian uncertainty propagation.
- PAC-style verification guarantees.
- Information-theoretic retrieval bounds.
- Asymptotic convergence of VTS under expanding evidence.