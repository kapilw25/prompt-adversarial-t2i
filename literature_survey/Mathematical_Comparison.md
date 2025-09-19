# Mathematical Comparison of Speculative Decoding Papers

---

## Section 1: What Each Paper is Mathematically Doing

**1. Accelerating AR T2I with Training-free Speculative Jacobi Decoding (SJD)**

* Introduces a *probabilistic Jacobi accept rule* for autoregressive T2I.
* Accept draft token $x_i^{(j)}$ with probability:

  $$
  \min\left(1, \frac{p_\theta(x_i^{(j)}|x_{1:i-1}^{(j)})}{p_\theta(x_i^{(j)}|x_{1:i-1}^{(j-1)})}\right)
  $$
* If rejected, resample from a *calibrated residual distribution* and restart from rejection point.
* Fixes deterministic Jacobi limitation for stochastic/top-K sampling in T2I.

**2. Continuous Speculative Decoding (for AR Image Generation)**

* Works on *continuous densities* (diffusion tokenization).
* Accept sample $x$ if $p(x)/q(x) > 1$; else reject with prob $1-p/q$, resample from:

  $$
  p'(x) = \text{norm}\,\max(0, p(x) - q(x))
  $$
* Factorization along reverse diffusion chain:

  $$
  \frac{p(x_0|x_T)}{q(x_0|x_T)} = \frac{p(x_T) \prod_t p(x_{t-1}|x_t)}{q(x_T) \prod_t q(x_{t-1}|x_t)}
  $$
* Adds *trajectory alignment* using reparameterization trick with shared Gaussian noise.

**3. MASSV (Speculative Decoding for VLMs)**

* Extends standard SD math $\min(1, p/q)$ acceptance.
* Key contribution: makes drafter multimodal via **vision encoder + trainable projector**.
* Uses **self-distilled visual instruction tuning** to align $q$ with $p$, improving acceptance rate.

**4. Speculative Safety-Aware Decoding (SSD)**

* Maintains two distributions per step: utility $P_M$ and safety $P_m$.
* Chooses between **Intersection** (utility) or **Union** (safety) token sets.
* Composite distribution: $P_M + \alpha(P_m - P_M)$.
* Uses *match-ratio* statistic:

  $$
  \beta_i = \frac{1}{b}\sum_{n=1}^b \mathbf{1}\{\text{draft from } m \text{ accepted}\}
  $$
* Switches between modes based on $\beta_i$.

**5. Root Defence Strategies (RDS)**

* Formulates per-step token scoring with classifier $f$ on hidden states + candidate token.
* Equation:

  $$
  x_i = [x_{i-1}; \max(C_i)], \quad C_i = f(I_i, x_{i-1})
  $$
* Score via PCA + linear classifier:

  $$
  m_k = V^T(h_i^k - u), \quad c_k = W^T m_k + b
  $$
* Prioritizes safe tokens step-by-step; integrates speculative decoding for efficiency.

---

## Section 2: Same Idea, Contrasted on 6 Axes

**1. Space & Distribution Preservation**

* SJD: discrete tokens; preserves AR semantics under stochastic decoding.
* Continuous SD: continuous densities; proves exact preservation of $p$.
* MASSV: same SD math, but adapted drafter raises acceptance.
* SSD: trades strict preservation for controllable *safety/utility trade-off*.
* RDS: directly modifies token logits → prioritizes safety over distribution invariance.

**2. Acceptance / Switching Statistic**

* SJD: tokenwise ratio between Jacobi iterations.
* Continuous SD: accept if $p/q > 1$.
* MASSV: standard SD acceptance, enhanced via alignment.
* SSD: *match-ratio* $\beta_i$ over bins decides Intersection vs Union.
* RDS: no ratio—uses classifier scores to select safe token.

**3. Residual / Modified Distribution**

* SJD: calibrated residual from probability vector diffs.
* Continuous SD: $p'(x)=\max(0,p-q)$.
* MASSV: unchanged; residual used less due to alignment.
* SSD: restricted token set (Union/Intersection) + composite distribution.
* RDS: no residual; direct scoring and selection.

**4. Extras to Improve Acceptance/Safety**

* SJD: probabilistic Jacobi; spatial-locality init.
* Continuous SD: trajectory alignment + accept-reject proofs.
* MASSV: multimodal projector + self-distillation.
* SSD: token-set algebra + annealed thresholds.
* RDS: token-wise classifier with PCA and hidden-state projections.

**5. Guarantees / Proofs**

* SJD: rationale for stochastic AR consistency.
* Continuous SD: formal correctness $P_{out}=p$.
* MASSV: empirical speedups; no new proof.
* SSD: safe/utility switching, no invariance guarantee.
* RDS: empirical token-level correction; no lossless proof.

**6. Application Domain**

* SJD: AR text-to-image with random sampling.
* Continuous SD: diffusion-based AR continuous models.
* MASSV: multimodal VLM speculative decoding.
* SSD: safety-aware LLMs.
* RDS: decoding-level root safety with speculative acceleration.

---

## Section 3: TL;DR (Math Lens)

* **Continuous SD** → Continuous tokens; accept if $p/q > 1$; correctness proven; trajectory alignment.
* **SJD** → Multi-token stochastic AR; probabilistic Jacobi accept rule; residual resampling.
* **MASSV** → VLMs; adds multimodal projector + self-distilled tuning; raises acceptance length.
* **SSD** → Safety decoding; Intersection/Union token sets; composite distribution; match-ratio switching.
* **RDS** → Root-level safety; token-wise classifier over hidden states; speculative decoding for speed.

---
