# Speculative Decoding Defense Literature Survey

## Six-Axis Comparative Analysis for Adversarial Defense Integration

| Paper | Space & Distribution Preservation | Acceptance/Switching Statistic | Residual/Modified Distribution | Safety Enhancement Extras | Theoretical Guarantees | Application Domain |
|-------|----------------------------------|-------------------------------|--------------------------------|------------------------|----------------------|-------------------|
| **[Accelerating Auto-regressive Text-to-Image Generation with Training-free Speculative Jacobi Decoding](https://arxiv.org/abs/2410.01699) [SJD]** | Discrete tokens; preserves AR semantics under stochastic decoding | Tokenwise ratio between Jacobi iterations | Calibrated residual from probability vector diffs | Probabilistic Jacobi + spatial-locality init | Rationale for stochastic AR consistency | AR text-to-image with random sampling |
| **[Continuous Speculative Decoding for Autoregressive Image Generation](https://arxiv.org/abs/2411.11925) [Continuous SD]** | Continuous densities; proves exact preservation of $p$ | Accept if $p/q > 1$ | $p'(x)=\max(0,p-q)$ | Trajectory alignment + accept-reject proofs | Formal correctness $P_{out}=p$ | Diffusion-based AR continuous models |
| **[Speculative Safety-Aware Decoding](https://arxiv.org/abs/2508.17739) [SSD]** | Trades strict preservation for controllable safety/utility trade-off | Match-ratio $\beta_i$ over bins decides Intersection vs Union | Restricted token set (Union/Intersection) + composite distribution | Token-set algebra + annealed thresholds | Safe/utility switching, no invariance guarantee | Safety-aware LLMs |
| **[Root Defence Strategies: Ensuring Safety of LLM at the Decoding Level](https://arxiv.org/abs/2410.06809) [Root Defence]** | Directly modifies token logits → prioritizes safety over distribution invariance | No ratio—uses classifier scores to select safe token | No residual; direct scoring and selection | Token-wise classifier with PCA and hidden-state projections | Empirical token-level correction; no lossless proof | Decoding-level root safety with speculative acceleration |
| **[MASSV: Multimodal Adaptation and Self-Data Distillation for Speculative Decoding of Vision-Language Models](https://arxiv.org/abs/2505.10526) [MASSV]** | Same SD math, but adapted drafter raises acceptance | Standard SD acceptance, enhanced via alignment | Unchanged; residual used less due to alignment | Multimodal projector + self-distillation | Empirical speedups; no new proof | Multimodal VLM speculative decoding |

## Core Mathematical Formulations & Defense Integration Potential

| Paper | Core Mathematical Formula | Defense Integration Potential |
|-------|---------------------------|------------------------------|
| **SJD** | $$\min\left(1, \frac{p_\theta(x_i^{(j)}\|x_{1:i-1}^{(j)})}{p_\theta(x_i^{(j)}\|x_{1:i-1}^{(j-1)})}\right)$$ | Adversarial detection during probabilistic acceptance phase; training-free implementation for AR T2I |
| **Continuous SD** | $$\frac{p(x_0\|x_T)}{q(x_0\|x_T)} = \frac{p(x_T) \prod_t p(x_{t-1}\|x_t)}{q(x_T) \prod_t q(x_{t-1}\|x_t)}$$ | Continuous-space adversarial detection using density ratio verification with formal correctness guarantees |
| **SSD** | $$\beta_i = \frac{1}{b}\sum_{n=1}^b \mathbf{1}\{\text{draft from } m \text{ accepted}\}$$ | Direct adversarial defense with real-time risk assessment and dynamic safety/utility switching |
| **Root Defence** | $$x_i = [x_{i-1}; \max(C_i)], \quad C_i = f(I_i, x_{i-1})$$ | Proactive harmful content correction via token-level safety classification with PCA projections |
| **MASSV** | $$\mathcal{L}_{proj} + \mathcal{L}_{distill} \text{ (multimodal alignment + self-distillation)}$$ | Foundation for VLM adversarial detection through enhanced visual-text token alignment |


