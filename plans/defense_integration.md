# Speculative Safety-Aware Decoding (SSD) Integration Plan

## Core Implementation

### Module 1: `src/6_defense_ssd.py`
```python
class SpeculativeSafetyDecoder:
    def __init__(self, llamagen_model, safety_threshold=0.7, beta=0.5)
    def classify_token_safety(self, token, context) -> float
    def acceptance_mechanism(self, draft_tokens, safety_scores) -> List[bool]
    def defended_generate(self, prompt, **kwargs) -> Image
```

**Key Functions:**
- Token-level CLIP-based safety classification
- ²-threshold acceptance/rejection mechanism
- Integration hook into `LlamaGenInference.generate_image()`

### Module 2: `src/7_defense_evaluation.py`
```python
class DefenseEvaluator:
    def __init__(self, defended_model, undefended_model)
    def compare_attack_success_rates(self, attack_results_defended, attack_results_undefended)
    def measure_defense_overhead(self, prompts) -> Dict[str, float]
    def statistical_significance_test(self, defended_scores, undefended_scores)
```

**Key Functions:**
- ASR comparison (defended vs undefended)
- Latency/quality degradation measurement
- Statistical significance analysis

### Module 3: `src/8_multi_objective_defense.py`
```python
class DefenseAwareObjective(AttackObjective):
    def compute_defense_bypass_score(self, image, prompt) -> float
    def measure_quality_preservation(self, defended_image, original_image) -> float
    def calculate_defense_robustness(self, attack_trajectory) -> float
```

**Enhanced Objectives:**
- Safety bypass detection using CLIP embeddings
- Image quality preservation (PSNR/SSIM)
- Defense robustness scoring

## Integration Points

### 1. LlamaGenInference Extension
```python
# Add to src/1_inference.py
class LlamaGenInference:
    def enable_ssd_defense(self, safety_threshold=0.7, beta=0.5):
        self.ssd_decoder = SpeculativeSafetyDecoder(self, safety_threshold, beta)

    def generate_image(self, prompt, use_defense=False):
        if use_defense and hasattr(self, 'ssd_decoder'):
            return self.ssd_decoder.defended_generate(prompt)
        # existing generation logic
```

### 2. Attack Pipeline Modification
```python
# Modify src/2_blackbox_attacks.py and src/3_whitebox_attacks.py
def evaluate_prompt(self, prompt, target_text, use_defense=False):
    images = self.image_generator.generate_image(prompt, use_defense=use_defense)
    # existing evaluation logic
```

### 3. Evaluation Framework Extension
```python
# Add to src/5_evaluation.py
def run_defense_comparison(self, attack_results_dir):
    # Run attacks with and without defense
    # Compare ASR, quality, latency
    # Generate comparative plots
```

## Implementation Priority

**Phase 1:** Core SSD mechanism in `6_defense_ssd.py`
**Phase 2:** Defense evaluation in `7_defense_evaluation.py`
**Phase 3:** Enhanced objectives in `8_multi_objective_defense.py`

## Expected Outputs

- `results/defense_evaluation/` - Comparative analysis
- `logs/defense_comparison.json` - Metrics comparison
- `plots/defense_effectiveness.png` - ASR comparison plots