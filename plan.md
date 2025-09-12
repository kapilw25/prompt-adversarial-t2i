We can start with models
1) LlamaGen >>
https://github.com/FoundationVision/LlamaGen
2) Anole-7b-v0.1 >>
https://github.com/GAIR-NLP/anole

Here is our 5 step plan
step1) Setup Inference: Load checkpoints (LlamaGen, Anole), write generate_image(prompt) using repo tokenizer + decoder.
  Next Steps:
  1. git clone https://github.com/FoundationVision/LlamaGen
  2. Download pretrained models per their README
  3. Place in ./pretrained_models/ directory
  4. Run: python3 step1_llamagen_inference.py

step2) Black-box Attack: Use CLIP/NSFW as score_fn, apply random or NES/SPSA search over prompt tokens.

step3) White-box Attack: Inject learnable soft embeddings; optimize w.r.t. CLIP loss via backprop through AR model.

step4) Attack Objectives: Use CLIP similarity, NSFW scores, realism, or safety classifiers as target functions.

step5) Evaluation: Track ASR, query budget, success rate, and show visual prompt-image pairs for comparison.

  prompt-adversarial-t2i/
  ├── step1_llamagen_inference.py    ✅ LlamaGen T2I inference
  ├── step2_blackbox_attack.py       ✅ CLIP scoring + random search
  ├── step3_whitebox_attack.py       ✅ Soft embeddings + gradient optimization
  ├── step4_objectives.py            ✅ CLIP/NSFW/Safety/Realism objectives
  ├── step5_evaluation.py            ✅ ASR tracking + comprehensive evaluation
  ├── requirements.txt               ✅ All dependencies
  ├── models/                        ✅ Git submodules
  │   ├── LlamaGen/
  │   └── anole/
  └── pretrained_models/             ✅ Model checkpoints