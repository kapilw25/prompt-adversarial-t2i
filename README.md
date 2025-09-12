# Prompt Adversarial Text-to-Image Generation

## System Design

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                SYSTEM ARCHITECTURE                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   1_inference.py    │    │ 2_blackbox_attacks  │    │ 3_whitebox_attacks  │
│                     │    │                     │    │                     │
│ ┌─────────────────┐ │    │ ┌─────────────────┐ │    │ ┌─────────────────┐ │
│ │ LlamaGenInfer   │ │    │ │ BlackBoxAttacker│ │    │ │SoftEmbedAttacker│ │
│ │                 │ │    │ │                 │ │    │ │                 │ │
│ │ • VQ Model      │ │    │ │ • CLIP Scoring  │ │    │ │ • Soft Embeddings│ │
│ │ • GPT Model     │ │    │ │ • Random Search │ │    │ │ • Gradient Opt   │ │
│ │ • T5 Encoder    │ │    │ │ • Prompt Mutate │ │    │ │ • CLIP Loss      │ │
│ │ • generate_img()│ │    │ │ • Evaluate      │ │    │ │ • Backprop       │ │
│ └─────────────────┘ │    │ └─────────────────┘ │    │ └─────────────────┘ │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
         │                           │                           │
         │                           │                           │
         └───────────────┬───────────┴───────────────┬───────────┘
                         │                           │
                         ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        4_objectives.py                                      │
│                                                                             │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────┐ │
│ │CLIPSimilarity   │ │NSFWDetection    │ │RealismObjective │ │SafetyBypass │ │
│ │Objective        │ │Objective        │ │                 │ │Objective    │ │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────┘ │
│                                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │                    MultiObjectiveEvaluator                              │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        5_evaluation.py                                     │
│                                                                             │
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────┐ │
│ │ASRCalculator    │ │EvaluationDB     │ │VisualEvaluator  │ │Comprehensive│ │
│ │                 │ │                 │ │                 │ │Evaluator    │ │
│ │• Success Rate   │ │• SQLite Storage │ │• Comparison Grid│ │• Orchestrate│ │
│ │• Thresholds     │ │• Attack Results │ │• ASR Plots      │ │• Full Pipeline│ │
│ │• Metrics        │ │• Query Logs     │ │• Performance    │ │• Reports    │ │
│ └─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Input Prompt ──┐
               │
               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            BASELINE GENERATION                               │
│                                                                              │
│  Text Prompt → T5 Encoder → GPT Model → VQ Decoder → Baseline Image         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          ADVERSARIAL ATTACKS                                │
│                                                                              │
│  ┌─────────────────────┐              ┌─────────────────────┐               │
│  │    BLACK-BOX        │              │    WHITE-BOX        │               │
│  │                     │              │                     │               │
│  │ Mutate Prompt ──────┼──────────────┼────── Soft Tokens   │               │
│  │      │              │              │         │           │               │
│  │      ▼              │              │         ▼           │               │
│  │ Generate Image      │              │ Inject Embeddings   │               │
│  │      │              │              │         │           │               │
│  │      ▼              │              │         ▼           │               │
│  │ CLIP Score ─────────┼──────────────┼────── CLIP Loss     │               │
│  │      │              │              │         │           │               │
│  │      ▼              │              │         ▼           │               │
│  │ Update Best         │              │ Gradient Update     │               │
│  └─────────────────────┘              └─────────────────────┘               │
└──────────────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        OBJECTIVE EVALUATION                                 │
│                                                                              │
│  Generated Image ──┬── CLIP Similarity ──┬── Multi-Objective Score          │
│                    ├── NSFW Detection ───┤                                  │
│                    ├── Realism Score ────┤                                  │
│                    └── Safety Bypass ────┘                                  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      COMPREHENSIVE EVALUATION                               │
│                                                                              │
│  Attack Results ──┬── ASR Calculation ──┬── Visual Comparisons              │
│                   ├── Database Storage ─┤                                   │
│                   ├── Performance Plots ┤                                   │
│                   └── Final Report ─────┘                                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Key Components

**1_inference.py**: Baseline text-to-image generation using LlamaGen (VQ-VAE + GPT + T5)

**2_blackbox_attacks.py**: Query-based attacks with prompt mutations and CLIP scoring

**3_whitebox_attacks.py**: Gradient-based attacks using learnable soft embeddings

**4_objectives.py**: Multi-objective attack evaluation (CLIP, NSFW, Safety, Realism)

**5_evaluation.py**: Comprehensive ASR analysis, visualization, and reporting framework

## Usage

```bash
python src/1_inference.py      # Generate baseline images
python src/2_blackbox_attacks.py   # Run black-box attacks  
python src/3_whitebox_attacks.py   # Run white-box attacks
python src/4_objectives.py     # Test objective functions
python src/5_evaluation.py     # Comprehensive evaluation
```
