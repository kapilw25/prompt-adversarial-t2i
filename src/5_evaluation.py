#!/usr/bin/env python3
"""
Step 5: Evaluation Framework - Database Analysis → Research Plots
"""

import os
import sqlite3
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass
from typing import List, Dict

# Import centralized database
from centralized_db import CentralizedDB

@dataclass
class AttackResult:
    attack_id: str
    attack_type: str
    success: bool
    final_score: float
    objective_scores: Dict[str, float]

class VisualEvaluator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_asr_analysis(self, attack_results: List[AttackResult]) -> str:
        # Calculate ASR by attack type
        blackbox_results = [r for r in attack_results if r.attack_type == 'blackbox']
        whitebox_results = [r for r in attack_results if r.attack_type == 'whitebox']
        
        blackbox_asr = sum(1 for r in blackbox_results if r.success) / len(blackbox_results) if blackbox_results else 0
        whitebox_asr = sum(1 for r in whitebox_results if r.success) / len(whitebox_results) if whitebox_results else 0
        
        # Create ASR plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        attack_types = ['Blackbox', 'Whitebox']
        asr_values = [blackbox_asr, whitebox_asr]
        
        bars = ax.bar(attack_types, asr_values, color=['skyblue', 'lightcoral'])
        ax.set_ylabel('Attack Success Rate')
        ax.set_title('ASR by Attack Type')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, asr_values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        output_path = os.path.join(self.output_dir, "asr_analysis.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path
    
    def plot_performance_metrics(self, attack_results: List[AttackResult]) -> str:
        # Create 4 separate charts for each objective
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract objective scores by attack type
        blackbox_results = [r for r in attack_results if r.attack_type == 'blackbox']
        whitebox_results = [r for r in attack_results if r.attack_type == 'whitebox']
        
        # 1. CLIP Similarity
        bb_clip = [r.objective_scores.get('CLIP_Similarity_to_dangerous_weapon', 0) for r in blackbox_results]
        wb_clip = [r.objective_scores.get('CLIP_Similarity_to_dangerous_weapon', 0) for r in whitebox_results]
        
        ax1.hist([bb_clip, wb_clip], bins=10, alpha=0.7, 
                label=['Blackbox', 'Whitebox'], color=['skyblue', 'lightcoral'])
        ax1.set_xlabel('CLIP Similarity Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('CLIP Similarity - Target Concept Matching')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. NSFW Detection
        bb_nsfw = [r.objective_scores.get('NSFW_Detection_Maximize', 0) for r in blackbox_results]
        wb_nsfw = [r.objective_scores.get('NSFW_Detection_Maximize', 0) for r in whitebox_results]
        
        ax2.hist([bb_nsfw, wb_nsfw], bins=10, alpha=0.7,
                label=['Blackbox', 'Whitebox'], color=['skyblue', 'lightcoral'])
        ax2.set_xlabel('NSFW Detection Score')
        ax2.set_ylabel('Frequency')
        ax2.set_title('NSFW Detection - Safety Filter Bypass')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Realism/Quality
        bb_realism = [r.objective_scores.get('Image_Realism_Quality', 0) for r in blackbox_results]
        wb_realism = [r.objective_scores.get('Image_Realism_Quality', 0) for r in whitebox_results]
        
        ax3.hist([bb_realism, wb_realism], bins=10, alpha=0.7,
                label=['Blackbox', 'Whitebox'], color=['skyblue', 'lightcoral'])
        ax3.set_xlabel('Realism Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Image Realism/Quality')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Safety Bypass
        bb_safety = [r.objective_scores.get('Safety_Classifier_Bypass', 0) for r in blackbox_results]
        wb_safety = [r.objective_scores.get('Safety_Classifier_Bypass', 0) for r in whitebox_results]
        
        ax4.hist([bb_safety, wb_safety], bins=10, alpha=0.7,
                label=['Blackbox', 'Whitebox'], color=['skyblue', 'lightcoral'])
        ax4.set_xlabel('Safety Bypass Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Safety Mechanism Bypass')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "performance_metrics.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return output_path

def main():
    """Step 5: Research Evaluation Pipeline using Centralized Database"""
    print("="*60)
    print("STEP 5: Research Paper Evaluation Pipeline")
    print("="*60)

    # Initialize centralized database
    db = CentralizedDB()

    # Get all attack data
    blackbox_attacks = db.get_blackbox_attacks()
    whitebox_attacks = db.get_whitebox_attacks()
    objective_evaluations = db.get_objective_evaluations()

    print(f"Found: {len(blackbox_attacks)} blackbox attacks")
    print(f"Found: {len(whitebox_attacks)} whitebox attacks")
    print(f"Found: {len(objective_evaluations)} objective evaluations")

    if not blackbox_attacks and not whitebox_attacks:
        print("ERROR: No attack data found in database!")
        print("Run previous steps (2_blackbox_attacks.py, 3_whitebox_attacks.py, 4_objectives.py) first!")
        return

    # Check if evaluation summary already exists BEFORE processing
    import sqlite3
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()
    
    total_attacks = len(blackbox_attacks) + len(whitebox_attacks)
    blackbox_count = len(blackbox_attacks)
    whitebox_count = len(whitebox_attacks)
    
    evaluation_run_id = f"eval_run_{total_attacks}attacks_{blackbox_count}bb_{whitebox_count}wb"
    
    cursor.execute("SELECT evaluation_run_id FROM step5_evaluation_summary WHERE evaluation_run_id = ?", (evaluation_run_id,))
    existing = cursor.fetchone()
    conn.close()
    
    if existing:
        response = input(f"Evaluation summary for {total_attacks} attacks already exists. Replace? [yes/No]: ").strip().lower()
        if response not in ['yes', 'y']:
            print("Skipping evaluation...")
            return

    # Setup output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "step5_evaluation")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Extract data from centralized database
    print("1. Loading attack and evaluation data from centralized database...")

    print(f"   Loaded: {len(blackbox_attacks)} blackbox attacks")
    print(f"   Loaded: {len(whitebox_attacks)} whitebox attacks")
    print(f"   Loaded: {len(objective_evaluations)} objective evaluations")

    # 2. Convert database data to attack results format
    print("2. Processing attack results for analysis...")

    attack_results = []

    # Process blackbox attacks
    for attack in blackbox_attacks:
        attack_result = AttackResult(
            attack_id=attack['attack_id'],
            attack_type='blackbox',
            success=attack['attack_success'],
            final_score=attack['best_score'],
            objective_scores={}  # Will be filled from evaluations
        )
        attack_results.append(attack_result)

    # Process whitebox attacks
    for attack in whitebox_attacks:
        attack_result = AttackResult(
            attack_id=attack['attack_id'],
            attack_type='whitebox',
            success=attack['attack_success'],
            final_score=attack['best_score'],
            objective_scores={}  # Will be filled from evaluations
        )
        attack_results.append(attack_result)

    # Add objective scores from evaluations
    for eval_data in objective_evaluations:
        # Find matching attack result
        for attack_result in attack_results:
            if attack_result.attack_id == eval_data['source_attack_id']:
                attack_result.objective_scores = eval_data['individual_scores']
                break

    print(f"   Processed {len(attack_results)} attack results")

    # 3. Generate research paper plots
    print("3. Generating research paper visualizations...")
    visual_evaluator = VisualEvaluator(os.path.join(output_dir, "research_plots"))

    asr_plot = visual_evaluator.plot_asr_analysis(attack_results)
    performance_plot = visual_evaluator.plot_performance_metrics(attack_results)

    print(f"   ASR Analysis Plot: {asr_plot}")
    print(f"   Performance Plot: {performance_plot}")

    # 4. Calculate comprehensive research metrics
    print("4. Calculating research metrics...")

    blackbox_count = len([r for r in attack_results if r.attack_type == 'blackbox'])
    whitebox_count = len([r for r in attack_results if r.attack_type == 'whitebox'])

    blackbox_asr = sum(1 for r in attack_results if r.attack_type == 'blackbox' and r.success) / blackbox_count if blackbox_count else 0
    whitebox_asr = sum(1 for r in attack_results if r.attack_type == 'whitebox' and r.success) / whitebox_count if whitebox_count else 0
    overall_asr = sum(1 for r in attack_results if r.success) / len(attack_results) if attack_results else 0

    # Calculate detailed objective metrics
    blackbox_results = [r for r in attack_results if r.attack_type == 'blackbox']
    whitebox_results = [r for r in attack_results if r.attack_type == 'whitebox']
    
    # Extract all objective scores
    objectives = ['CLIP_Similarity_to_dangerous_weapon', 'CLIP_Similarity_to_explicit_adult_conte', 
                 'CLIP_Similarity_to_violent_scene', 'NSFW_Detection_Maximize', 
                 'Image_Realism_Quality', 'Safety_Classifier_Bypass']
    
    detailed_metrics = {}
    for obj in objectives:
        bb_scores = [r.objective_scores.get(obj, 0) for r in blackbox_results]
        wb_scores = [r.objective_scores.get(obj, 0) for r in whitebox_results]
        all_scores = bb_scores + wb_scores
        
        detailed_metrics[obj] = {
            'blackbox_mean': sum(bb_scores) / len(bb_scores) if bb_scores else 0,
            'blackbox_max': max(bb_scores) if bb_scores else 0,
            'blackbox_min': min(bb_scores) if bb_scores else 0,
            'whitebox_mean': sum(wb_scores) / len(wb_scores) if wb_scores else 0,
            'whitebox_max': max(wb_scores) if wb_scores else 0,
            'whitebox_min': min(wb_scores) if wb_scores else 0,
            'overall_mean': sum(all_scores) / len(all_scores) if all_scores else 0,
            'overall_max': max(all_scores) if all_scores else 0,
            'overall_min': min(all_scores) if all_scores else 0
        }

    # Get database statistics
    db_stats = db.get_database_stats()
    objective_stats = db.get_objective_summary_stats()

    # 5. Store evaluation summary in database
    print("5. Storing evaluation summary in database...")

    # Create comprehensive research summary with detailed metrics
    research_summary = f"""Research Evaluation Summary:
- Total Attacks Analyzed: {len(attack_results)}
- Blackbox ASR: {blackbox_asr:.4f} ({blackbox_count} attacks)
- Whitebox ASR: {whitebox_asr:.4f} ({whitebox_count} attacks)
- Overall ASR: {overall_asr:.4f}
- Database Records: {sum(db_stats.values())} total
- Objective Evaluations: {objective_stats['overall']['total_evaluations']}

Detailed Objective Metrics:
"""
    
    for obj, metrics in detailed_metrics.items():
        obj_name = obj.replace('_', ' ').replace('to ', '').title()
        research_summary += f"""
{obj_name}:
  Blackbox: Mean={metrics['blackbox_mean']:.4f}, Max={metrics['blackbox_max']:.4f}, Min={metrics['blackbox_min']:.4f}
  Whitebox: Mean={metrics['whitebox_mean']:.4f}, Max={metrics['whitebox_max']:.4f}, Min={metrics['whitebox_min']:.4f}
  Overall:  Mean={metrics['overall_mean']:.4f}, Max={metrics['overall_max']:.4f}, Min={metrics['overall_min']:.4f}"""

    plots_generated = [
        os.path.basename(asr_plot),
        os.path.basename(performance_plot)
    ]

    summary_id = db.store_evaluation_summary(
        evaluation_run_id="",  # Will be generated by database
        total_attacks=len(attack_results),
        blackbox_count=blackbox_count,
        whitebox_count=whitebox_count,
        overall_asr=overall_asr,
        blackbox_asr=blackbox_asr,
        whitebox_asr=whitebox_asr,
        research_summary=research_summary,
        plots_generated=plots_generated
    )

    print(f"   Stored summary with ID: {summary_id}")

    # 6. All data stored in centralized database
    print("6. All data stored in centralized database...")

    print(f"\n{'='*60}")
    print("STEP 5 COMPLETED: Research Pipeline")
    print(f"{'='*60}")
    print(f"Centralized Database: {db.db_path}")
    print(f"Research Plots: {os.path.join(output_dir, 'research_plots')}")
    print(f"Evaluation Summary ID: {summary_id}")
    print(f"\nResearch Metrics:")
    print(f"  Total Attacks: {len(attack_results)}")
    print(f"  Overall ASR: {overall_asr:.4f}")
    print(f"  Blackbox ASR: {blackbox_asr:.4f} (n={blackbox_count})")
    print(f"  Whitebox ASR: {whitebox_asr:.4f} (n={whitebox_count})")
    print(f"  Database Records: {sum(db_stats.values())}")
    
    print(f"\nDetailed Objective Performance:")
    for obj, metrics in detailed_metrics.items():
        obj_name = obj.replace('_', ' ').replace('to ', '').title()
        print(f"  {obj_name}:")
        print(f"    BB: μ={metrics['blackbox_mean']:.3f} max={metrics['blackbox_max']:.3f} min={metrics['blackbox_min']:.3f}")
        print(f"    WB: μ={metrics['whitebox_mean']:.3f} max={metrics['whitebox_max']:.3f} min={metrics['whitebox_min']:.3f}")
    
    print("All data centralized in database - no JSON files!")
    print("="*60)

if __name__ == "__main__":
    main()
