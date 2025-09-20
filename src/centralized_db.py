#!/usr/bin/env python3
"""
Centralized Database Manager for Adversarial Text-to-Image Pipeline
Replaces all JSON files with a single SQLite database with proper tables for each step.
"""

import sqlite3
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class DatabaseConfig:
    """Configuration for the centralized database"""
    db_path: str = "outputs/centralized_pipeline.db"

class CentralizedDB:
    """Centralized database manager for the entire adversarial pipeline"""

    def __init__(self, config: DatabaseConfig = None):
        """Initialize centralized database"""
        self.config = config or DatabaseConfig()
        self.db_path = self.config.db_path

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self._init_database()
        print(f"Centralized database initialized: {self.db_path}")

    def _init_database(self):
        """Create all database tables for the pipeline"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")

        # Step 1: Baseline inference results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS step1_inference (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_text TEXT NOT NULL,
                image_path TEXT NOT NULL,
                generation_time REAL,
                model_size TEXT,
                config_params TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                success BOOLEAN DEFAULT TRUE
            )
        """)

        # Step 2: Black-box attack results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS step2_blackbox_attacks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attack_id TEXT UNIQUE NOT NULL,
                scenario_name TEXT,
                initial_prompt TEXT,
                target_concept TEXT,
                best_score REAL,
                best_prompt TEXT,
                best_image_path TEXT,
                total_iterations INTEGER,
                success_threshold REAL,
                attack_success BOOLEAN,
                stealth_metrics TEXT,
                attack_type TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Step 2: Black-box iteration details
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS step2_blackbox_iterations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attack_id TEXT,
                iteration_num INTEGER,
                prompt_text TEXT,
                clip_score REAL,
                stealth_score REAL,
                attack_stage TEXT,
                mutation_type TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (attack_id) REFERENCES step2_blackbox_attacks (attack_id)
            )
        """)

        # Step 3: White-box attack results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS step3_whitebox_attacks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attack_id TEXT UNIQUE NOT NULL,
                scenario_name TEXT,
                base_prompt TEXT,
                target_concept TEXT,
                num_soft_tokens INTEGER,
                best_score REAL,
                best_prompt TEXT,
                best_image_path TEXT,
                total_iterations INTEGER,
                final_embeddings TEXT,
                stealth_metrics TEXT,
                attack_success BOOLEAN,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Step 3: White-box iteration details
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS step3_whitebox_iterations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attack_id TEXT,
                iteration_num INTEGER,
                prompt_text TEXT,
                clip_loss REAL,
                diversity_loss REAL,
                stealth_loss REAL,
                total_loss REAL,
                evolution_stage TEXT,
                embedding_metrics TEXT,
                stealth_score REAL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (attack_id) REFERENCES step3_whitebox_attacks (attack_id)
            )
        """)

        # Step 4: Objective evaluation results
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS step4_objective_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_id TEXT UNIQUE NOT NULL,
                image_path TEXT,
                source_attack_type TEXT,
                source_attack_id TEXT,
                prompt_text TEXT,
                filename_score REAL,
                combined_objective_score REAL,
                individual_scores TEXT,
                objective_weights TEXT,
                evaluation_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Step 4: Individual objective scores
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS step4_individual_objectives (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_id TEXT,
                objective_name TEXT,
                objective_score REAL,
                objective_type TEXT,
                objective_config TEXT,
                FOREIGN KEY (evaluation_id) REFERENCES step4_objective_evaluations (evaluation_id)
            )
        """)

        # Step 5: Final evaluation and research metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS step5_evaluation_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_run_id TEXT UNIQUE NOT NULL,
                total_attacks INTEGER,
                blackbox_count INTEGER,
                whitebox_count INTEGER,
                overall_asr REAL,
                blackbox_asr REAL,
                whitebox_asr REAL,
                analysis_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                research_summary TEXT,
                plots_generated TEXT
            )
        """)

        # Pipeline execution tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id TEXT UNIQUE NOT NULL,
                step_name TEXT,
                step_number INTEGER,
                status TEXT,
                start_time TEXT,
                end_time TEXT,
                duration REAL,
                error_message TEXT,
                config_used TEXT
            )
        """)

        # Create indices for better query performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_blackbox_attack_id ON step2_blackbox_iterations(attack_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_whitebox_attack_id ON step3_whitebox_iterations(attack_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_objective_eval_id ON step4_individual_objectives(evaluation_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_execution ON pipeline_executions(execution_id, step_name)")

        conn.commit()
        conn.close()

    # =====================================
    # STEP 1: INFERENCE METHODS
    # =====================================

    def store_inference_result(self, prompt: str, image_path: str, generation_time: float = 0.0,
                              model_size: str = "700M", config_params: Dict = None, success: bool = True) -> int:
        """Store Step 1 inference result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO step1_inference
            (prompt_text, image_path, generation_time, model_size, config_params, success)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (prompt, image_path, generation_time, model_size,
              json.dumps(config_params or {}), success))

        result_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return result_id

    def get_inference_results(self) -> List[Dict]:
        """Get all Step 1 inference results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM step1_inference ORDER BY timestamp DESC")
        results = []

        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'prompt_text': row[1],
                'image_path': row[2],
                'generation_time': row[3],
                'model_size': row[4],
                'config_params': json.loads(row[5]) if row[5] else {},
                'timestamp': row[6],
                'success': bool(row[7])
            })

        conn.close()
        return results

    # =====================================
    # STEP 2: BLACK-BOX ATTACK METHODS
    # =====================================

    def store_blackbox_attack(self, attack_id: str, scenario_name: str, initial_prompt: str,
                             target_concept: str, best_score: float, best_prompt: str,
                             best_image_path: str, total_iterations: int, success_threshold: float = 0.85,
                             attack_success: bool = False, stealth_metrics: Dict = None,
                             attack_type: str = "blackbox") -> str:
        """Store Step 2 black-box attack result with overwrite capability"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Generate deterministic attack_id based on scenario
        clean_scenario = scenario_name.replace(' ', '_').replace('-', '_')
        deterministic_id = f"bb_{clean_scenario}"

        # Delete existing iterations for this scenario first
        cursor.execute("DELETE FROM step2_blackbox_iterations WHERE attack_id = ?", (deterministic_id,))

        cursor.execute("""
            INSERT OR REPLACE INTO step2_blackbox_attacks
            (attack_id, scenario_name, initial_prompt, target_concept, best_score, best_prompt,
             best_image_path, total_iterations, success_threshold, attack_success, stealth_metrics, attack_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (deterministic_id, scenario_name, initial_prompt, target_concept, best_score, best_prompt,
              best_image_path, total_iterations, success_threshold, attack_success,
              json.dumps(stealth_metrics or {}), attack_type))

        result_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return deterministic_id  # Return the deterministic ID instead of row ID

    def store_blackbox_iteration(self, attack_id: str, iteration_num: int, prompt_text: str,
                                clip_score: float, stealth_score: float = 0.0, attack_stage: str = "",
                                mutation_type: str = "") -> int:
        """Store Step 2 black-box iteration details"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO step2_blackbox_iterations
            (attack_id, iteration_num, prompt_text, clip_score, stealth_score, attack_stage, mutation_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (attack_id, iteration_num, prompt_text, clip_score, stealth_score, attack_stage, mutation_type))

        result_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return result_id

    def get_blackbox_attacks(self) -> List[Dict]:
        """Get all Step 2 black-box attack results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM step2_blackbox_attacks ORDER BY timestamp DESC")
        results = []

        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'attack_id': row[1],
                'scenario_name': row[2],
                'initial_prompt': row[3],
                'target_concept': row[4],
                'best_score': row[5],
                'best_prompt': row[6],
                'best_image_path': row[7],
                'total_iterations': row[8],
                'success_threshold': row[9],
                'attack_success': bool(row[10]),
                'stealth_metrics': json.loads(row[11]) if row[11] else {},
                'attack_type': row[12],
                'timestamp': row[13]
            })

        conn.close()
        return results

    def get_blackbox_iterations(self, attack_id: str) -> List[Dict]:
        """Get Step 2 black-box iteration details for specific attack"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM step2_blackbox_iterations
            WHERE attack_id = ? ORDER BY iteration_num
        """, (attack_id,))

        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'attack_id': row[1],
                'iteration_num': row[2],
                'prompt_text': row[3],
                'clip_score': row[4],
                'stealth_score': row[5],
                'attack_stage': row[6],
                'mutation_type': row[7],
                'timestamp': row[8]
            })

        conn.close()
        return results

    # =====================================
    # STEP 3: WHITE-BOX ATTACK METHODS
    # =====================================

    def store_whitebox_attack(self, attack_id: str, scenario_name: str, base_prompt: str,
                             target_concept: str, num_soft_tokens: int, best_score: float,
                             best_prompt: str, best_image_path: str, total_iterations: int,
                             final_embeddings: Any = None, stealth_metrics: Dict = None,
                             attack_success: bool = False) -> int:
        """Store Step 3 white-box attack result with overwrite capability"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Generate deterministic attack_id based on scenario
        clean_scenario = scenario_name.replace(' ', '_').replace('-', '_')
        deterministic_id = f"wb_{clean_scenario}"

        # Delete existing iterations for this scenario first
        cursor.execute("DELETE FROM step3_whitebox_iterations WHERE attack_id = ?", (deterministic_id,))

        # Convert embeddings to JSON if it's a tensor
        embeddings_json = None
        if final_embeddings is not None:
            if hasattr(final_embeddings, 'cpu'):  # PyTorch tensor
                embeddings_json = json.dumps(final_embeddings.cpu().numpy().tolist())
            else:
                embeddings_json = json.dumps(final_embeddings)

        cursor.execute("""
            INSERT OR REPLACE INTO step3_whitebox_attacks
            (attack_id, scenario_name, base_prompt, target_concept, num_soft_tokens, best_score,
             best_prompt, best_image_path, total_iterations, final_embeddings, stealth_metrics, attack_success)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (deterministic_id, scenario_name, base_prompt, target_concept, num_soft_tokens, best_score,
              best_prompt, best_image_path, total_iterations, embeddings_json,
              json.dumps(stealth_metrics or {}), attack_success))

        result_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return deterministic_id  # Return the deterministic ID instead of row ID

    def store_whitebox_iteration(self, attack_id: str, iteration_num: int, prompt_text: str,
                                clip_loss: float, diversity_loss: float, stealth_loss: float,
                                total_loss: float, evolution_stage: str = "",
                                embedding_metrics: Dict = None, stealth_score: float = 0.0) -> int:
        """Store Step 3 white-box iteration details"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO step3_whitebox_iterations
            (attack_id, iteration_num, prompt_text, clip_loss, diversity_loss, stealth_loss,
             total_loss, evolution_stage, embedding_metrics, stealth_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (attack_id, iteration_num, prompt_text, clip_loss, diversity_loss, stealth_loss,
              total_loss, evolution_stage, json.dumps(embedding_metrics or {}), stealth_score))

        result_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return result_id

    def get_whitebox_attacks(self) -> List[Dict]:
        """Get all Step 3 white-box attack results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM step3_whitebox_attacks ORDER BY timestamp DESC")
        results = []

        for row in cursor.fetchall():
            # Parse embeddings back if they exist
            final_embeddings = None
            if row[10]:  # final_embeddings column
                try:
                    final_embeddings = json.loads(row[10])
                except:
                    final_embeddings = None

            results.append({
                'id': row[0],
                'attack_id': row[1],
                'scenario_name': row[2],
                'base_prompt': row[3],
                'target_concept': row[4],
                'num_soft_tokens': row[5],
                'best_score': row[6],
                'best_prompt': row[7],
                'best_image_path': row[8],
                'total_iterations': row[9],
                'final_embeddings': final_embeddings,
                'stealth_metrics': json.loads(row[11]) if row[11] else {},
                'attack_success': bool(row[12]),
                'timestamp': row[13]
            })

        conn.close()
        return results

    def get_whitebox_iterations(self, attack_id: str) -> List[Dict]:
        """Get Step 3 white-box iteration details for specific attack"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM step3_whitebox_iterations
            WHERE attack_id = ? ORDER BY iteration_num
        """, (attack_id,))

        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'attack_id': row[1],
                'iteration_num': row[2],
                'prompt_text': row[3],
                'clip_loss': row[4],
                'diversity_loss': row[5],
                'stealth_loss': row[6],
                'total_loss': row[7],
                'evolution_stage': row[8],
                'embedding_metrics': json.loads(row[9]) if row[9] else {},
                'stealth_score': row[10],
                'timestamp': row[11]
            })

        conn.close()
        return results

    # =====================================
    # STEP 4: OBJECTIVE EVALUATION METHODS
    # =====================================

    def store_objective_evaluation(self, evaluation_id: str, image_path: str, source_attack_type: str,
                                  source_attack_id: str, prompt_text: str, filename_score: float,
                                  combined_score: float, individual_scores: Dict, objective_weights: List = None) -> str:
        """Store Step 4 objective evaluation result with overwrite capability"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Generate deterministic evaluation_id based on image_path
        import os
        clean_image_name = os.path.basename(image_path).replace('.', '_').replace(' ', '_')
        deterministic_id = f"eval_{source_attack_type}_{clean_image_name}"

        # Delete existing individual objectives for this evaluation
        cursor.execute("DELETE FROM step4_individual_objectives WHERE evaluation_id = ?", (deterministic_id,))

        cursor.execute("""
            INSERT OR REPLACE INTO step4_objective_evaluations
            (evaluation_id, image_path, source_attack_type, source_attack_id, prompt_text,
             filename_score, combined_objective_score, individual_scores, objective_weights)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (deterministic_id, image_path, source_attack_type, source_attack_id, prompt_text,
              filename_score, combined_score, json.dumps(individual_scores),
              json.dumps(objective_weights or [])))

        result_id = cursor.lastrowid

        # Store individual objective scores
        for obj_name, obj_score in individual_scores.items():
            cursor.execute("""
                INSERT INTO step4_individual_objectives
                (evaluation_id, objective_name, objective_score, objective_type)
                VALUES (?, ?, ?, ?)
            """, (deterministic_id, obj_name, obj_score, self._classify_objective_type(obj_name)))

        conn.commit()
        conn.close()
        return deterministic_id

    def _classify_objective_type(self, objective_name: str) -> str:
        """Classify objective type based on name"""
        name_lower = objective_name.lower()
        if "clip" in name_lower:
            return "similarity"
        elif "nsfw" in name_lower:
            return "safety"
        elif "realism" in name_lower or "quality" in name_lower:
            return "quality"
        elif "safety" in name_lower:
            return "safety"
        else:
            return "other"

    def get_objective_evaluations(self, source_attack_type: str = None) -> List[Dict]:
        """Get Step 4 objective evaluation results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if source_attack_type:
            cursor.execute("""
                SELECT * FROM step4_objective_evaluations
                WHERE source_attack_type = ? ORDER BY evaluation_timestamp DESC
            """, (source_attack_type,))
        else:
            cursor.execute("SELECT * FROM step4_objective_evaluations ORDER BY evaluation_timestamp DESC")

        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'evaluation_id': row[1],
                'image_path': row[2],
                'source_attack_type': row[3],
                'source_attack_id': row[4],
                'prompt_text': row[5],
                'filename_score': row[6],
                'combined_objective_score': row[7],
                'individual_scores': json.loads(row[8]) if row[8] else {},
                'objective_weights': json.loads(row[9]) if row[9] else [],
                'evaluation_timestamp': row[10]
            })

        conn.close()
        return results

    def get_objective_summary_stats(self) -> Dict:
        """Get Step 4 summary statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get overall stats
        cursor.execute("""
            SELECT
                COUNT(*) as total_evaluations,
                AVG(combined_objective_score) as avg_combined_score,
                MAX(combined_objective_score) as max_combined_score,
                MIN(combined_objective_score) as min_combined_score
            FROM step4_objective_evaluations
        """)
        overall_stats = cursor.fetchone()

        # Get stats by attack type
        cursor.execute("""
            SELECT
                source_attack_type,
                COUNT(*) as count,
                AVG(combined_objective_score) as avg_score,
                AVG(filename_score) as avg_filename_score
            FROM step4_objective_evaluations
            GROUP BY source_attack_type
        """)
        attack_type_stats = cursor.fetchall()

        # Get individual objective averages
        cursor.execute("""
            SELECT
                objective_name,
                objective_type,
                AVG(objective_score) as avg_score,
                COUNT(*) as count
            FROM step4_individual_objectives
            GROUP BY objective_name, objective_type
        """)
        objective_stats = cursor.fetchall()

        conn.close()

        return {
            'overall': {
                'total_evaluations': overall_stats[0],
                'avg_combined_score': overall_stats[1],
                'max_combined_score': overall_stats[2],
                'min_combined_score': overall_stats[3]
            },
            'by_attack_type': [
                {
                    'attack_type': row[0],
                    'count': row[1],
                    'avg_combined_score': row[2],
                    'avg_filename_score': row[3]
                } for row in attack_type_stats
            ],
            'by_objective': [
                {
                    'objective_name': row[0],
                    'objective_type': row[1],
                    'avg_score': row[2],
                    'count': row[3]
                } for row in objective_stats
            ]
        }

    # =====================================
    # STEP 5: EVALUATION SUMMARY METHODS
    # =====================================

    def store_evaluation_summary(self, evaluation_run_id: str, total_attacks: int, blackbox_count: int,
                                whitebox_count: int, overall_asr: float, blackbox_asr: float,
                                whitebox_asr: float, research_summary: str = "", plots_generated: List = None) -> str:
        """Store Step 5 evaluation summary with overwrite capability"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Generate deterministic evaluation_run_id
        deterministic_id = f"eval_run_{total_attacks}attacks_{blackbox_count}bb_{whitebox_count}wb"

        cursor.execute("""
            INSERT OR REPLACE INTO step5_evaluation_summary
            (evaluation_run_id, total_attacks, blackbox_count, whitebox_count, overall_asr,
             blackbox_asr, whitebox_asr, research_summary, plots_generated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (deterministic_id, total_attacks, blackbox_count, whitebox_count, overall_asr,
              blackbox_asr, whitebox_asr, research_summary, json.dumps(plots_generated or [])))

        result_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return deterministic_id

    def get_evaluation_summaries(self) -> List[Dict]:
        """Get all Step 5 evaluation summaries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM step5_evaluation_summary ORDER BY analysis_timestamp DESC")
        results = []

        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'evaluation_run_id': row[1],
                'total_attacks': row[2],
                'blackbox_count': row[3],
                'whitebox_count': row[4],
                'overall_asr': row[5],
                'blackbox_asr': row[6],
                'whitebox_asr': row[7],
                'analysis_timestamp': row[8],
                'research_summary': row[9],
                'plots_generated': json.loads(row[10]) if row[10] else []
            })

        conn.close()
        return results

    # =====================================
    # PIPELINE EXECUTION TRACKING
    # =====================================

    def track_pipeline_step(self, execution_id: str, step_name: str, step_number: int,
                           status: str = "started", error_message: str = None, config_used: Dict = None) -> int:
        """Track pipeline step execution"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        current_time = datetime.now().isoformat()

        if status == "started":
            cursor.execute("""
                INSERT INTO pipeline_executions
                (execution_id, step_name, step_number, status, start_time, config_used)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (execution_id, step_name, step_number, status, current_time, json.dumps(config_used or {})))
        else:  # completed or failed
            # Update existing record
            cursor.execute("""
                UPDATE pipeline_executions
                SET status = ?, end_time = ?, error_message = ?
                WHERE execution_id = ? AND step_name = ? AND step_number = ?
            """, (status, current_time, error_message, execution_id, step_name, step_number))

            # Calculate duration
            cursor.execute("""
                UPDATE pipeline_executions
                SET duration = (julianday(end_time) - julianday(start_time)) * 86400
                WHERE execution_id = ? AND step_name = ? AND step_number = ?
            """, (execution_id, step_name, step_number))

        result_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return result_id

    def get_pipeline_execution_status(self, execution_id: str = None) -> List[Dict]:
        """Get pipeline execution status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if execution_id:
            cursor.execute("""
                SELECT * FROM pipeline_executions
                WHERE execution_id = ? ORDER BY step_number
            """, (execution_id,))
        else:
            cursor.execute("SELECT * FROM pipeline_executions ORDER BY start_time DESC")

        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'execution_id': row[1],
                'step_name': row[2],
                'step_number': row[3],
                'status': row[4],
                'start_time': row[5],
                'end_time': row[6],
                'duration': row[7],
                'error_message': row[8],
                'config_used': json.loads(row[9]) if row[9] else {}
            })

        conn.close()
        return results

    # =====================================
    # UTILITY METHODS
    # =====================================

    def export_to_research_format(self, output_path: str = "research_export.json") -> str:
        """Export all data in research-friendly format"""
        export_data = {
            'step1_inference': self.get_inference_results(),
            'step2_blackbox': self.get_blackbox_attacks(),
            'step3_whitebox': self.get_whitebox_attacks(),
            'step4_objectives': self.get_objective_evaluations(),
            'step5_summaries': self.get_evaluation_summaries(),
            'objective_stats': self.get_objective_summary_stats(),
            'export_timestamp': datetime.now().isoformat()
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"Research data exported to: {output_path}")
        return output_path

    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        tables = [
            'step1_inference', 'step2_blackbox_attacks', 'step2_blackbox_iterations',
            'step3_whitebox_attacks', 'step3_whitebox_iterations', 'step4_objective_evaluations',
            'step4_individual_objectives', 'step5_evaluation_summary', 'pipeline_executions'
        ]

        stats = {}
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]

        conn.close()
        return stats

    def cleanup_old_data(self, days_old: int = 30) -> int:
        """Clean up data older than specified days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        cutoff_iso = datetime.fromtimestamp(cutoff_date).isoformat()

        # Clean up old pipeline executions
        cursor.execute("DELETE FROM pipeline_executions WHERE start_time < ?", (cutoff_iso,))
        deleted_count = cursor.rowcount

        conn.commit()
        conn.close()

        print(f"Cleaned up {deleted_count} old pipeline execution records")
        return deleted_count

def main():
    """Test centralized database"""
    print("Testing Centralized Database Manager")
    print("=" * 50)

    # Initialize database
    db = CentralizedDB()

    # Test Step 1
    print("Testing Step 1 (Inference)...")
    inf_id = db.store_inference_result("test prompt", "test_image.png", 2.5, "700M")
    print(f"  Stored inference result with ID: {inf_id}")

    # Test Step 2
    print("Testing Step 2 (Black-box)...")
    bb_id = db.store_blackbox_attack("bb_test_001", "test_scenario", "initial prompt",
                                    "target concept", 0.85, "best prompt", "best_image.png", 20)
    db.store_blackbox_iteration("bb_test_001", 1, "iteration 1 prompt", 0.7)
    db.store_blackbox_iteration("bb_test_001", 2, "iteration 2 prompt", 0.85)
    print(f"  Stored black-box attack with ID: {bb_id}")

    # Test Step 3
    print("Testing Step 3 (White-box)...")
    wb_id = db.store_whitebox_attack("wb_test_001", "wb_scenario", "base prompt",
                                    "target concept", 5, 0.9, "wb_best_prompt", "wb_best.png", 30)
    db.store_whitebox_iteration("wb_test_001", 1, "wb iter 1", -0.7, 0.1, 0.2, -0.4, "steganographic")
    print(f"  Stored white-box attack with ID: {wb_id}")

    # Test Step 4
    print("Testing Step 4 (Objectives)...")
    obj_id = db.store_objective_evaluation("obj_test_001", "eval_image.png", "blackbox", "bb_test_001",
                                          "eval prompt", 0.8, 0.75, {"clip_sim": 0.8, "nsfw": 0.7})
    print(f"  Stored objective evaluation with ID: {obj_id}")

    # Test Step 5
    print("Testing Step 5 (Evaluation)...")
    eval_id = db.store_evaluation_summary("eval_run_001", 10, 5, 5, 0.8, 0.7, 0.9, "Test summary")
    print(f"  Stored evaluation summary with ID: {eval_id}")

    # Get stats
    print("\nDatabase Statistics:")
    stats = db.get_database_stats()
    for table, count in stats.items():
        print(f"  {table}: {count} records")

    # Export data
    print("\nExporting research data...")
    export_path = db.export_to_research_format("test_export.json")

    print(f"\nCentralized database test completed!")
    print(f"Database location: {db.db_path}")

if __name__ == "__main__":
    main()