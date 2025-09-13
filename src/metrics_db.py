#!/usr/bin/env python3
"""
Metrics Database - Store attack results and metrics persistently
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Any

class MetricsDB:
    def __init__(self, db_path: str = "attack_metrics.db"):
        """Initialize metrics database"""
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Create database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Attack results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attack_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attack_type TEXT,
                scenario_name TEXT,
                initial_prompt TEXT,
                target_concept TEXT,
                best_score REAL,
                best_prompt TEXT,
                iterations INTEGER,
                timestamp TEXT,
                metadata TEXT
            )
        """)
        
        # Iteration details table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS iteration_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attack_id INTEGER,
                iteration_num INTEGER,
                prompt TEXT,
                clip_loss REAL,
                diversity_loss REAL,
                total_loss REAL,
                score REAL,
                FOREIGN KEY (attack_id) REFERENCES attack_results (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def store_attack_result(self, 
                           attack_type: str,
                           scenario_name: str,
                           initial_prompt: str,
                           target_concept: str,
                           best_score: float,
                           best_prompt: str,
                           iterations: int,
                           iteration_data: List[Dict] = None,
                           metadata: Dict = None) -> int:
        """Store attack result and return attack_id"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert main attack result
        cursor.execute("""
            INSERT INTO attack_results 
            (attack_type, scenario_name, initial_prompt, target_concept, 
             best_score, best_prompt, iterations, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            attack_type, scenario_name, initial_prompt, target_concept,
            best_score, best_prompt, iterations, 
            datetime.now().isoformat(),
            json.dumps(metadata or {})
        ))
        
        attack_id = cursor.lastrowid
        
        # Insert iteration details if provided
        if iteration_data:
            for iter_data in iteration_data:
                cursor.execute("""
                    INSERT INTO iteration_details
                    (attack_id, iteration_num, prompt, clip_loss, diversity_loss, total_loss, score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    attack_id,
                    iter_data.get('iteration', 0),
                    iter_data.get('prompt', ''),
                    iter_data.get('clip_loss', 0.0),
                    iter_data.get('diversity_loss', 0.0),
                    iter_data.get('total_loss', 0.0),
                    iter_data.get('score', 0.0)
                ))
        
        conn.commit()
        conn.close()
        return attack_id
    
    def get_attack_results(self, attack_type: str = None) -> List[Dict]:
        """Retrieve attack results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if attack_type:
            cursor.execute("SELECT * FROM attack_results WHERE attack_type = ?", (attack_type,))
        else:
            cursor.execute("SELECT * FROM attack_results")
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'attack_type': row[1],
                'scenario_name': row[2],
                'initial_prompt': row[3],
                'target_concept': row[4],
                'best_score': row[5],
                'best_prompt': row[6],
                'iterations': row[7],
                'timestamp': row[8],
                'metadata': json.loads(row[9])
            })
        
        conn.close()
        return results
    
    def get_iteration_details(self, attack_id: int) -> List[Dict]:
        """Get detailed iteration data for an attack"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT iteration_num, prompt, clip_loss, diversity_loss, total_loss, score
            FROM iteration_details 
            WHERE attack_id = ? 
            ORDER BY iteration_num
        """, (attack_id,))
        
        details = []
        for row in cursor.fetchall():
            details.append({
                'iteration': row[0],
                'prompt': row[1],
                'clip_loss': row[2],
                'diversity_loss': row[3],
                'total_loss': row[4],
                'score': row[5]
            })
        
        conn.close()
        return details

def main():
    """Test metrics database"""
    db = MetricsDB("test_metrics.db")
    
    # Test data
    attack_id = db.store_attack_result(
        attack_type="blackbox",
        scenario_name="test_scenario",
        initial_prompt="test prompt",
        target_concept="test target",
        best_score=0.85,
        best_prompt="best test prompt",
        iterations=10,
        iteration_data=[
            {'iteration': 1, 'prompt': 'iter1', 'score': 0.7, 'clip_loss': -0.7},
            {'iteration': 2, 'prompt': 'iter2', 'score': 0.85, 'clip_loss': -0.85}
        ]
    )
    
    print(f"Stored attack with ID: {attack_id}")
    
    # Retrieve results
    results = db.get_attack_results()
    print(f"Retrieved {len(results)} attack results")
    
    details = db.get_iteration_details(attack_id)
    print(f"Retrieved {len(details)} iteration details")

if __name__ == "__main__":
    main()
