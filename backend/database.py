"""
SQLite Metadata and Experiment Persistence Layer.
"""
import sqlite3
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


DB_DIR = Path("experiments_storage")
DB_PATH = DB_DIR / "experiments.db"


class Database:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self.get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                algorithm_name TEXT NOT NULL,
                preset_name TEXT NOT NULL,
                weights_json TEXT NOT NULL,
                num_agents INTEGER NOT NULL,
                num_env_steps INTEGER NOT NULL,
                episode_length INTEGER NOT NULL,
                seed INTEGER NOT NULL,
                current_step INTEGER DEFAULT 0,
                total_steps INTEGER NOT NULL,
                progress_percent REAL DEFAULT 0.0,
                created_at REAL NOT NULL,
                started_at REAL,
                finished_at REAL,
                runtime_s REAL DEFAULT 0.0,
                git_commit TEXT,
                git_dirty INTEGER DEFAULT 0,
                solver_name TEXT NOT NULL,
                use_corrected_cpu_alloc INTEGER DEFAULT 1,
                channel_mode TEXT DEFAULT 'legacy',
                config_json TEXT NOT NULL,
                final_metrics_json TEXT,
                error_message TEXT,
                log_dir TEXT NOT NULL
            )
            """)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                episode INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                reward REAL NOT NULL,
                delay_ms REAL NOT NULL,
                energy_mj REAL NOT NULL,
                overhead_kb REAL NOT NULL,
                task_success_rate REAL NOT NULL,
                deadline_satisfaction_rate REAL NOT NULL,
                channel_success_rate REAL NOT NULL,
                local_users REAL NOT NULL,
                offloaded_users REAL NOT NULL,
                solver_mean_ms REAL NOT NULL,
                solver_max_residual REAL NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_exp_step ON metrics (experiment_id, step)")
            conn.commit()

    def reconcile_interrupted_jobs(self):
        """Mark any jobs left in 'running' or 'queued' state as failed upon service restart."""
        with self.get_connection() as conn:
            conn.execute("""
            UPDATE experiments
            SET status = 'failed',
                finished_at = ?,
                error_message = 'Interrupted by service restart'
            WHERE status IN ('running', 'queued')
            """, (time.time(),))
            conn.commit()

    def create_experiment(self, exp_data: Dict[str, Any]):
        with self.get_connection() as conn:
            conn.execute("""
            INSERT INTO experiments (
                id, name, status, algorithm_name, preset_name, weights_json,
                num_agents, num_env_steps, episode_length, seed, current_step,
                total_steps, progress_percent, created_at, started_at, finished_at,
                runtime_s, git_commit, git_dirty, solver_name, use_corrected_cpu_alloc,
                channel_mode, config_json, final_metrics_json, error_message, log_dir
            ) VALUES (
                :id, :name, :status, :algorithm_name, :preset_name, :weights_json,
                :num_agents, :num_env_steps, :episode_length, :seed, :current_step,
                :total_steps, :progress_percent, :created_at, :started_at, :finished_at,
                :runtime_s, :git_commit, :git_dirty, :solver_name, :use_corrected_cpu_alloc,
                :channel_mode, :config_json, :final_metrics_json, :error_message, :log_dir
            )
            """, exp_data)
            conn.commit()

    def update_experiment_status(
        self,
        exp_id: str,
        status: str,
        current_step: Optional[int] = None,
        progress_percent: Optional[float] = None,
        error_message: Optional[str] = None,
        started_at: Optional[float] = None,
        finished_at: Optional[float] = None,
        runtime_s: Optional[float] = None,
        final_metrics: Optional[Dict[str, Any]] = None,
    ):
        with self.get_connection() as conn:
            updates = ["status = ?"]
            params = [status]

            if current_step is not None:
                updates.append("current_step = ?")
                params.append(current_step)
            if progress_percent is not None:
                updates.append("progress_percent = ?")
                params.append(progress_percent)
            if error_message is not None:
                updates.append("error_message = ?")
                params.append(error_message)
            if started_at is not None:
                updates.append("started_at = ?")
                params.append(started_at)
            if finished_at is not None:
                updates.append("finished_at = ?")
                params.append(finished_at)
            if runtime_s is not None:
                updates.append("runtime_s = ?")
                params.append(runtime_s)
            if final_metrics is not None:
                updates.append("final_metrics_json = ?")
                params.append(json.dumps(final_metrics))

            params.append(exp_id)
            query = f"UPDATE experiments SET {', '.join(updates)} WHERE id = ?"
            conn.execute(query, params)
            conn.commit()

    def insert_metric_record(self, exp_id: str, metric: Dict[str, Any]):
        with self.get_connection() as conn:
            conn.execute("""
            INSERT INTO metrics (
                experiment_id, step, episode, timestamp, reward, delay_ms,
                energy_mj, overhead_kb, task_success_rate, deadline_satisfaction_rate,
                channel_success_rate, local_users, offloaded_users, solver_mean_ms,
                solver_max_residual
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """, (
                exp_id,
                metric["step"],
                metric["episode"],
                metric["timestamp"],
                metric["reward"],
                metric["delay_ms"],
                metric["energy_mj"],
                metric["overhead_kb"],
                metric["task_success_rate"],
                metric["deadline_satisfaction_rate"],
                metric["channel_success_rate"],
                metric["local_users"],
                metric["offloaded_users"],
                metric["solver_mean_ms"],
                metric["solver_max_residual"],
            ))
            conn.commit()

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
            if not row:
                return None
            return self._row_to_exp_dict(row)

    def list_experiments(self, page: int = 1, page_size: int = 20) -> Tuple[int, List[Dict[str, Any]]]:
        offset = (page - 1) * page_size
        with self.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (page_size, offset)
            ).fetchall()
            return count, [self._row_to_exp_dict(r) for r in rows]

    def get_experiment_metrics(self, exp_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM metrics WHERE experiment_id = ? ORDER BY step ASC",
                (exp_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_completed_experiments(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM experiments WHERE status = 'completed' ORDER BY created_at DESC"
            ).fetchall()
            return [self._row_to_exp_dict(r) for r in rows]

    def _row_to_exp_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["weights"] = json.loads(d["weights_json"]) if d.get("weights_json") else {}
        d["config"] = json.loads(d["config_json"]) if d.get("config_json") else {}
        d["final_metrics"] = json.loads(d["final_metrics_json"]) if d.get("final_metrics_json") else None
        d["git_dirty"] = bool(d.get("git_dirty", 0))
        d["use_corrected_cpu_alloc"] = bool(d.get("use_corrected_cpu_alloc", 1))
        return d
