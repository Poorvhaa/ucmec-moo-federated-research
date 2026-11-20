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

            # Federated learning tables
            conn.execute("""
            CREATE TABLE IF NOT EXISTS federated_experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                preset_name TEXT NOT NULL,
                scenario_type TEXT NOT NULL,
                num_clients INTEGER NOT NULL,
                num_rounds INTEGER NOT NULL,
                current_round INTEGER DEFAULT 0,
                min_clients INTEGER NOT NULL,
                participation_rate REAL NOT NULL,
                dropout_prob REAL NOT NULL,
                seed INTEGER NOT NULL,
                progress_percent REAL DEFAULT 0.0,
                created_at REAL NOT NULL,
                started_at REAL,
                finished_at REAL,
                runtime_s REAL DEFAULT 0.0,
                git_commit TEXT,
                git_dirty INTEGER DEFAULT 0,
                solver_name TEXT DEFAULT 'CLARABEL',
                channel_mode TEXT DEFAULT 'legacy',
                config_json TEXT NOT NULL,
                final_metrics_json TEXT,
                communication_summary_json TEXT,
                error_message TEXT,
                log_dir TEXT NOT NULL
            )
            """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS federated_rounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                round INTEGER NOT NULL,
                global_version INTEGER NOT NULL,
                status TEXT NOT NULL,
                runtime_s REAL NOT NULL,
                participating_clients_json TEXT NOT NULL,
                accepted_clients_json TEXT NOT NULL,
                client_weights_json TEXT NOT NULL,
                total_valid_samples INTEGER NOT NULL,
                mean_delay_ms REAL,
                mean_energy_mj REAL,
                mean_overhead_kb REAL,
                mean_task_success_rate REAL,
                worst_task_success_rate REAL,
                worst_delay_ms REAL,
                jains_fairness_index REAL,
                eval_delay_ms REAL,
                eval_energy_mj REAL,
                eval_overhead_kb REAL,
                eval_task_success_rate REAL,
                eval_reward REAL,
                round_download_bytes INTEGER,
                round_upload_bytes INTEGER,
                round_failed_bytes INTEGER,
                checkpoint_filename TEXT,
                FOREIGN KEY (experiment_id) REFERENCES federated_experiments (id)
            )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fed_rounds_exp ON federated_rounds (experiment_id, round)")

            conn.execute("""
            CREATE TABLE IF NOT EXISTS federated_clients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                client_id TEXT NOT NULL,
                site_name TEXT NOT NULL,
                seed INTEGER NOT NULL,
                task_size_range_json TEXT NOT NULL,
                c_edge_range_json TEXT NOT NULL,
                epsilon REAL NOT NULL,
                local_rollout_steps INTEGER NOT NULL,
                local_epochs INTEGER NOT NULL,
                FOREIGN KEY (experiment_id) REFERENCES federated_experiments (id)
            )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_fed_clients_exp ON federated_clients (experiment_id)")
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
            conn.execute("""
            UPDATE federated_experiments
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

    def create_federated_experiment(self, fed_data: Dict[str, Any]):
        with self.get_connection() as conn:
            conn.execute("""
            INSERT INTO federated_experiments (
                id, name, status, preset_name, scenario_type, num_clients,
                num_rounds, current_round, min_clients, participation_rate,
                dropout_prob, seed, progress_percent, created_at, started_at,
                finished_at, runtime_s, git_commit, git_dirty, solver_name,
                channel_mode, config_json, final_metrics_json, communication_summary_json,
                error_message, log_dir
            ) VALUES (
                :id, :name, :status, :preset_name, :scenario_type, :num_clients,
                :num_rounds, :current_round, :min_clients, :participation_rate,
                :dropout_prob, :seed, :progress_percent, :created_at, :started_at,
                :finished_at, :runtime_s, :git_commit, :git_dirty, :solver_name,
                :channel_mode, :config_json, :final_metrics_json, :communication_summary_json,
                :error_message, :log_dir
            )
            """, fed_data)
            conn.commit()

    def update_federated_experiment_status(
        self,
        exp_id: str,
        status: str,
        current_round: Optional[int] = None,
        progress_percent: Optional[float] = None,
        error_message: Optional[str] = None,
        started_at: Optional[float] = None,
        finished_at: Optional[float] = None,
        runtime_s: Optional[float] = None,
        final_metrics: Optional[Dict[str, Any]] = None,
        communication_summary: Optional[Dict[str, Any]] = None,
    ):
        with self.get_connection() as conn:
            updates = ["status = ?"]
            params = [status]

            if current_round is not None:
                updates.append("current_round = ?")
                params.append(current_round)
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
            if communication_summary is not None:
                updates.append("communication_summary_json = ?")
                params.append(json.dumps(communication_summary))

            params.append(exp_id)
            query = f"UPDATE federated_experiments SET {', '.join(updates)} WHERE id = ?"
            conn.execute(query, params)
            conn.commit()

    def insert_federated_round(self, exp_id: str, r_data: Dict[str, Any]):
        with self.get_connection() as conn:
            lm = r_data.get("local_training_metrics", {})
            em = r_data.get("eval_metrics", {})
            comm = r_data.get("communication", {})

            conn.execute("""
            INSERT INTO federated_rounds (
                experiment_id, round, global_version, status, runtime_s,
                participating_clients_json, accepted_clients_json, client_weights_json,
                total_valid_samples, mean_delay_ms, mean_energy_mj, mean_overhead_kb,
                mean_task_success_rate, worst_task_success_rate, worst_delay_ms,
                jains_fairness_index, eval_delay_ms, eval_energy_mj, eval_overhead_kb,
                eval_task_success_rate, eval_reward, round_download_bytes,
                round_upload_bytes, round_failed_bytes, checkpoint_filename
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """, (
                exp_id,
                r_data["round"],
                r_data["global_version"],
                r_data.get("status", "completed"),
                r_data.get("runtime_s", 0.0),
                json.dumps(r_data.get("participating_clients", [])),
                json.dumps(r_data.get("accepted_clients", [])),
                json.dumps(r_data.get("client_weights", {})),
                r_data.get("total_valid_samples", 0),
                lm.get("mean_delay_ms"),
                lm.get("mean_energy_mj"),
                lm.get("mean_overhead_kb"),
                lm.get("mean_task_success_rate"),
                lm.get("worst_task_success_rate"),
                lm.get("worst_delay_ms"),
                lm.get("jains_fairness_index"),
                em.get("eval_delay_ms"),
                em.get("eval_energy_mj"),
                em.get("eval_overhead_kb"),
                em.get("eval_task_success_rate"),
                em.get("eval_reward"),
                comm.get("download_bytes", 0),
                comm.get("upload_bytes", 0),
                comm.get("failed_bytes", 0),
                r_data.get("checkpoint_path"),
            ))
            conn.commit()

    def insert_federated_clients(self, exp_id: str, clients: List[Dict[str, Any]]):
        with self.get_connection() as conn:
            for c in clients:
                conn.execute("""
                INSERT INTO federated_clients (
                    experiment_id, client_id, site_name, seed,
                    task_size_range_json, c_edge_range_json, epsilon,
                    local_rollout_steps, local_epochs
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    exp_id,
                    c["client_id"],
                    c["site_name"],
                    c["seed"],
                    json.dumps(c.get("task_size_range", [50000.0, 100000.0])),
                    json.dumps(c.get("c_edge_range", [10e9, 20e9])),
                    c.get("epsilon", 6e-4),
                    c.get("local_rollout_steps", 200),
                    c.get("local_epochs", 5),
                ))
            conn.commit()

    def get_federated_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM federated_experiments WHERE id = ?", (exp_id,)).fetchone()
            if not row:
                return None
            return self._row_to_fed_exp_dict(row)

    def list_federated_experiments(self, page: int = 1, page_size: int = 20) -> Tuple[int, List[Dict[str, Any]]]:
        offset = (page - 1) * page_size
        with self.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM federated_experiments").fetchone()[0]
            rows = conn.execute(
                "SELECT * FROM federated_experiments ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (page_size, offset)
            ).fetchall()
            return count, [self._row_to_fed_exp_dict(r) for r in rows]

    def get_federated_rounds(self, exp_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM federated_rounds WHERE experiment_id = ? ORDER BY round ASC",
                (exp_id,)
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["participating_clients"] = json.loads(d["participating_clients_json"]) if d.get("participating_clients_json") else []
                d["accepted_clients"] = json.loads(d["accepted_clients_json"]) if d.get("accepted_clients_json") else []
                d["client_weights"] = json.loads(d["client_weights_json"]) if d.get("client_weights_json") else {}
                out.append(d)
            return out

    def get_federated_clients(self, exp_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM federated_clients WHERE experiment_id = ? ORDER BY client_id ASC",
                (exp_id,)
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["task_size_range"] = json.loads(d["task_size_range_json"]) if d.get("task_size_range_json") else []
                d["c_edge_range"] = json.loads(d["c_edge_range_json"]) if d.get("c_edge_range_json") else []
                out.append(d)
            return out

    def _row_to_fed_exp_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["config"] = json.loads(d["config_json"]) if d.get("config_json") else {}
        d["final_metrics"] = json.loads(d["final_metrics_json"]) if d.get("final_metrics_json") else None
        d["communication_summary"] = json.loads(d["communication_summary_json"]) if d.get("communication_summary_json") else None
        d["git_dirty"] = bool(d.get("git_dirty", 0))
        return d
