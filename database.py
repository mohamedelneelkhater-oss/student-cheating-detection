import sqlite3
from datetime import datetime


class AlertDatabase:

    def __init__(self, db_path="alerts.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    detected_at TEXT,
                    behavior TEXT,
                    confidence REAL,
                    screenshot_path TEXT
                )
            """
            )
            conn.commit()

    def insert_alert(
        self, behavior, confidence, screenshot_path=None, detected_at=None
    ):
        if detected_at is None:
            detected_at = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO alerts (detected_at, behavior, confidence, screenshot_path)
                VALUES (?, ?, ?, ?)
            """,
                (detected_at, behavior, confidence, screenshot_path),
            )
            conn.commit()

    def fetch_all(self, limit=500):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, detected_at, behavior, confidence, screenshot_path
                FROM alerts
                ORDER BY id DESC
                LIMIT ?
            """,
                (limit,),
            )
            return cursor.fetchall()
