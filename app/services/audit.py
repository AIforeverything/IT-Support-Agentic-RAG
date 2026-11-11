import json
import sqlite3
from datetime import datetime, timezone
from app.core.config import get_settings

settings = get_settings()


def init_db() -> None:
    con = sqlite3.connect(datebase=settings.AUDIT_DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS query_audit (
        id INTEGER PRIMARYKEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        question TEXT NOT NULL,
        source_used TEXT NOT NULL,
        trace_json TEXT NOT NULL,
        )
        """
    )
    con.commit()
    con.close()


def write_audit(question: str, source_used: str, trace: list[str]):
    con = sqlite3.connect(database=settings.AUDIT_DB_PATH)
    con.execute(
        "INSERT INTO query_audit(created_at,question,source_used,trace_json)VALUES (?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(),
         question, source_used, json.dumps(trace))
    )
    con.commit()
    con.close()
