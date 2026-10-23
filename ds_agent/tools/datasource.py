"""External data sources: connect enterprise databases and offline spreadsheets.

The agent ships with a demo warehouse; real DS work starts from *your* data.
This registry attaches external sources into the same DuckDB session so the
existing tools (natural-language SQL, diagnosis, modeling, dashboards) work
over them unchanged:

- offline files: CSV / TSV / Parquet / JSON / Excel (``read_csv_auto`` views or
  an Excel sheet imported into a table);
- databases: another DuckDB file (``ATTACH``), SQLite (sqlite extension),
  PostgreSQL / MySQL (extension + connection string, best effort offline).

Registrations persist to a JSON manifest and are re-applied when the agent
starts, so a source attached yesterday is still there today. Ingestion is a
governed path — file paths are validated to exist and names are sanitized —
while user-facing querying still goes through the read-only SQL guard.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .sql import Warehouse

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SUFFIX_KINDS = {
    ".csv": "csv",
    ".tsv": "csv",
    ".txt": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".json": "json",
    ".ndjson": "json",
    ".xlsx": "excel",
    ".xls": "excel",
}


class DataSourceError(ValueError):
    """Raised when a source cannot be registered or queried."""


def _quote(path: str) -> str:
    return path.replace("'", "''")


class DataSourceRegistry:
    """Registers and re-applies external data sources for one warehouse."""

    def __init__(self, warehouse: Warehouse, path: Optional[Path] = None) -> None:
        self.warehouse = warehouse
        self.path = Path(path) if path else PROJECT_ROOT / "data" / "data_sources.json"
        self.sources: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            try:
                self.sources = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.sources = {}
        self.reapply()

    # ------------------------------------------------------------------ persist
    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.sources, ensure_ascii=False, indent=2), encoding="utf-8")

    # --------------------------------------------------------------- connection
    def _con(self):
        """A read-write connection (reopening the warehouse if needed)."""
        return self.warehouse.writable_con()

    # ---------------------------------------------------------------- register
    def register_file(self, name: str, path: str, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Attach an offline spreadsheet / data file as a queryable table."""
        src = Path(path).expanduser().resolve()
        if not src.exists():
            raise DataSourceError(f"file not found: {src}")
        name = (table_name or name or src.stem).strip().lower().replace("-", "_")
        if not _NAME_RE.match(name):
            raise DataSourceError(f"invalid source name `{name}`: use letters, digits and underscores")
        kind = _SUFFIX_KINDS.get(src.suffix.lower())
        if kind is None:
            raise DataSourceError(
                f"unsupported file type `{src.suffix}` — supported: {', '.join(sorted(_SUFFIX_KINDS))}"
            )
        con = self._con()
        try:
            if kind == "csv":
                con.execute(
                    f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_csv_auto('{_quote(str(src))}', header=true)"
                )
            elif kind == "parquet":
                con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{_quote(str(src))}')")
            elif kind == "json":
                con.execute(
                    f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_json_auto('{_quote(str(src))}')"
                )
            else:  # excel: import the first sheet into a real table
                frame = pd.read_excel(src)
                if frame.empty:
                    raise DataSourceError(f"the first sheet of `{src.name}` is empty")
                con.register(f"__src_{name}", frame)
                con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM __src_{name}")
                con.unregister(f"__src_{name}")
        except DataSourceError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface a readable cause
            raise DataSourceError(f"failed to register `{src.name}`: {exc}") from exc

        info = {
            "name": name,
            "mode": "view" if kind != "excel" else "table",
            "source_kind": "file",
            "format": kind,
            "path": str(src),
            "registered_at": date.today().isoformat(),
        }
        self.sources[name] = info
        self._save()
        return self.describe(name)

    def register_database(self, name: str, uri: str) -> Dict[str, Any]:
        """ATTACH a database (duckdb file / sqlite / postgres / mysql)."""
        name = name.strip().lower().replace("-", "_")
        if not _NAME_RE.match(name):
            raise DataSourceError(f"invalid source name `{name}`")
        uri = uri.strip()
        con = self._con()
        alias = f"src_{name}"
        try:
            if uri.lower().endswith(".duckdb"):
                db_path = Path(uri).expanduser().resolve()
                if not db_path.exists():
                    raise DataSourceError(f"database file not found: {db_path}")
                con.execute(f"ATTACH '{_quote(str(db_path))}' AS {alias}")
                mode = "duckdb-attach"
            elif uri.lower().startswith("sqlite"):
                target = uri.split(":", 1)[1].strip("/") if ":" in uri else uri
                db_path = Path(target).expanduser().resolve()
                if not db_path.exists():
                    raise DataSourceError(f"sqlite file not found: {db_path}")
                con.execute("INSTALL sqlite")
                con.execute("LOAD sqlite")
                con.execute(f"ATTACH '{_quote(str(db_path))}' AS {alias} (TYPE SQLITE)")
                mode = "sqlite-attach"
            elif re.match(r"^(postgres|postgresql|mysql)://", uri, re.I):
                ext = "postgres" if uri.lower().startswith(("postgres", "postgresql")) else "mysql"
                con.execute(f"INSTALL {ext}")
                con.execute(f"LOAD {ext}")
                con.execute(f"ATTACH '{_quote(uri)}' AS {alias}")
                mode = f"{ext}-attach"
            else:
                raise DataSourceError(
                    "unsupported connection string — use a .duckdb file path, sqlite://path, or postgres://user:pass@host/db"
                )
        except DataSourceError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DataSourceError(f"failed to attach database: {exc}") from exc

        tables = self._attached_tables(alias)
        info = {
            "name": name,
            "mode": "attach",
            "source_kind": "database",
            "db_type": mode,
            "uri": uri if mode != "duckdb-attach" else str(Path(uri).expanduser().resolve()),
            "alias": alias,
            "tables": tables,
            "registered_at": date.today().isoformat(),
        }
        self.sources[name] = info
        self._save()
        return info

    def _attached_tables(self, alias: str) -> List[str]:
        try:
            rows = self._con().execute(
                "SELECT table_name FROM information_schema.tables WHERE table_catalog = ?", [alias]
            ).fetchall()
            return sorted(r[0] for r in rows)
        except Exception:  # noqa: BLE001
            return []

    # ------------------------------------------------------------------ remove
    def drop(self, name: str) -> Dict[str, Any]:
        info = self.sources.pop(name.strip().lower(), None)
        if info is None:
            raise DataSourceError(f"data source `{name}` is not registered")
        con = self._con()
        try:
            if info.get("mode") == "attach":
                con.execute(f"DETACH {info['alias']}")
            else:
                con.execute(f"DROP {'TABLE' if info['mode'] == 'table' else 'VIEW'} IF EXISTS {info['name']}")
        except Exception:  # noqa: BLE001 - removal is best effort
            pass
        self._save()
        return info

    # ------------------------------------------------------------- introspect
    def list(self) -> List[Dict[str, Any]]:
        return list(self.sources.values())

    def describe(self, name: str) -> Dict[str, Any]:
        info = self.sources.get(name.strip().lower())
        if info is None:
            raise DataSourceError(f"data source `{name}` is not registered")
        out = dict(info)
        try:
            con = self.warehouse.con
            cols = con.execute(f"DESCRIBE {info['name']}").df()
            out["columns"] = cols["column_name"].tolist()
            out["column_types"] = dict(zip(cols["column_name"], cols["column_type"].astype(str)))
            out["row_count"] = int(con.execute(f"SELECT COUNT(*) FROM {info['name']}").fetchone()[0])
            out["sample"] = con.execute(f"SELECT * FROM {info['name']} LIMIT 5").df().to_dict(orient="records")
        except Exception as exc:  # noqa: BLE001 - attached DBs may need quoting
            out["columns"] = []
            out["note"] = f"schema probe failed: {exc}"
        return out

    # ---------------------------------------------------------------- re-apply
    def reapply(self) -> None:
        """Re-register persisted sources into a fresh session (best effort)."""
        if not self.sources:
            return
        for name, info in list(self.sources.items()):
            try:
                if info.get("mode") == "attach":
                    con = self._con()
                    if info.get("db_type") == "sqlite-attach":
                        con.execute("INSTALL sqlite")
                        con.execute("LOAD sqlite")
                    con.execute(f"ATTACH '{_quote(info['uri'])}' AS {info['alias']}")
                    info["tables"] = self._attached_tables(info["alias"])
                elif info.get("mode") == "table":
                    self.warehouse.con.execute(f"SELECT 1 FROM {name} LIMIT 1")  # already there
                else:
                    try:
                        self.warehouse.con.execute(f"SELECT 1 FROM {name} LIMIT 1")
                    except Exception:  # noqa: BLE001 - view lost, rebuild it
                        self.register_file(name, info["path"])
            except Exception:  # noqa: BLE001 - a stale source must not block startup
                info["stale"] = True
        try:
            self._save()
        except OSError:
            pass
