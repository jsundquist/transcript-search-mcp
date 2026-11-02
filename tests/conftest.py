from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from transcript_search_v2 import config, db

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sample_transcripts"


@pytest.fixture
def tmp_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    watch_root = tmp_path / "watch"
    watch_root.mkdir()
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "DB_PATH", data_dir / "index.db")
    monkeypatch.setattr(config, "WATCH_ROOT", watch_root)
    return {"data_dir": data_dir, "watch_root": watch_root}


@pytest.fixture
def conn(tmp_env):
    c = db.connect()
    db.init_db(c)
    yield c
    c.close()


def copy_fixture(
    watch_root: Path, fixture_name: str, project_dir: str = "-tmp-fixture-proj"
) -> Path:
    """Copy a fixture .jsonl into watch_root/<project_dir>/<sessionId>.jsonl,
    simulating a session file discovered under ~/.claude/projects/ -- the
    real filename is always the session UUID (matching the sessionId field
    inside), so the destination name is derived from the fixture's own first
    line rather than reusing the fixture's on-disk filename."""
    src = FIXTURES_DIR / fixture_name
    session_id = Path(fixture_name).stem
    with open(src) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if "sessionId" in record:
                session_id = record["sessionId"]
                break

    dest_dir = watch_root / project_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{session_id}.jsonl"
    shutil.copy(src, dest)
    return dest
