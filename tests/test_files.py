import json

import pytest
from Agent_Assistant.tooling.filesystem import digest, iter_files, resolve_in_workspace
from Agent_Assistant.tooling.rag import LocalRagIndex
from Agent_Assistant.tooling.tools import (
    ApplyPatchTool,
    ListDirTool,
    ReadFileTool,
    SearchTextTool,
    WriteFileTool,
)


def test_all_read_paths_reject_external_symlinks(tmp_path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("CANARY_BOUNDARY_TEST")
    (ws / "link.txt").symlink_to(outside)
    folder = tmp_path / "external"
    folder.mkdir()
    (folder / "private.txt").write_text("CANARY_BOUNDARY_TEST")
    (ws / "external").symlink_to(folder)
    with pytest.raises(PermissionError):
        ReadFileTool().execute({"path": "link.txt"}, ws)
    assert not LocalRagIndex(ws).search("CANARY_BOUNDARY_TEST")
    assert (
        json.loads(SearchTextTool().execute({"query": "CANARY_BOUNDARY_TEST"}, ws).content)["hits"]
        == []
    )
    assert json.loads(ListDirTool().execute({}, ws).content)["files"] == []


@pytest.mark.parametrize(
    "path", ["../outside.txt", ".env", ".git/config", ".runs/log.txt", "secret.pem"]
)
def test_protected_paths(tmp_path, path):
    with pytest.raises(PermissionError):
        resolve_in_workspace(tmp_path, path)


def test_ignore_and_deterministic_discovery(tmp_path):
    (tmp_path / "keep.py").write_text("ok")
    (tmp_path / "ignored.txt").write_text("secret")
    (tmp_path / ".agentignore").write_text("ignored*\n")
    assert [p.name for p in iter_files(tmp_path)] == [".agentignore", "keep.py"]


def test_read_tail_and_long_line_pagination(tmp_path):
    text = "x" * 9000 + "TAIL\nsecond line\n"
    (tmp_path / "large.txt").write_text(text)
    tool = ReadFileTool()
    first = json.loads(tool.execute({"path": "large.txt", "max_chars": 8000}, tmp_path).content)
    assert first["next_line"] == 1 and first["next_column"] == 8000
    second = json.loads(
        tool.execute(
            {
                "path": "large.txt",
                "start_line": first["next_line"],
                "start_column": first["next_column"],
            },
            tmp_path,
        ).content
    )
    assert "TAIL" in second["lines"][0]["text"]
    assert second["next_line"] is None
    assert first["sha256"] == digest(text.encode())
    third = json.loads(tool.execute({"path": "large.txt", "start_line": 2}, tmp_path).content)
    assert third["lines"][0]["text"] == "second line\n"


def test_patch_is_reviewable_and_rejects_stale_file(tmp_path):
    file = tmp_path / "code.py"
    file.write_text("value = 1\n")
    patch = ApplyPatchTool().prepare(
        {
            "path": "code.py",
            "old_text": "value = 1",
            "new_text": "value = 2",
            "expected_sha256": digest(file.read_bytes()),
        },
        tmp_path,
    )
    assert "+value = 2" in patch.diff
    assert file.read_text() == "value = 1\n"
    file.write_text("value = 3\n")
    with pytest.raises(ValueError, match="变化"):
        patch.commit(tmp_path)
    assert file.read_text() == "value = 3\n"


def test_patch_only_changes_unique_target(tmp_path):
    file = tmp_path / "code.py"
    file.write_text("header\nvalue = 1\nfooter\n")
    patch = ApplyPatchTool().prepare(
        {
            "path": "code.py",
            "old_text": "value = 1",
            "new_text": "value = 2",
            "expected_sha256": digest(file.read_bytes()),
        },
        tmp_path,
    )
    patch.commit(tmp_path)
    assert file.read_text() == "header\nvalue = 2\nfooter\n"
    with pytest.raises(PermissionError):
        WriteFileTool().execute({"path": "a", "content": "x"}, tmp_path)


def test_create_does_not_overwrite_existing_file(tmp_path):
    file = tmp_path / "a.txt"
    file.write_text("keep")
    with pytest.raises(ValueError, match="sha256"):
        WriteFileTool().prepare({"path": "a.txt", "content": "replace"}, tmp_path)


def test_rag_reuses_unchanged_documents_and_invalidates(tmp_path, monkeypatch):
    from Agent_Assistant.tooling import rag

    file = tmp_path / "code.py"
    file.write_text("def alpha_function(): pass\n")
    index = LocalRagIndex(tmp_path)
    reads = []
    original = rag._safe_read

    def spy(path, limit):
        reads.append(path)
        return original(path, limit)

    monkeypatch.setattr(rag, "_safe_read", spy)
    assert index.search("alpha_function")[0].line == 1
    assert index.search("alpha_function")
    assert len(reads) == 1
    file.write_text("# changed\ndef beta_function(): pass\n")
    assert not index.search("alpha_function")
    assert index.search("beta_function")[0].line == 2
    file.unlink()
    assert not index.search("beta_function")


def test_search_returns_line_evidence(tmp_path):
    (tmp_path / "auth.py").write_text("# auth\ndef validate_token():\n    return True\n")
    hits = json.loads(
        SearchTextTool().execute({"query": "validate_token", "glob": "*.py"}, tmp_path).content
    )["hits"]
    assert hits == [{"path": "auth.py", "line": 2, "text": "def validate_token():"}]


def test_read_rejects_fifo_without_blocking(tmp_path):
    import os

    os.mkfifo(tmp_path / "fifo.txt")
    with pytest.raises(ValueError, match="regular file"):
        ReadFileTool().execute({"path": "fifo.txt"}, tmp_path)
