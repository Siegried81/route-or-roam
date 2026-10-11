"""rr.checkout: a grounded-rag checkout made from its GitHub archive, offline.

The archive is built in memory and served by a fake `requests.get`, so the
tests cover the unpacking rules (one top folder becomes the destination, an
existing destination is kept, an escaping member writes nothing) without the
network the conftest blocks.
"""

from __future__ import annotations

import io
import zipfile

import pytest
import requests

from rr import checkout


class _Response:
    """The two things fetch_checkout reads from a requests response."""

    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:
        pass


def _archive(members: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, text in members.items():
            zf.writestr(name, text)
    return buf.getvalue()


def _serve(monkeypatch, content: bytes) -> list[str]:
    """Answer every GET with `content`; return the list of URLs asked for."""
    urls: list[str] = []

    def fake_get(url, timeout):
        urls.append(url)
        return _Response(content)

    monkeypatch.setattr(requests, "get", fake_get)
    return urls


def test_archive_url_names_the_ref():
    assert checkout.archive_url("https://github.com/x/grounded-rag/", "v1.2") == (
        "https://github.com/x/grounded-rag/archive/v1.2.zip"
    )


def test_fetch_checkout_unpacks_the_single_top_folder_as_dest(tmp_path, monkeypatch):
    """GitHub's `<repo>-<ref>/` folder becomes `dest` itself, so the files sit
    where a `git clone` would have put them."""
    urls = _serve(monkeypatch, _archive({
        "grounded-rag-main/rag/retrieve.py": "# retrieval",
        "grounded-rag-main/config.py": "ROOT = 1",
    }))
    dest = tmp_path / ".grounded-rag"

    assert checkout.fetch_checkout(dest, "https://github.com/x/grounded-rag", "main") == dest
    assert (dest / "rag" / "retrieve.py").read_text() == "# retrieval"
    assert (dest / "config.py").is_file()
    assert urls == ["https://github.com/x/grounded-rag/archive/main.zip"]


def test_fetch_checkout_never_replaces_an_existing_folder(tmp_path, monkeypatch):
    """A present checkout (even an empty folder) is left alone and nothing is
    downloaded: the call is safe to make on every import."""
    urls = _serve(monkeypatch, _archive({"grounded-rag-main/rag/retrieve.py": ""}))
    dest = tmp_path / ".grounded-rag"
    dest.mkdir()

    with pytest.raises(FileExistsError):
        checkout.fetch_checkout(dest, "https://github.com/x/grounded-rag", "main")
    assert urls == []


def test_fetch_checkout_rejects_an_escaping_member(tmp_path, monkeypatch):
    """A member path climbing out of the extraction folder fails before any
    file is written, and the destination is not created."""
    _serve(monkeypatch, _archive({
        "grounded-rag-main/rag/retrieve.py": "",
        "../outside.txt": "escaped",
    }))
    dest = tmp_path / ".grounded-rag"

    with pytest.raises(ValueError, match="escapes"):
        checkout.fetch_checkout(dest, "https://github.com/x/grounded-rag", "main")
    assert not dest.exists()
    assert not (tmp_path / "outside.txt").exists()


def test_fetch_checkout_needs_exactly_one_top_folder(tmp_path, monkeypatch):
    _serve(monkeypatch, _archive({"a/x.py": "", "b/y.py": ""}))

    with pytest.raises(ValueError, match="one top-level folder"):
        checkout.fetch_checkout(tmp_path / "dest", "https://github.com/x/grounded-rag", "main")
