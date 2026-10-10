"""Fetch a grounded-rag checkout from its GitHub archive when none is on disk.

A platform that clones only this repository (Streamlit Community Cloud) has no
sibling grounded-rag folder and no step to clone one, while rr/grounded.py
needs that folder on sys.path before anything else imports. Downloading the
archive of a branch, tag or commit once at first import is the smallest thing
that gives such a platform the same checkout the Dockerfile clones.
"""

from __future__ import annotations

import io
import shutil
import tempfile
import zipfile
from pathlib import Path

import requests

_TIMEOUT_S = 120


def archive_url(repo: str, ref: str) -> str:
    """GitHub's zip archive of a ref: a branch name, a tag or a commit sha."""
    return f"{repo.rstrip('/')}/archive/{ref}.zip"


def fetch_checkout(dest: Path, repo: str, ref: str) -> Path:
    """Download the archive of `repo` at `ref` and unpack its single top folder as `dest`.

    GitHub archives hold one top-level folder named after the repo and ref;
    its contents become `dest`, so `dest/rag/retrieve.py` exists afterwards
    exactly as it would after `git clone`. Every member path is checked to
    stay under the extraction folder before anything is written, so an
    archive with `..` entries writes nothing. `dest` must not exist: a present
    checkout is never replaced, which is what makes the call safe to repeat.
    """
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(f"{dest} already exists; refusing to replace a checkout")
    resp = requests.get(archive_url(repo, ref), timeout=_TIMEOUT_S)
    resp.raise_for_status()

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name).resolve()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
            for member in archive.namelist():
                if not (tmp / member).resolve().is_relative_to(tmp):
                    raise ValueError(f"archive member escapes the checkout folder: {member!r}")
            archive.extractall(tmp)
        tops = [p for p in tmp.iterdir() if p.is_dir()]
        if len(tops) != 1:
            raise ValueError(f"expected one top-level folder in the archive, found {len(tops)}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tops[0]), str(dest))
    return dest
