#!/usr/bin/env python3
"""Download a small, reproducibly manifested subset of a Hugging Face dataset."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Callable, Iterable, Sequence

from huggingface_hub import HfApi, hf_hub_download


DownloadFile = Callable[[str, str, str], Path]


def _natural_sort_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def select_shards(
    repo_files: Iterable[str], includes: Sequence[str], max_shards: int
) -> list[str]:
    if max_shards <= 0:
        raise ValueError("max_shards must be positive")
    if not includes:
        raise ValueError("at least one include pattern is required")

    repo_files = list(repo_files)
    per_pattern = [
        sorted(
            {
                filename
                for filename in repo_files
                if fnmatch.fnmatch(filename, pattern)
            },
            key=_natural_sort_key,
        )
        for pattern in includes
    ]
    if not any(per_pattern):
        raise ValueError(f"no repository files matched include patterns: {list(includes)}")

    selected: list[str] = []
    seen: set[str] = set()
    round_index = 0
    while len(selected) < max_shards:
        added = False
        for candidates in per_pattern:
            if round_index >= len(candidates):
                continue
            filename = candidates[round_index]
            if filename not in seen:
                seen.add(filename)
                selected.append(filename)
                added = True
                if len(selected) == max_shards:
                    break
        if not added and all(round_index + 1 >= len(items) for items in per_pattern):
            break
        round_index += 1
    return selected


def _static_directory_prefix(pattern: str) -> str | None:
    wildcard_positions = [
        position for token in "*[?" if (position := pattern.find(token)) >= 0
    ]
    boundary = min(wildcard_positions) if wildcard_positions else len(pattern)
    static_part = pattern[:boundary]
    if "/" not in static_part:
        return None
    prefix = static_part.rstrip("/") if static_part.endswith("/") else static_part.rsplit("/", 1)[0]
    return prefix or None


def _list_matching_boundaries(
    api: object,
    repo: str,
    resolved_revision: str,
    includes: Sequence[str],
    per_prefix_limit: int,
) -> list[str]:
    prefixes = [_static_directory_prefix(pattern) for pattern in includes]
    can_use_prefixes = all(prefixes) and hasattr(api, "list_repo_tree")
    if not can_use_prefixes:
        return list(
            api.list_repo_files(
                repo, repo_type="dataset", revision=resolved_revision
            )
        )

    files: list[str] = []
    visited: set[str] = set()
    for prefix in prefixes:
        assert prefix is not None
        if prefix in visited:
            continue
        visited.add(prefix)
        entries = api.list_repo_tree(
            repo,
            path_in_repo=prefix,
            recursive=False,
            expand=False,
            repo_type="dataset",
            revision=resolved_revision,
        )
        file_count = 0
        for entry in entries:
            is_file = getattr(entry, "type", None) == "file" or hasattr(entry, "size")
            if not is_file:
                continue
            files.append(entry.path)
            file_count += 1
            if file_count == per_prefix_limit:
                break
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matching_prefix_length(source: Path, partial: Path) -> int:
    if not partial.exists():
        return 0
    source_size = source.stat().st_size
    partial_size = partial.stat().st_size
    if partial_size > source_size:
        return 0

    remaining = partial_size
    with source.open("rb") as src, partial.open("rb") as dst:
        while remaining:
            block_size = min(8 * 1024 * 1024, remaining)
            if src.read(block_size) != dst.read(block_size):
                return 0
            remaining -= block_size
    return partial_size


def _resume_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    offset = _matching_prefix_length(source, destination)
    mode = "ab" if offset else "wb"
    with source.open("rb") as src, destination.open(mode) as dst:
        src.seek(offset)
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())


def _default_download_file(repo: str, revision: str, filename: str) -> Path:
    resolver_endpoint = os.environ.get("DREAM_HF_DOWNLOAD_ENDPOINT")
    return Path(
        hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            endpoint=resolver_endpoint,
        )
    )


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f"{path.name}.incomplete")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_existing_manifest(
    manifest_path: Path, repo: str, resolved_revision: str
) -> None:
    if not manifest_path.exists():
        return
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    if existing.get("repo") != repo:
        raise ValueError(
            f"existing manifest repo {existing.get('repo')!r} does not match {repo!r}"
        )
    if existing.get("resolved_revision") != resolved_revision:
        raise ValueError(
            "existing manifest revision "
            f"{existing.get('resolved_revision')!r} does not match {resolved_revision!r}"
        )


def download_subset(
    *,
    repo: str,
    revision: str,
    output_dir: Path | str,
    max_shards: int,
    includes: Sequence[str],
    api: HfApi | object | None = None,
    download_file: DownloadFile | None = None,
    list_only: bool = False,
) -> dict[str, object]:
    api = api or HfApi()
    download_file = download_file or _default_download_file
    output_dir = Path(output_dir)

    info = api.repo_info(repo, repo_type="dataset", revision=revision)
    resolved_revision = info.sha
    if not resolved_revision:
        raise RuntimeError(f"Hugging Face did not resolve a commit for {repo}@{revision}")
    repo_files = _list_matching_boundaries(
        api, repo, resolved_revision, includes, max_shards
    )
    selected = select_shards(repo_files, includes, max_shards)

    if list_only:
        return {
            "repo": repo,
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "selected_shards": selected,
        }

    manifest_path = output_dir / "source_manifest.json"
    _validate_existing_manifest(manifest_path, repo, resolved_revision)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, object]] = []
    for filename in selected:
        final_path = output_dir / filename
        incomplete_path = final_path.with_name(f"{final_path.name}.incomplete")
        if not final_path.exists():
            cached_path = Path(download_file(repo, resolved_revision, filename))
            if not cached_path.is_file():
                raise FileNotFoundError(
                    f"download boundary returned no file for {filename}: {cached_path}"
                )
            _resume_copy(cached_path, incomplete_path)
            if incomplete_path.stat().st_size != cached_path.stat().st_size:
                raise IOError(f"incomplete byte count after copying {filename}")
            os.replace(incomplete_path, final_path)

        entries.append(
            {
                "path": filename,
                "bytes": final_path.stat().st_size,
                "sha256": _sha256(final_path),
            }
        )

    manifest: dict[str, object] = {
        "format_version": 1,
        "repo": repo,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "include_patterns": list(includes),
        "max_shards": max_shards,
        "shards": entries,
        "total_bytes": sum(int(item["bytes"]) for item in entries),
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_shards", type=int, required=True)
    parser.add_argument(
        "--include",
        dest="includes",
        action="append",
        default=None,
        help="fnmatch pattern; repeat for multiple corpus components",
    )
    parser.add_argument("--list_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = download_subset(
        repo=args.repo,
        revision=args.revision,
        output_dir=args.output_dir,
        max_shards=args.max_shards,
        includes=args.includes or ["*.jsonl.zst"],
        list_only=args.list_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
