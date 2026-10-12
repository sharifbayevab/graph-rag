"""Graph artifact pinning, backup and restore (Ruling R31 (1); M2 final review I3).

The accepted graph version and the vector caches it is rebuilt from live only in the git-ignored data/graph/.
This script pins them (a sha256 manifest committed in the repo) and backs them up outside the repo.

    python scripts/restore_artifacts.py manifest [--version V ...] [--out docs/m2/artifacts.sha256]
    python scripts/restore_artifacts.py backup   [--version V] [--dest /Users/adon/PythonProjects/grl-artifacts]
    python scripts/restore_artifacts.py verify   [--root DIR] [--manifest F]
    python scripts/restore_artifacts.py restore  ARCHIVE [--into DIR] [--manifest F] [--force]

Pinned files (paths relative to advanced-rag-v2/): data/graph/<version>/* for every listed version, and the
shared inputs data/graph/{node_vecs*, chunk_vecs*, related_*, audit.json, audit_*}, and the M3 artifacts M3_GLOBS
(data/rankings/, data/dataset/, data/gold/). The manifest is in `sha256sum` format
("<sha256>  <path>", sorted by path), so `shasum -a 256 -c` works from advanced-rag-v2/ as well.

backup writes <dest>/<version>.tar.zst (zstd when installed, else <version>.tar.gz) holding the pinned files of
that version + the shared inputs + the manifest, and <archive>.sha256 next to it. restore extracts an archive
into --into (default: this package root; existing files with different bytes are refused unless --force) and
verifies every extracted pinned file against the manifest; exit 1 on any mismatch."""
from __future__ import annotations

import hashlib
import io
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = "data/graph"
SHARED_GLOBS = ("node_vecs*", "chunk_vecs*", "related_*", "audit*")   # audit* = audit.json + audit_*
# M3 artifacts, relative to the package root (M3 Task 11): U9 rankings (<family>/latest.json and
# <family>/<version>/*), the Q-real gold and the dataset. data/rankings/*/*/* is needed for the version files.
M3_GLOBS = ("data/rankings/*/*", "data/rankings/*/*/*", "data/rankings/latest.json", "data/dataset/*", "data/gold/*")
MANIFEST = "docs/m2/artifacts.sha256"
DEFAULT_DEST = Path("/Users/adon/PythonProjects/grl-artifacts")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def latest_version(root: Path = ROOT) -> str:
    import json
    return json.loads((root / GRAPH_DIR / "latest.json").read_text(encoding="utf-8"))["version"]


def pinned_files(versions, root: Path = ROOT) -> list[str]:
    """Relative posix paths of the pinned files: every file of data/graph/<v>/ plus the shared inputs."""
    g = root / GRAPH_DIR
    out = set()
    for v in versions:
        d = g / v
        if not d.is_dir():
            raise FileNotFoundError(f"{d} is not a graph version directory")
        out |= {p.relative_to(root).as_posix() for p in d.iterdir() if p.is_file() and not p.name.endswith(".tmp")}
    for pat in SHARED_GLOBS:
        out |= {p.relative_to(root).as_posix() for p in g.glob(pat) if p.is_file() and not p.name.endswith(".tmp")}
    for pat in M3_GLOBS:
        out |= {p.relative_to(root).as_posix() for p in root.glob(pat) if p.is_file() and not p.name.endswith(".tmp")}
    return sorted(out)


def manifest_text(files, root: Path = ROOT) -> str:
    return "".join(f"{sha256_file(root / f)}  {f}\n" for f in sorted(files))


def read_manifest(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            sha, rel = line.split("  ", 1)
            out[rel] = sha
    return out


def verify(root: Path, manifest: dict[str, str], only=None) -> list[str]:
    """Problems (missing / different files) of `root` against the manifest ([] = verified)."""
    probs = []
    for rel, sha in sorted(manifest.items()):
        if only is not None and rel not in only:
            continue
        p = root / rel
        if not p.is_file():
            probs.append(f"missing {rel}")
        elif sha256_file(p) != sha:
            probs.append(f"sha256 differs: {rel}")
    return probs


def backup(version: str, dest: Path, root: Path = ROOT, manifest_path: Path | None = None) -> dict:
    """<dest>/<version>.tar.zst (+ .sha256): the version's pinned files, the shared inputs and the manifest
    (checked against the files first: a stale manifest is an error)."""
    files = pinned_files([version], root)
    mpath = manifest_path or (root / MANIFEST)
    man = read_manifest(mpath)
    missing = [f for f in files if f not in man]
    if missing:
        raise ValueError(f"{mpath} does not pin {missing[:5]} (run `manifest` first)")
    probs = verify(root, man, only=set(files))
    if probs:
        raise ValueError(f"files differ from {mpath}: {probs[:5]}")
    dest.mkdir(parents=True, exist_ok=True)
    members = files + [mpath.relative_to(root).as_posix()]
    zstd = shutil.which("zstd")
    name = f"{version}.tar.zst" if zstd else f"{version}.tar.gz"
    out = dest / name
    tmp = out.with_name(out.name + ".tmp")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for rel in members:
            info = tf.gettarinfo(str(root / rel), arcname=rel)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with open(root / rel, "rb") as f:
                tf.addfile(info, f)
    raw = buf.getvalue()
    if zstd:
        subprocess.run([zstd, "-q", "-19", "-T0", "-f", "-o", str(tmp)], input=raw, check=True)
    else:
        import gzip
        with gzip.GzipFile(tmp, "wb", mtime=0) as f:
            f.write(raw)
    tmp.replace(out)
    sha = sha256_file(out)
    (dest / (out.name + ".sha256")).write_text(f"{sha}  {out.name}\n", encoding="utf-8")
    return {"archive": str(out), "bytes": out.stat().st_size, "sha256": sha, "files": len(members)}


def _open_tar(archive: Path) -> tarfile.TarFile:
    if archive.name.endswith(".zst"):
        zstd = shutil.which("zstd")
        if not zstd:
            raise RuntimeError("zstd is not installed: cannot read a .tar.zst archive")
        raw = subprocess.run([zstd, "-q", "-d", "-c", str(archive)], capture_output=True, check=True).stdout
        return tarfile.open(fileobj=io.BytesIO(raw), mode="r:")
    return tarfile.open(archive, mode="r:*")


def restore(archive: Path, into: Path, manifest_path: Path | None = None, force: bool = False) -> dict:
    """Extract `archive` into `into` and verify every extracted pinned file against the manifest (the one in
    the archive unless `manifest_path` is given). The archive's own .sha256 is checked first when present."""
    side = archive.with_name(archive.name + ".sha256")
    if side.exists():
        want = side.read_text(encoding="utf-8").split()[0]
        if sha256_file(archive) != want:
            raise ValueError(f"{archive.name}: sha256 differs from {side.name}")
    with _open_tar(archive) as tf:
        members = tf.getmembers()
        for m in members:
            if not m.isfile() or m.name.startswith("/") or ".." in Path(m.name).parts:
                raise ValueError(f"refusing archive member {m.name!r}")
        clash = []
        for m in members:
            p = into / m.name
            if p.exists() and not force:
                data = tf.extractfile(m).read()
                if hashlib.sha256(data).hexdigest() != sha256_file(p):
                    clash.append(m.name)
        if clash:
            raise FileExistsError(f"{into} holds different {clash[:5]} (use --force to overwrite)")
        for m in members:
            p = into / m.name
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + ".tmp")
            tmp.write_bytes(tf.extractfile(m).read())
            tmp.replace(p)
        names = [m.name for m in members]
    man = read_manifest(manifest_path or (into / MANIFEST))
    pinned = {n for n in names if n in man}
    probs = verify(into, man, only=pinned)
    return {"extracted": len(names), "verified": len(pinned) - len(probs), "problems": probs}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--version", action="append")
    m.add_argument("--out", default=MANIFEST)
    b = sub.add_parser("backup")
    b.add_argument("--version")
    b.add_argument("--dest", default=str(DEFAULT_DEST))
    v = sub.add_parser("verify")
    v.add_argument("--root", default=str(ROOT))
    v.add_argument("--manifest")
    r = sub.add_parser("restore")
    r.add_argument("archive")
    r.add_argument("--into", default=str(ROOT))
    r.add_argument("--manifest")
    r.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "manifest":
        versions = a.version or [latest_version()]
        files = pinned_files(versions)
        out = ROOT / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(manifest_text(files), encoding="utf-8")
        print(f"{out.relative_to(ROOT)}: {len(files)} files, versions {versions}")
        return 0
    if a.cmd == "backup":
        res = backup(a.version or latest_version(), Path(a.dest))
        print(res)
        return 0
    if a.cmd == "verify":
        root = Path(a.root)
        probs = verify(root, read_manifest(Path(a.manifest) if a.manifest else root / MANIFEST))
        print("\n".join(probs) if probs else "verified: every pinned file matches the manifest")
        return 1 if probs else 0
    res = restore(Path(a.archive), Path(a.into), Path(a.manifest) if a.manifest else None, a.force)
    print(res)
    return 1 if res["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
