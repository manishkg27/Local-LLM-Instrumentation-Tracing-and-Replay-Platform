# =============================================================================
# GOOGLE COLAB SETUP — Run this cell ONLY on Google Colab
# =============================================================================
# A robust, idempotent setup script for the captcha OCR project.
# Detects the runtime, downloads the dataset from Google Drive (if needed),
# realigns the directory structure, installs dependencies, and verifies state.
#
# Safe to re-run: skips steps that are already complete, validates each stage.
# =============================================================================

from __future__ import annotations

import os
import shutil
import sys
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SetupConfig:
    """All tunable parameters in one place. Override via env vars if needed."""

    # Project location on Colab
    project_dir: Path = Path(os.environ.get("CAPTCHA_PROJECT_DIR", "/content/captcha_project"))

    # Dataset root directory name (relative to project_dir)
    data_root_name: str = os.environ.get("CAPTCHA_DATA_ROOT", "cig_ps")

    # Google Drive folder containing the dataset
    gdrive_folder_id: str = os.environ.get(
        "CAPTCHA_GDRIVE_FOLDER", "1lRUA-1uCCXfks8kpypFV-4f0UepWoLkU"
    )

    # Expected final layout
    train_dir_name: str = "train_images"
    test_dir_name: str = "test_images"
    csv_file_name: str = "train-labels.csv"

    # Minimum files needed to consider a dataset "valid" (skip re-download)
    min_train_files: int = 100
    min_test_files: int = 50

    # Output directories
    checkpoint_dir: Path = Path("models/checkpoints")
    submission_dir: Path = Path("submission")

    # Extra pip packages to install
    pip_packages: tuple[str, ...] = ("albumentations", "editdistance")


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------
class Log:
    """Tiny structured logger. Prefixes each line with an emoji for scannability."""

    @staticmethod
    def _emit(level: str, icon: str, msg: str) -> None:
        print(f"{icon} [{level}] {msg}", flush=True)

    @classmethod
    def info(cls, msg: str) -> None: cls._emit("INFO", "ℹ️ ", msg)
    @classmethod
    def ok(cls, msg: str) -> None:   cls._emit("OK",   "✅", msg)
    @classmethod
    def warn(cls, msg: str) -> None: cls._emit("WARN", "⚠️ ", msg)
    @classmethod
    def err(cls, msg: str) -> None:  cls._emit("ERROR", "❌", msg)
    @classmethod
    def step(cls, msg: str) -> None: cls._emit("STEP", "🔹", msg)


# -----------------------------------------------------------------------------
# Environment detection
# -----------------------------------------------------------------------------
def detect_colab() -> bool:
    """Reliable Colab detection: tries the official import path first."""
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


# -----------------------------------------------------------------------------
# Filesystem utilities
# -----------------------------------------------------------------------------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def count_images(directory: Optional[Path]) -> int:
    if not directory or not directory.is_dir():
        return 0
    return sum(1 for p in directory.iterdir() if p.is_file() and is_image(p))


def safe_rmtree(path: Path) -> None:
    """Remove a directory tree if it exists. Tolerates concurrent absence."""
    if path.exists():
        shutil.rmtree(path)


def safe_rmdir_empty(path: Path) -> None:
    """Remove a directory only if it exists and is empty."""
    if path.is_dir() and not any(path.iterdir()):
        path.rmdir()


def remove_empty_dirs(root: Path) -> int:
    """Recursively delete empty directories under `root`. Returns count."""
    removed = 0
    # bottom-up so children are removed before parents
    for dirpath, dirnames, _ in os.walk(root, topdown=False):
        for d in dirnames:
            p = Path(dirpath) / d
            if p.is_dir() and not any(p.iterdir()):
                p.rmdir()
                removed += 1
    return removed


def move_into_place(src: Path, dst: Path) -> None:
    """Move `src` to `dst`, replacing if needed, with clear error reporting."""
    if src == dst:
        return
    if not src.exists():
        raise FileNotFoundError(f"Source does not exist: {src}")
    if dst.exists():
        if dst.is_dir():
            safe_rmtree(dst)
        else:
            dst.unlink()
    shutil.move(str(src), str(dst))


# -----------------------------------------------------------------------------
# Stage: Workspace setup
# -----------------------------------------------------------------------------
def setup_workspace(cfg: SetupConfig) -> Path:
    """Mount Drive (if Colab), create project + output dirs. Returns data root."""
    Log.step("Setting up workspace")
    cfg.project_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(cfg.project_dir)

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.submission_dir.mkdir(parents=True, exist_ok=True)

    data_root = (cfg.project_dir / cfg.data_root_name).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    Log.ok(f"Working directory: {os.getcwd()}")
    Log.ok(f"Data root: {data_root}")
    return data_root


# -----------------------------------------------------------------------------
# Stage: Dependency install
# -----------------------------------------------------------------------------
def install_dependencies(cfg: SetupConfig) -> None:
    Log.step("Installing Python dependencies")
    # Use `pip` from the current interpreter to be safe in venv / Colab alike.
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", *cfg.pip_packages]
    import subprocess
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        Log.ok(f"Installed: {', '.join(cfg.pip_packages)}")
    except subprocess.CalledProcessError as e:
        Log.warn(f"pip install failed: {e.stderr or e.stdout}")
        raise


# -----------------------------------------------------------------------------
# Stage: Download
# -----------------------------------------------------------------------------
def install_gdown() -> None:
    try:
        import gdown  # type: ignore
    except ImportError:
        Log.info("Installing gdown...")
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "gdown"],
            check=True,
        )


def download_dataset(cfg: SetupConfig, data_root: Path) -> None:
    Log.step("Downloading dataset from Google Drive")
    install_gdown()
    import gdown  # type: ignore

    url = f"https://drive.google.com/drive/folders/{cfg.gdrive_folder_id}"
    gdown.download_folder(url, output=str(data_root), quiet=False, use_cookies=False)
    Log.ok("Download complete")


def extract_zips(directory: Path) -> int:
    """Extract any .zip files found at the top level of `directory`. Returns count."""
    zips = sorted(directory.glob("*.zip"))
    if not zips:
        return 0
    Log.step(f"Extracting {len(zips)} archive(s)")
    for zp in zips:
        Log.info(f"  • {zp.name}")
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(directory)
        zp.unlink()
    return len(zips)


# -----------------------------------------------------------------------------
# Stage: Path detection & realignment
# -----------------------------------------------------------------------------
def _name_score(name: str, keywords: tuple[str, ...]) -> int:
    name = name.lower()
    return sum(1 for k in keywords if k in name)


def find_csv(data_root: Path) -> Optional[Path]:
    candidates = list(data_root.rglob("*.csv")) + list(data_root.rglob("*.txt"))
    if not candidates:
        return None
    # Prefer names that look like labels/annotations
    priority_kw = ("label", "train", "annot", "gt")
    candidates.sort(key=lambda p: (
        0 if any(k in p.stem.lower() for k in priority_kw) else 1,
        len(p.parts),  # prefer shallower paths
    ))
    return candidates[0]


def find_image_dir(
    data_root: Path,
    *,
    prefer_keywords: tuple[str, ...],
    fallback_rank: int,
) -> Optional[Path]:
    """Find an image directory. Prefer directories whose name matches keywords;
    otherwise fall back to the Nth directory ranked by image count."""

    all_dirs = [p for p in data_root.rglob("*") if p.is_dir()]
    # Filter out any directory that would conflict with the canonical name
    all_dirs = [d for d in all_dirs if d.name.lower() not in prefer_keywords]

    matched = [
        d for d in all_dirs
        if any(k in d.name.lower() for k in prefer_keywords)
    ]
    if matched:
        # Choose the one with the most images (in case of multiple matches)
        matched.sort(key=lambda d: count_images(d), reverse=True)
        return matched[0]

    # Fallback: rank directories by image count
    all_dirs.sort(key=count_images, reverse=True)
    return all_dirs[fallback_rank] if len(all_dirs) > fallback_rank else None


def realign_layout(cfg: SetupConfig, data_root: Path) -> None:
    Log.step("Realigning dataset layout")
    target_train = data_root / cfg.train_dir_name
    target_test  = data_root / cfg.test_dir_name
    target_csv   = data_root / cfg.csv_file_name

    detected_csv       = find_csv(data_root)
    detected_train_dir = find_image_dir(
        data_root, prefer_keywords=("train",), fallback_rank=0
    )
    detected_test_dir  = find_image_dir(
        data_root, prefer_keywords=("test", "val", "valid"), fallback_rank=1
    )

    if detected_csv and detected_csv != target_csv:
        Log.info(f"  CSV: {detected_csv.relative_to(data_root)} → {target_csv.name}")
        move_into_place(detected_csv, target_csv)

    if detected_train_dir and detected_train_dir != target_train:
        Log.info(f"  Train: {detected_train_dir.relative_to(data_root)} → {target_train.name}")
        move_into_place(detected_train_dir, target_train)

    if detected_test_dir and detected_test_dir != target_test:
        Log.info(f"  Test:  {detected_test_dir.relative_to(data_root)} → {target_test.name}")
        move_into_place(detected_test_dir, target_test)

    # Tidy: drop any leftover empty directories
    removed = remove_empty_dirs(data_root)
    if removed:
        Log.info(f"Removed {removed} empty director{'y' if removed == 1 else 'ies'}")


# -----------------------------------------------------------------------------
# Stage: Validation
# -----------------------------------------------------------------------------
def validate_layout(cfg: SetupConfig, data_root: Path) -> tuple[Path, Path, Path]:
    """Verify the canonical layout is present and non-empty. Returns paths."""
    Log.step("Validating final layout")
    train_p = data_root / cfg.train_dir_name
    test_p  = data_root / cfg.test_dir_name
    csv_p   = data_root / cfg.csv_file_name

    n_train = count_images(train_p)
    n_test  = count_images(test_p)

    if not train_p.is_dir():
        raise RuntimeError(f"Missing train directory: {train_p}")
    if n_train < cfg.min_train_files:
        raise RuntimeError(
            f"Train directory has only {n_train} images "
            f"(expected ≥ {cfg.min_train_files}). Re-run with --force to redownload."
        )
    if not test_p.is_dir():
        raise RuntimeError(f"Missing test directory: {test_p}")
    if n_test < cfg.min_test_files:
        raise RuntimeError(
            f"Test directory has only {n_test} images "
            f"(expected ≥ {cfg.min_test_files})."
        )
    if not csv_p.is_file():
        raise RuntimeError(f"Missing labels CSV: {csv_p}")

    Log.ok(f"Train images: {n_train}")
    Log.ok(f"Test images:  {n_test}")
    Log.ok(f"Labels CSV:   {csv_p}")
    return train_p, test_p, csv_p


# -----------------------------------------------------------------------------
# Main orchestrator
# -----------------------------------------------------------------------------
@contextmanager
def section(title: str) -> Iterator[None]:
    print("\n" + "=" * 60, flush=True)
    print(f"  {title}", flush=True)
    print("=" * 60, flush=True)
    yield


def run_setup(force: bool = False) -> dict:
    """End-to-end setup. Returns a dict with the resolved paths.
    `force=True` will re-download and re-extract even if data already exists.
    """
    cfg = SetupConfig()
    result: dict = {"is_colab": detect_colab(), "config": cfg}

    with section("Google Colab Setup"):
        if not result["is_colab"]:
            Log.warn("Not running on Google Colab — skipping data setup.")
            Log.info("To run locally, place the dataset at "
                     f"`{(cfg.project_dir / cfg.data_root_name)}/` "
                     "with the canonical layout.")
            return result

        # 1. Mount Drive and create directories
        from google.colab import drive  # type: ignore
        Log.step("Mounting Google Drive")
        drive.mount("/content/drive")

        data_root = setup_workspace(cfg)
        result["data_root"] = data_root

        train_p = data_root / cfg.train_dir_name
        test_p  = data_root / cfg.test_dir_name
        csv_p   = data_root / cfg.csv_file_name

        layout_valid = (
            train_p.is_dir() and count_images(train_p) >= cfg.min_train_files
            and test_p.is_dir() and count_images(test_p) >= cfg.min_test_files
            and csv_p.is_file()
        )

        if layout_valid and not force:
            Log.ok("Dataset already aligned — skipping download.")
        else:
            if force and layout_valid:
                Log.warn("`force=True` — wiping existing dataset")
                safe_rmtree(train_p)
                safe_rmtree(test_p)
                if csv_p.exists():
                    csv_p.unlink()

            try:
                with section("1/4  Download"):
                    download_dataset(cfg, data_root)

                with section("2/4  Extract"):
                    extract_zips(data_root)

                with section("3/4  Realign"):
                    realign_layout(cfg, data_root)
            except Exception as e:
                Log.err(f"Setup failed: {e}")
                Log.warn("Workspace may be in a partial state. "
                         "Re-run with `force=True` to retry from scratch.")
                raise

        with section("4/4  Install & Validate"):
            install_dependencies(cfg)
            paths = validate_layout(cfg, data_root)
            result.update({
                "train_dir": paths[0],
                "test_dir":  paths[1],
                "csv_path":  paths[2],
            })

    print("\n🎉 Setup complete!\n", flush=True)
    return result


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__" or "google.colab" in sys.modules:
    # Run with `run_setup(force=True)` to force a clean re-download.
    setup_result = run_setup(force=False)
