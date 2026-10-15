#!/usr/bin/env python3
"""Download the released WAM and reasoner checkpoints."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "Atomic-WAM_RoboLab_repro/cosmos-framework/checkpoints"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main", help="Hugging Face branch, tag, or commit (default: main)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    destination = ROOT / "weights"
    target = destination / PREFIX
    link = ROOT / "cosmos-framework/checkpoints"
    if link.exists() or link.is_symlink():
        if link.resolve() != target.resolve():
            parser.error(f"{link} already points elsewhere; it will not be overwritten")
    if args.dry_run:
        print(f"Download MochunniaN1/Atomic-WAM_Robolab ({args.revision}): {PREFIX}/*")
        print(f"Destination: {destination}\nCheckpoint link: {link} -> {target}")
        return
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="MochunniaN1/Atomic-WAM_Robolab", repo_type="dataset",
                      revision=args.revision, allow_patterns=[f"{PREFIX}/*"], local_dir=destination)
    if not link.is_symlink():
        link.symlink_to(Path("../weights") / PREFIX, target_is_directory=True)
    print(f"Checkpoints ready: {link}")

if __name__ == "__main__":
    main()
