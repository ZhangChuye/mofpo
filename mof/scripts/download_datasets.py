"""Download MoF demonstration datasets from the HuggingFace Hub into ``data/``.

The files land exactly where the task configs expect them, so after downloading
you can train directly with ``task=<task>`` (no path overrides):

    data/bigym/<task>/*.safetensors          (BiGym RBY1)
    data/dexmimicgen/<task>_abs.hdf5          (DexMimicGen)

Examples
--------
    # all 9 tasks
    python -m mof.scripts.download_datasets

    # just the BiGym (or just the DexMimicGen) tasks
    python -m mof.scripts.download_datasets --tasks bigym
    python -m mof.scripts.download_datasets --tasks dex

    # specific tasks (by task_name)
    python -m mof.scripts.download_datasets --tasks rby1_flip_cup two_arm_threading

    # list everything without downloading
    python -m mof.scripts.download_datasets --list
"""
import argparse
from huggingface_hub import snapshot_download

REPO_ID = "dian-wang/mof-datasets"

# task_name -> (repo path stem, matching `task=` config to train with)
BIGYM = {
    "rby1_flip_cup":               ("bigym/rby1_flip_cup",               "bigym_rby1_flip_cup"),
    "rby1_move_two_plates":        ("bigym/rby1_move_two_plates",        "bigym_rby1_move_two_plates"),
    "rby1_store_kitchenware":      ("bigym/rby1_store_kitchenware",      "bigym_rby1_store_kitchenware"),
    "rby1_flip_sandwich":          ("bigym/rby1_flip_sandwich",          "bigym_rby1_flip_sandwich"),
    "rby1_dishwasher_load_plates": ("bigym/rby1_dishwasher_load_plates", "bigym_rby1_dishwasher_load_plates"),
}
DEX = {
    "two_arm_threading":            ("dexmimicgen/two_arm_threading_abs",            "dexmimicgen_two_arm_threading"),
    "two_arm_three_piece_assembly": ("dexmimicgen/two_arm_three_piece_assembly_abs", "dexmimicgen_two_arm_three_piece_assembly"),
    "two_arm_box_cleanup":          ("dexmimicgen/two_arm_box_cleanup_abs",          "dexmimicgen_two_arm_box_cleanup"),
    "two_arm_drawer_cleanup":       ("dexmimicgen/two_arm_drawer_cleanup_abs",       "dexmimicgen_two_arm_drawer_cleanup"),
}
ALL = {**BIGYM, **DEX}


def allow_patterns(task):
    stem = ALL[task][0]
    # BiGym is a directory of per-demo safetensors; DexMimicGen is a single hdf5.
    return [f"{stem}/*"] if task in BIGYM else [f"{stem}.hdf5"]


def resolve(selectors):
    sel = []
    for s in selectors:
        if s == "all":
            sel = list(ALL); break
        elif s == "bigym":
            sel += list(BIGYM)
        elif s == "dex":
            sel += list(DEX)
        elif s in ALL:
            sel.append(s)
        else:
            raise SystemExit(
                f"Unknown task '{s}'. Use --list to see options, or 'all'/'bigym'/'dex'.")
    return list(dict.fromkeys(sel))  # dedup, keep order


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=["all"],
                    metavar="TASK", help="'all' (default), 'bigym', 'dex', or task_name(s)")
    ap.add_argument("--data-dir", default="data",
                    help="destination root (default: data/, where the configs look)")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--list", action="store_true", help="list available tasks and exit")
    a = ap.parse_args()

    if a.list:
        print(f"Available datasets in {a.repo_id}:\n")
        for grp, d in (("BiGym (RBY1)", BIGYM), ("DexMimicGen", DEX)):
            print(f"  {grp}:")
            for t, (stem, cfg) in d.items():
                print(f"    {t:32s} -> data/{stem}   (train with task={cfg})")
        return

    tasks = resolve(a.tasks)
    patterns = [p for t in tasks for p in allow_patterns(t)]
    print(f"Downloading {len(tasks)} dataset(s) from {a.repo_id} into {a.data_dir}/ :")
    for t in tasks:
        print(f"  - {t}  ->  {a.data_dir}/{ALL[t][0]}")
    snapshot_download(repo_id=a.repo_id, repo_type="dataset",
                      local_dir=a.data_dir, allow_patterns=patterns)
    print("\nDone. Train directly, e.g.:")
    print(f"  ./run_async_eval.sh train.py --config-name=train_mof_moe task={ALL[tasks[0]][1]}")


if __name__ == "__main__":
    main()
