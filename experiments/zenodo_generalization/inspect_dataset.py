import argparse
import json
import os

from core.dataset import create_dataloaders, dataset_export_manifest, dataset_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="./401")
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()

    exp_dir = os.path.abspath(args.exp_dir)
    with open(os.path.join(exp_dir, "config.json"), "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    if not os.path.isabs(cfg["data"]["data_dir"]):
        cfg["data"]["data_dir"] = os.path.abspath(os.path.join(exp_dir, cfg["data"]["data_dir"]))

    out_dir = args.out_dir or os.path.join(exp_dir, "logs")
    os.makedirs(out_dir, exist_ok=True)

    _, _, _, train_ds, val_ds, test_ds = create_dataloaders(cfg, return_datasets=True)
    print(dataset_summary(train_ds, split_name="train"))
    print(dataset_summary(val_ds, split_name="val"))
    print(dataset_summary(test_ds, split_name="test"))
    dataset_export_manifest(train_ds, os.path.join(out_dir, "manifest_train.csv"), split_name="train")
    dataset_export_manifest(val_ds, os.path.join(out_dir, "manifest_val.csv"), split_name="val")
    dataset_export_manifest(test_ds, os.path.join(out_dir, "manifest_test.csv"), split_name="test")


if __name__ == "__main__":
    main()
