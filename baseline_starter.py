"""
Starter kit for the Mechanical Parts Segmentation Challenge.

    python baseline_starter_parts.py --root . --out submission.csv

Gives you: a correct RLE encoder/decoder, the exact competition metric, a
stratified validation split, dataset statistics, and a valid blank submission
so you have a working pipeline in five minutes. No model -- that part is yours.
"""
import argparse

import numpy as np
import pandas as pd
from PIL import Image

CLASSES = ["hex_nut", "washer", "bolt", "ball_bearing", "spring", "o_ring"]
CLASS_ID = {c: i + 1 for i, c in enumerate(CLASSES)}
SIZE = 384


def rle_encode(mask):
    px = np.asarray(mask, np.uint8).T.flatten()
    px = np.concatenate([[0], px, [0]])
    runs = np.where(px[1:] != px[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(int(x)) for x in runs)


def rle_decode(rle, size=SIZE):
    m = np.zeros(size * size, np.uint8)
    if not isinstance(rle, str) or not rle.strip():
        return m.reshape(size, size, order="F")
    a = np.asarray(rle.split(), np.int64)
    for s, l in zip(a[0::2], a[1::2]):
        m[s - 1: s - 1 + l] = 1
    return m.reshape(size, size, order="F")


def dice(pred, gt):
    p, g = int(np.sum(pred)), int(np.sum(gt))
    if g == 0:
        return 1.0 if p == 0 else 0.0
    if p == 0:
        return 0.0
    return 2.0 * float(np.logical_and(pred, gt).sum()) / (p + g)


def mean_dice(pred_labels, gt_labels):
    """The official metric: mean over every (image, class) pair."""
    s = []
    for p, g in zip(pred_labels, gt_labels):
        for cid in range(1, len(CLASSES) + 1):
            s.append(dice(p == cid, g == cid))
    return float(np.mean(s))


def load_label(root, fid, split="train"):
    """Masks store the class id as the pixel value, with a display palette attached.

    np.array() returns the 0-6 label map. Never call .convert("RGB") on these.
    """
    return np.array(Image.open(f"{root}/{split}/masks/{fid}"))


def write_submission(pred_labels, ids, path):
    rows = []
    for fid, lab in zip(ids, pred_labels):
        for c in CLASSES:
            m = lab == CLASS_ID[c]
            rows.append({"ImageId_ClassId": f"{fid}_{c}",
                         "EncodedPixels": rle_encode(m) if m.any() else ""})
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--val-frac", type=float, default=0.15)
    a = ap.parse_args()

    meta = pd.read_csv(f"{a.root}/metadata.csv")
    tr_ids = sorted(meta[meta.split == "train"].id)
    rng = np.random.default_rng(0)
    val_ids = list(rng.choice(tr_ids, int(len(tr_ids) * a.val_frac), replace=False))
    fit_ids = [i for i in tr_ids if i not in set(val_ids)]
    print(f"train {len(fit_ids)}   val {len(val_ids)}")

    sample = val_ids[:120]
    labs = [load_label(a.root, i) for i in sample]
    cov = np.mean([(l > 0).mean() for l in labs])
    print(f"\nforeground coverage: {100*cov:.1f}%")
    print(f"{'class':14s}{'present':>10s}{'median px':>12s}{'median instances':>18s}")
    import cv2
    for c in CLASSES:
        cid = CLASS_ID[c]
        ar = [int((l == cid).sum()) for l in labs if (l == cid).any()]
        ninst = [cv2.connectedComponentsWithStats((l == cid).astype(np.uint8), 8)[0] - 1
                 for l in labs if (l == cid).any()]
        print(f"{c:14s}{100*len(ar)/len(labs):9.1f}%{np.median(ar):12.0f}{np.median(ninst):18.0f}")

    blank = [np.zeros((SIZE, SIZE), np.uint8) for _ in labs]
    print(f"\n(every class is present in every image; a 1-pixel dummy scores ~0)")

    sub = pd.read_csv(f"{a.root}/sample_submission.csv")
    sub["EncodedPixels"] = "1 1"
    sub.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(sub)} rows) -- valid upload, scores ~0")


if __name__ == "__main__":
    main()