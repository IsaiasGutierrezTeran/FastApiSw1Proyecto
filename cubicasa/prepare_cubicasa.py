"""
Preparacion del dataset CubiCasa5K para validar la deteccion de planos.

Que hace:
  - Recorre el dataset y usa el SPLIT OFICIAL que ya trae (train.txt / val.txt /
    test.txt). CubiCasa5K viene dividido; reusar su split es lo correcto y lo
    reproducible (no hay aleatoriedad). Con --resplit se puede forzar un 80/20
    propio con semilla fija si se desea.
  - Convierte las anotaciones model.svg a cajas de 3 clases: wall/door/window
    (en el espacio de pixeles de F1_scaled.png).
  - Genera un manifiesto JSON por split con la lista de imagenes y su GT.

Uso (Python con Pillow; no necesita TensorFlow):
  python prepare_cubicasa.py
  python prepare_cubicasa.py --splits train.txt test.txt --limit 50
  python prepare_cubicasa.py --resplit --ratio 0.8 --seed 42

Salidas (por defecto en cubicasa/out/):
  manifest_train.json, manifest_test.json
"""

import argparse
import json
import os
import random
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import svg_gt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
AIAPI_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(AIAPI_DIR)
DEFAULT_DATASET = os.path.join(ROOT, "BACKEND", "data", "cubicasa5k", "cubicasa5k")
DEFAULT_OUT = os.path.join(HERE, "out")


def build_manifest(dataset_root, plans, limit=0):
    """Para cada plano: tamano de imagen + GT (wall/door/window) en px."""
    items = []
    counts = {c: 0 for c in svg_gt.CLASSES}
    skipped = 0
    if limit:
        plans = plans[:limit]
    for i, rel in enumerate(plans, 1):
        img_path, svg_path = svg_gt.plan_files(dataset_root, rel)
        if not (os.path.exists(img_path) and os.path.exists(svg_path)):
            skipped += 1
            continue
        try:
            with Image.open(img_path) as im:
                w, h = im.size
            gt = svg_gt.gt_for_image(svg_path, w, h)
        except Exception as exc:  # noqa: BLE001 — un plano corrupto no debe abortar todo
            print("  [skip] %s: %s" % (rel, exc))
            skipped += 1
            continue
        for g in gt:
            counts[g["cls"]] += 1
        items.append({
            "plan": rel,
            "image": os.path.relpath(img_path, dataset_root).replace("\\", "/"),
            "width": w, "height": h,
            "objects": gt,
        })
        if i % 50 == 0:
            print("  ...%d/%d planos" % (i, len(plans)))
    return items, counts, skipped


def reproducible_split(dataset_root, ratio, seed):
    """80/20 propio con semilla fija (solo si se pide --resplit)."""
    all_plans = []
    for split in ("train.txt", "val.txt", "test.txt"):
        p = os.path.join(dataset_root, split)
        if os.path.exists(p):
            all_plans.extend(svg_gt.read_split(dataset_root, split))
    all_plans = sorted(set(all_plans))
    rng = random.Random(seed)
    rng.shuffle(all_plans)
    cut = int(len(all_plans) * ratio)
    return {"train.txt": all_plans[:cut], "test.txt": all_plans[cut:]}


def main():
    ap = argparse.ArgumentParser(description="Preparar CubiCasa5K (manifiesto GT).")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--splits", nargs="+", default=["train.txt", "test.txt"],
                    help="Splits oficiales a procesar (por defecto train y test).")
    ap.add_argument("--limit", type=int, default=0, help="Maximo de planos por split (0 = todos).")
    ap.add_argument("--resplit", action="store_true",
                    help="Ignorar el split oficial y hacer 80/20 reproducible.")
    ap.add_argument("--ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.isdir(args.dataset):
        sys.exit("No existe el dataset: %s" % args.dataset)
    os.makedirs(args.out, exist_ok=True)

    if args.resplit:
        print("Split 80/20 reproducible (semilla=%d)." % args.seed)
        split_plans = reproducible_split(args.dataset, args.ratio, args.seed)
    else:
        print("Usando el split OFICIAL de CubiCasa5K.")
        split_plans = {s: svg_gt.read_split(args.dataset, s) for s in args.splits}

    for split, plans in split_plans.items():
        name = os.path.splitext(split)[0]  # train / test / val
        print("\n== Split '%s': %d planos ==" % (name, len(plans)))
        items, counts, skipped = build_manifest(args.dataset, plans, args.limit)
        out_path = os.path.join(args.out, "manifest_%s.json" % name)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"dataset": args.dataset, "split": name,
                       "count": len(items), "class_counts": counts,
                       "items": items}, fh)
        print("  GT por clase: %s" % counts)
        print("  Planos validos: %d (saltados: %d)" % (len(items), skipped))
        print("  -> %s" % out_path)


if __name__ == "__main__":
    main()
