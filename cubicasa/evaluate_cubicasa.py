"""
Evaluacion de la deteccion de planos sobre el conjunto de prueba de CubiCasa5K,
usando EXACTAMENTE el mismo modelo e inferencia que corre en produccion (AIAPI).

- NO reentrena: carga los pesos ya entrenados (weights/maskrcnn_15_epochs.h5).
- Reproduce la inferencia de application.prediction(): mold_image -> model.detect.
- Reusa la config y el mapeo de clases de application.py (1=wall,2=window,3=door).
- Calcula metricas por clase: precision, recall, AP@IoU y IoU media.
- Guarda metrics.json y 5-10 imagenes con GT (verde) vs predicciones (color).

IMPORTANTE: ejecutar en el entorno conda del modelo (imageTo3D, Py3.6/TF1.15):
  conda activate imageTo3D
  cd AIAPI/cubicasa
  python evaluate_cubicasa.py --limit 50 --num-vis 8

Requiere haber corrido antes prepare_cubicasa.py (genera out/manifest_test.json).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
AIAPI_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(AIAPI_DIR)
sys.path.insert(0, HERE)
sys.path.insert(0, AIAPI_DIR)  # para importar application.py y mrcnn

import svg_gt  # noqa: E402

# id de clase del modelo -> nombre (igual que application.getClassNames)
MODEL_ID_TO_NAME = {1: "wall", 2: "window", 3: "door"}
DEFAULT_WEIGHTS = os.path.join(AIAPI_DIR, "weights", "maskrcnn_15_epochs.h5")
DEFAULT_MANIFEST = os.path.join(HERE, "out", "manifest_test.json")
DEFAULT_OUT = os.path.join(HERE, "out")


# --------------------------------------------------------------------------- #
# Inferencia (identica a produccion)
# --------------------------------------------------------------------------- #
def load_model(weights_path):
    """Carga el Mask R-CNN tal como lo hace application.load_model()."""
    from application import PredictionConfig  # config de produccion
    from mrcnn.model import MaskRCNN

    cfg = PredictionConfig()
    model = MaskRCNN(mode="inference", model_dir=os.path.join(AIAPI_DIR, "mrcnn"), config=cfg)
    model.load_weights(weights_path, by_name=True)
    return model, cfg


def detect(model, cfg, image_path, preprocess="production"):
    """Inferencia. Devuelve (preds, pred_mask_por_clase).

    preprocess:
      'production' -> identico a application.prediction(): aplica mold_image y
                      luego model.detect (que vuelve a normalizar internamente).
      'correct'    -> pasa la imagen cruda a model.detect (normaliza UNA vez).
    preds: [{cls, bbox[x1,y1,x2,y2], score}]
    pred_mask: {clase -> mascara booleana HxW} (union de instancias de la clase)
    """
    from mrcnn.model import mold_image

    image = np.asarray(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]
    if preprocess == "correct":
        sample = [image]  # model.detect normaliza una sola vez
    else:
        sample = np.expand_dims(mold_image(image, cfg), 0)
    r = model.detect(sample, verbose=0)[0]
    masks = r.get("masks")
    preds = []
    pred_mask = {c: np.zeros((h, w), bool) for c in svg_gt.CLASSES}
    for k, (roi, cid, score) in enumerate(zip(r["rois"], r["class_ids"], r["scores"])):
        name = MODEL_ID_TO_NAME.get(int(cid))
        if name is None:
            continue
        y1, x1, y2, x2 = [float(v) for v in roi]
        preds.append({"cls": name, "bbox": [x1, y1, x2, y2], "score": float(score)})
        if masks is not None and masks.shape[-1] > k:
            pred_mask[name] |= masks[:, :, k].astype(bool)
    return preds, pred_mask


def gt_class_masks(item, w, h):
    """Rasteriza el GT (poligonos) a una mascara booleana HxW por clase."""
    from PIL import ImageDraw

    out = {}
    for c in svg_gt.CLASSES:
        canvas = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(canvas)
        for o in item["objects"]:
            if o["cls"] != c:
                continue
            for poly in o.get("polys", []):
                if len(poly) >= 3:
                    draw.polygon([tuple(p) for p in poly], fill=1)
        out[c] = np.asarray(canvas, bool)
    return out


# --------------------------------------------------------------------------- #
# Metricas (deteccion por cajas, estilo VOC/COCO @ IoU)
# --------------------------------------------------------------------------- #
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter)


def average_precision(scores, tps, n_gt):
    """AP (area bajo la curva precision-recall, interpolacion all-points)."""
    if n_gt == 0:
        return None, None, None
    if not scores:
        return 0.0, 0.0, 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tps)[order]
    fp = 1 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    # interpolacion monotona decreciente y area
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    final_p = float(precision[-1])
    final_r = float(recall[-1])
    return ap, final_p, final_r


def evaluate(items, infer_fn, iou_thr, vis_dir, num_vis, dataset_root):
    acc = {c: {"scores": [], "tp": [], "n_gt": 0, "ious": []} for c in svg_gt.CLASSES}
    # Acumuladores de pixeles (segmentacion): interseccion / union por clase.
    pix = {c: {"inter": 0, "union": 0, "gt": 0, "pred": 0} for c in svg_gt.CLASSES}
    n_imgs = len(items)
    for i, item in enumerate(items, 1):
        img_path = os.path.join(dataset_root, item["image"])
        gt = item["objects"]
        t0 = time.time()
        preds, pred_mask = infer_fn(img_path)
        ms = (time.time() - t0) * 1000

        gt_mask = gt_class_masks(item, item["width"], item["height"])

        for c in svg_gt.CLASSES:
            # ---- deteccion por instancias (bbox IoU) ----
            gt_c = [g for g in gt if g["cls"] == c]
            preds_c = sorted([p for p in preds if p["cls"] == c],
                             key=lambda p: -p["score"])
            acc[c]["n_gt"] += len(gt_c)
            matched = [False] * len(gt_c)
            for p in preds_c:
                best_iou, best_j = 0.0, -1
                for j, g in enumerate(gt_c):
                    if matched[j]:
                        continue
                    v = iou(p["bbox"], g["bbox"])
                    if v > best_iou:
                        best_iou, best_j = v, j
                is_tp = best_j >= 0 and best_iou >= iou_thr
                if is_tp:
                    matched[best_j] = True
                    acc[c]["ious"].append(best_iou)
                acc[c]["scores"].append(p["score"])
                acc[c]["tp"].append(1 if is_tp else 0)

            # ---- segmentacion por pixeles (IoU de mascaras) ----
            gm = gt_mask[c]
            pm = pred_mask.get(c)
            if pm is not None and pm.shape != gm.shape:
                pm = None  # tamano inesperado: omitir esta imagen para esta clase
            if pm is not None:
                inter = int(np.logical_and(gm, pm).sum())
                union = int(np.logical_or(gm, pm).sum())
                pix[c]["inter"] += inter
                pix[c]["union"] += union
                pix[c]["gt"] += int(gm.sum())
                pix[c]["pred"] += int(pm.sum())

        if i <= num_vis:
            svg_gt.draw_comparison(img_path, gt, preds,
                                   os.path.join(vis_dir, "vis_%02d_%s.png"
                                                % (i, os.path.basename(item["plan"]))))
        print("  [%d/%d] %s  preds=%d  gt=%d  %.0f ms"
              % (i, n_imgs, item["plan"], len(preds), len(gt), ms))

    per_class = {}
    aps = []
    pixel_ious = []
    for c in svg_gt.CLASSES:
        d = acc[c]
        ap, p, r = average_precision(d["scores"], d["tp"], d["n_gt"])
        if ap is not None:
            aps.append(ap)
        px = pix[c]
        pixel_iou = (px["inter"] / px["union"]) if px["union"] else 0.0
        pixel_prec = (px["inter"] / px["pred"]) if px["pred"] else 0.0
        pixel_rec = (px["inter"] / px["gt"]) if px["gt"] else 0.0
        pixel_ious.append(pixel_iou)
        per_class[c] = {
            "pixel_iou": pixel_iou, "pixel_precision": pixel_prec, "pixel_recall": pixel_rec,
            "bbox_AP": ap, "bbox_precision": p, "bbox_recall": r,
            "bbox_mean_iou": float(np.mean(d["ious"])) if d["ious"] else 0.0,
            "n_gt": d["n_gt"], "n_pred": len(d["scores"]), "n_tp": int(sum(d["tp"])),
        }
    return {"per_class": per_class,
            "mean_pixel_iou": float(np.mean(pixel_ious)) if pixel_ious else 0.0,
            "bbox_mAP": float(np.mean(aps)) if aps else 0.0,
            "iou_threshold": iou_thr, "images": n_imgs}


def main():
    ap = argparse.ArgumentParser(description="Evaluar la deteccion en CubiCasa5K (test).")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0, help="Maximo de imagenes (0 = todas).")
    ap.add_argument("--num-vis", type=int, default=8, help="Cuantas imagenes de ejemplo dibujar.")
    ap.add_argument("--preprocess", choices=["production", "correct"], default="production",
                    help="production = identico al endpoint; correct = normaliza una sola vez.")
    args = ap.parse_args()

    if not os.path.exists(args.manifest):
        sys.exit("Falta el manifiesto: %s. Corre antes prepare_cubicasa.py." % args.manifest)
    if not os.path.exists(args.weights):
        sys.exit("No se encuentran los pesos: %s" % args.weights)

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    items = manifest["items"]
    if args.limit:
        items = items[:args.limit]
    dataset_root = manifest["dataset"]
    vis_dir = os.path.join(args.out, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    print("Cargando el modelo (pesos: %s)..." % args.weights)
    model, cfg = load_model(args.weights)
    print("Evaluando %d imagenes (IoU>=%.2f)..." % (len(items), args.iou))

    print("Preprocesado: %s" % args.preprocess)
    results = evaluate(items, lambda p: detect(model, cfg, p, args.preprocess),
                       args.iou, vis_dir, args.num_vis, dataset_root)

    out_path = os.path.join(args.out, "metrics.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print("\n===== RESULTADOS (%d imagenes de prueba) =====" % results["images"])
    print("  -- Segmentacion por pixel (metrica principal para planos) --")
    for c in svg_gt.CLASSES:
        m = results["per_class"][c]
        print("  %-7s  IoU=%.3f  P=%.3f  R=%.3f"
              % (c, m["pixel_iou"], m["pixel_precision"], m["pixel_recall"]))
    print("  IoU media (pixel) = %.3f" % results["mean_pixel_iou"])
    print("  -- Deteccion por instancias (bbox, IoU>=%.2f; severo en muros finos) --"
          % args.iou)
    for c in svg_gt.CLASSES:
        m = results["per_class"][c]
        print("  %-7s  AP=%.3f  P=%.3f  R=%.3f  (gt=%d, pred=%d, tp=%d)"
              % (c, m["bbox_AP"] or 0, m["bbox_precision"] or 0, m["bbox_recall"] or 0,
                 m["n_gt"], m["n_pred"], m["n_tp"]))
    print("  bbox mAP@%.2f = %.3f" % (args.iou, results["bbox_mAP"]))
    print("\nMetricas -> %s" % out_path)
    print("Imagenes de ejemplo -> %s" % vis_dir)


if __name__ == "__main__":
    main()
