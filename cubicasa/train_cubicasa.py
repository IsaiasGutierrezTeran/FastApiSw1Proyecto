"""
Reentrenamiento (fine-tuning) del Mask R-CNN con el 80% de CubiCasa5K (train.txt).

- Parte de los pesos ya entrenados (maskrcnn_15_epochs.h5) y sigue entrenando para
  mejorar las metricas, o desde COCO si se indica.
- Usa los manifiestos generados por prepare_cubicasa.py (objetos + poligonos) para
  rasterizar las mascaras de instancia de cada plano (wall/window/door).
- Las clases mantienen el MISMO mapeo que produccion (1=wall, 2=window, 3=door),
  para que los pesos resultantes sirvan tal cual en application.py.

IMPORTANTE: ejecutar en el entorno conda del modelo (imageTo3D, Py3.6/TF1.15).
Entrenar el modelo completo es PESADO: requiere GPU y varias horas. En CPU sirve
solo para pruebas chicas (--smoke).

Pasos:
  1) Generar los manifiestos (una vez):
       python prepare_cubicasa.py --splits train.txt val.txt
  2) Entrenar:
       conda activate imageTo3D
       cd AIAPI/cubicasa
       python train_cubicasa.py --epochs 20 --layers heads          # cabezas
       python train_cubicasa.py --epochs 40 --layers all --lr 0.0005 # afinado total
  3) Los pesos quedan en out/logs/.../mask_rcnn_floorplan_cfg_*.h5.
     Para usarlos en produccion, copiar el .h5 elegido a AIAPI/weights/.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
AIAPI_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, AIAPI_DIR)  # para importar mrcnn

import svg_gt  # noqa: E402
from mrcnn import model as modellib  # noqa: E402
from mrcnn import utils  # noqa: E402
from mrcnn.config import Config  # noqa: E402

# Mismo orden/ids que application.getClassNames: 1=wall, 2=window, 3=door
CLASS_ORDER = [("wall", 1), ("window", 2), ("door", 3)]
NAME_TO_ID = dict(CLASS_ORDER)

DEFAULT_TRAIN = os.path.join(HERE, "out", "manifest_train.json")
DEFAULT_VAL = os.path.join(HERE, "out", "manifest_val.json")
DEFAULT_WEIGHTS = os.path.join(AIAPI_DIR, "weights", "maskrcnn_15_epochs.h5")
LOGS_DIR = os.path.join(HERE, "out", "logs")


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CubicasaDataset(utils.Dataset):
    """Carga imagenes + mascaras de instancia desde un manifiesto de prepare."""

    def load_from_manifest(self, manifest_path, limit=0):
        manifest = json.load(open(manifest_path, encoding="utf-8"))
        root = manifest["dataset"]
        for name, cid in CLASS_ORDER:
            self.add_class("cubicasa", cid, name)
        items = manifest["items"]
        if limit:
            items = items[:limit]
        for it in items:
            objs = [o for o in it["objects"] if o.get("polys")]
            if not objs:
                continue  # sin instancias utiles: se omite
            self.add_image(
                "cubicasa", image_id=it["plan"],
                path=os.path.join(root, it["image"]),
                width=it["width"], height=it["height"], objects=objs,
            )

    def load_image(self, image_id):
        path = self.image_info[image_id]["path"]
        return np.asarray(Image.open(path).convert("RGB"))

    def image_reference(self, image_id):
        return self.image_info[image_id]["id"]

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        w, h = info["width"], info["height"]
        masks, class_ids = [], []
        for o in info["objects"]:
            canvas = Image.new("L", (w, h), 0)
            draw = ImageDraw.Draw(canvas)
            drew = False
            for poly in o["polys"]:
                if len(poly) >= 3:
                    draw.polygon([tuple(p) for p in poly], fill=1)
                    drew = True
            if not drew:
                continue
            masks.append(np.asarray(canvas, bool))
            class_ids.append(NAME_TO_ID[o["cls"]])
        if not masks:
            return np.zeros((h, w, 0), bool), np.zeros((0,), np.int32)
        return np.stack(masks, axis=-1), np.asarray(class_ids, np.int32)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class TrainConfig(Config):
    NAME = "floorPlan_cfg"
    NUM_CLASSES = 1 + 3            # fondo + wall/window/door
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 500
    VALIDATION_STEPS = 50
    LEARNING_RATE = 0.001
    DETECTION_MIN_CONFIDENCE = 0.7


def main():
    ap = argparse.ArgumentParser(description="Reentrenar Mask R-CNN con CubiCasa5K (80%).")
    ap.add_argument("--train-manifest", default=DEFAULT_TRAIN)
    ap.add_argument("--val-manifest", default=DEFAULT_VAL)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS,
                    help="Pesos iniciales: ruta .h5, 'coco', o 'last'.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--layers", default="heads", choices=["heads", "all", "3+", "4+", "5+"])
    ap.add_argument("--lr", type=float, default=None, help="Learning rate (sobreescribe).")
    ap.add_argument("--steps", type=int, default=None, help="STEPS_PER_EPOCH (sobreescribe).")
    ap.add_argument("--limit", type=int, default=0, help="Limitar planos (pruebas).")
    ap.add_argument("--smoke", action="store_true",
                    help="Prueba minima: 1 epoca, 1 paso, pocos planos (verifica que corre).")
    args = ap.parse_args()

    cfg = TrainConfig()
    if args.lr:
        cfg.LEARNING_RATE = args.lr
    if args.steps:
        cfg.STEPS_PER_EPOCH = args.steps
    if args.smoke:
        cfg.STEPS_PER_EPOCH = 1
        cfg.VALIDATION_STEPS = 1
        args.epochs = 1
        if not args.limit:
            args.limit = 4
    cfg.display()

    for p in (args.train_manifest, args.val_manifest):
        if not os.path.exists(p):
            sys.exit("Falta el manifiesto %s. Corre antes: prepare_cubicasa.py "
                     "--splits train.txt val.txt" % p)

    train_set = CubicasaDataset()
    train_set.load_from_manifest(args.train_manifest, args.limit)
    train_set.prepare()
    val_set = CubicasaDataset()
    val_set.load_from_manifest(args.val_manifest, args.limit)
    val_set.prepare()
    print("Train: %d imagenes | Val: %d imagenes | Clases: %s"
          % (len(train_set.image_ids), len(val_set.image_ids), train_set.class_names))

    os.makedirs(LOGS_DIR, exist_ok=True)
    model = modellib.MaskRCNN(mode="training", config=cfg, model_dir=LOGS_DIR)

    # Carga de pesos iniciales
    if args.weights == "last":
        model.load_weights(model.find_last(), by_name=True)
    elif args.weights == "coco":
        coco = os.path.join(AIAPI_DIR, "weights", "mask_rcnn_coco.h5")
        # COCO tiene otras clases: se excluyen las cabezas.
        model.load_weights(coco, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    else:
        # Continuar desde los pesos ya entrenados (misma arquitectura/clases).
        model.load_weights(args.weights, by_name=True)

    print("Entrenando capas '%s' por %d epocas (lr=%.4g, steps=%d)..."
          % (args.layers, args.epochs, cfg.LEARNING_RATE, cfg.STEPS_PER_EPOCH))
    model.train(train_set, val_set,
                learning_rate=cfg.LEARNING_RATE,
                epochs=args.epochs,
                layers=args.layers)
    print("\nListo. Pesos guardados en: %s" % LOGS_DIR)
    print("Copia el .h5 elegido a AIAPI/weights/ y actualiza application.py para usarlo.")


if __name__ == "__main__":
    main()
