"""
Utilidades para CubiCasa5K: leer los splits, parsear las anotaciones `model.svg`
a cajas de verdad-terreno (ground truth) de 3 clases -> wall / door / window, y
dibujar comparaciones GT vs predicciones.

No depende de TensorFlow: solo stdlib + Pillow. Lo usan tanto el script de
preparacion como el de evaluacion.

Detalle importante de CubiCasa5K: el `model.svg` esta en su propio sistema de
coordenadas (su width/height), que NO coincide con el tamano de `F1_scaled.png`
(y ni siquiera mantiene el mismo aspecto). Por eso, para alinear el GT con la
imagen real que se le pasa al modelo, las coordenadas del SVG se escalan POR EJE:
    px = sx * (png_w / svg_w)
    py = sy * (png_h / svg_h)
"""

import os
import re
import xml.etree.ElementTree as ET

# Clases del modelo (mismo orden que usa AIAPI). Ver application.getClassNames:
#   1 = wall, 2 = window, 3 = door
CLASSES = ["wall", "door", "window"]

# Primer token de la clase del SVG -> nuestra clase.
_CLASS_MAP = {"Wall": "wall", "Door": "door", "Window": "window"}

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


# --------------------------------------------------------------------------- #
# Transforms SVG (se acumulan al bajar por el arbol)
# --------------------------------------------------------------------------- #
_IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)  # (a, b, c, d, e, f)


def _mat_mul(m1, m2):
    """Compone dos matrices SVG: aplica m2 y luego m1 (m1 . m2)."""
    a1, b1, c1, d1, e1, f1 = m1
    a2, b2, c2, d2, e2, f2 = m2
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _apply(m, x, y):
    a, b, c, d, e, f = m
    return a * x + c * y + e, b * x + d * y + f


def _parse_transform(value):
    """Parsea un atributo transform (matrix/translate/scale) a una matriz."""
    if not value:
        return _IDENTITY
    mat = _IDENTITY
    for name, args in re.findall(r"(matrix|translate|scale)\s*\(([^)]*)\)", value):
        nums = [float(n) for n in _NUM_RE.findall(args)]
        if name == "matrix" and len(nums) == 6:
            t = tuple(nums)
        elif name == "translate":
            tx = nums[0] if nums else 0.0
            ty = nums[1] if len(nums) > 1 else 0.0
            t = (1.0, 0.0, 0.0, 1.0, tx, ty)
        elif name == "scale":
            sx = nums[0] if nums else 1.0
            sy = nums[1] if len(nums) > 1 else sx
            t = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        else:
            continue
        mat = _mat_mul(mat, t)
    return mat


def _localname(elem):
    tag = elem.tag
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _category(elem):
    """Devuelve wall/door/window si la clase del elemento lo marca, si no None."""
    cls = elem.get("class")
    if not cls:
        return None
    return _CLASS_MAP.get(cls.split()[0])


# --------------------------------------------------------------------------- #
# Extraccion de puntos de un elemento (y sus hijos), ya transformados
# --------------------------------------------------------------------------- #
def _shape_polys(elem, mat):
    """Poligonos propios de este elemento (polygon/polyline/rect), transformados.

    Devuelve una lista de poligonos; cada poligono es una lista de (x, y).
    """
    tag = _localname(elem)
    if tag in ("polygon", "polyline"):
        nums = [float(n) for n in _NUM_RE.findall(elem.get("points", ""))]
        poly = [_apply(mat, nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]
        return [poly] if poly else []
    if tag == "rect":
        x = float(elem.get("x", 0)); y = float(elem.get("y", 0))
        w = float(elem.get("width", 0)); h = float(elem.get("height", 0))
        return [[_apply(mat, x, y), _apply(mat, x + w, y),
                 _apply(mat, x + w, y + h), _apply(mat, x, y + h)]]
    return []


def _collect_polys(elem, mat):
    """Todos los poligonos del subarbol de `elem` (mat ya incluye su transform)."""
    polys = list(_shape_polys(elem, mat))
    for child in list(elem):
        cmat = _mat_mul(mat, _parse_transform(child.get("transform")))
        polys.extend(_collect_polys(child, cmat))
    return polys


def _walk(elem, mat, out):
    mat = _mat_mul(mat, _parse_transform(elem.get("transform")))
    cat = _category(elem)
    if cat is not None:
        polys = _shape_polys(elem, mat)
        # Solo poligonos DIRECTOS de esta instancia (las puertas/ventanas
        # anidadas se cuentan aparte al recursar).
        for child in list(elem):
            cmat = _mat_mul(mat, _parse_transform(child.get("transform")))
            if _category(child) is None:
                polys.extend(_collect_polys(child, cmat))
        pts = [p for poly in polys for p in poly]
        if pts:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            out.append({"cls": cat, "bbox": [min(xs), min(ys), max(xs), max(ys)],
                        "polys": polys})
    # Siempre se recursa: puertas/ventanas suelen estar DENTRO de los muros.
    for child in list(elem):
        _walk(child, mat, out)


# --------------------------------------------------------------------------- #
# API publica
# --------------------------------------------------------------------------- #
def parse_svg(svg_path):
    """Parsea un model.svg -> dict {svg_w, svg_h, objects:[{cls, bbox}]} en
    coordenadas del propio SVG (sin escalar a la imagen)."""
    root = ET.parse(svg_path).getroot()
    svg_w = float(root.get("width") or 0) or None
    svg_h = float(root.get("height") or 0) or None
    if not svg_w or not svg_h:
        vb = (root.get("viewBox") or "").split()
        if len(vb) == 4:
            svg_w, svg_h = float(vb[2]), float(vb[3])
    objects = []
    _walk(root, _IDENTITY, objects)
    return {"svg_w": svg_w, "svg_h": svg_h, "objects": objects}


def gt_for_image(svg_path, png_w, png_h):
    """GT escalado al espacio de pixeles de la imagen (F1_scaled.png).

    Cada objeto: {cls, bbox[x1,y1,x2,y2], polys[[[x,y],...], ...]} (ints).
    """
    parsed = parse_svg(svg_path)
    sw, sh = parsed["svg_w"], parsed["svg_h"]
    if not sw or not sh:
        return []
    kx, ky = png_w / sw, png_h / sh
    out = []
    for o in parsed["objects"]:
        x1, y1, x2, y2 = o["bbox"]
        polys = [[[int(round(px * kx)), int(round(py * ky))] for px, py in poly]
                 for poly in o["polys"]]
        out.append({"cls": o["cls"],
                    "bbox": [x1 * kx, y1 * ky, x2 * kx, y2 * ky],
                    "polys": polys})
    return out


# --------------------------------------------------------------------------- #
# Splits y rutas
# --------------------------------------------------------------------------- #
def read_split(dataset_root, split_name):
    """Lee train.txt/val.txt/test.txt -> lista de rutas relativas de plano
    (p. ej. 'high_quality_architectural/6044')."""
    path = os.path.join(dataset_root, split_name)
    plans = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            rel = line.strip().strip("/")
            if rel:
                plans.append(rel)
    return plans


def plan_files(dataset_root, plan_rel):
    """(image_path, svg_path) de un plano. Usa F1_scaled.png (alineado al SVG)."""
    folder = os.path.join(dataset_root, plan_rel)
    return os.path.join(folder, "F1_scaled.png"), os.path.join(folder, "model.svg")


# --------------------------------------------------------------------------- #
# Dibujo de comparaciones (GT verde, predicciones rojas)
# --------------------------------------------------------------------------- #
_COLORS = {"wall": (52, 152, 219), "door": (231, 76, 60), "window": (46, 204, 113)}


def draw_comparison(image_path, gt, preds, out_path):
    """Dibuja GT (lineas verdes) y predicciones (cajas por color de clase + score)."""
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for g in gt:
        x1, y1, x2, y2 = g["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(39, 174, 96), width=3)
    for p in preds:
        x1, y1, x2, y2 = p["bbox"]
        color = _COLORS.get(p["cls"], (241, 196, 15))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = "%s %.2f" % (p["cls"], p.get("score", 0))
        draw.text((x1 + 2, max(0, y1 - 12)), label, fill=color)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    return out_path
