"""Telefon kamerasıyla gerçek dünya testi (README, "Telefon kamerasıyla gerçek dünya testi").

Kullanım (Colab'da defter çalıştırır):
    Deneme seti (serbest; 4 yöntem karşılaştırılır, biri seçilir):
        python scripts/phone_eval.py --set pilot --photo-dir /content/drive/MyDrive/fod_proje/telefon/pilot/foto \
            --runs-dir /content/drive/MyDrive/fod_proje/runs --results-dir /content/drive/MyDrive/fod_proje/results
    Test seti (YALNIZCA BİR KEZ; denemede seçilen yöntemle):
        python scripts/phone_eval.py --set test --final-test --photo-dir .../telefon/test/foto ...
    Test'in hata görüntüleri (test sayıları repoya commit edildikten sonra):
        python scripts/phone_eval.py --set test --hata-goruntusu --photo-dir .../telefon/test/foto ...

Girdiler:
    Fotoğraflar        : Drive'da (--photo-dir). Repoya girmez.
    Etiketler          : repoda telefon/<set>/etiket/*.txt (Make Sense, YOLO formatı; sınıflar telefon/siniflar.txt)
    Çekim defteri      : repoda telefon/<set>/cekim_defteri.csv
    Etiketler ve defter commit edilmemiş ya da commit'ten sonra değiştirilmişse script çalışmaz.

Model ve eşik sabit: exp002_nano, val F2 eşiği (0,25). Bütün tahminler işlemciyle yapılır.
"""

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO

from evaluate import (COUNTS, LOW_CONF, MATCH_IOU, NEAR_PX, SIZE_BINS, SIZE_LABELS, box_iou, count, f_scores,
                      git_commit, match, update_summary)

METHODS = ["320", "640", "1280", "parca"]
# "parca" (tiling) ayarları README'de çekimden önce yazıldı; denemede değiştirilmez
TILE_LONG = 1920     # fotoğrafın uzun kenarı bundan büyükse buna küçültülür
TILE = 320           # parça boyutu
TILE_OVERLAP = 80    # komşu parçaların örtüşmesi (piksel)
MERGE_IOU = 0.5      # bu kadar örtüşen tekrar kutular ayıklanır
CONTAIN = 0.8        # alanının bu kadarı daha emin bir kutunun içinde kalan parça kutu ayıklanır
TILE_BATCH = 32

TYPE_COLUMNS = ["vida", "somun", "pul", "civata", "tas", "diger"]
LIGHTS = ["aydinlik", "los", "aksam", "islak"]
DISTANCES = ["yakin", "orta", "uzak"]
GROUNDS = ["temiz", "karistirici"]
NOTEBOOK_COLUMNS = ["dosya_adi", "tarih", "isik", "mesafe", "zemin", "karistirici_tur"] + TYPE_COLUMNS + ["not"]
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
MIN_N = 20           # bundan az kutuya (boşlarda fotoğrafa) dayanan satır "az veri"
WARMUP = 3           # hız ölçümünde ilk bu kadar fotoğraf ısınma sayılır
EXAMPLES_PER_KIND = 8
DISPLAY_LONG = 1280  # hata görüntülerinin uzun kenarı
TR = str.maketrans({"ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g",
                    "ü": "u", "Ü": "u", "ö": "o", "Ö": "o", "ç": "c", "Ç": "c"})


def normalize(text):
    return str(text).translate(TR).strip().lower()


# ---------------------------------------------------------------- girdiler ve kontroller

def check_committed(repo_dir, set_dir, files):
    """Etiketler ve çekim defteri repoya commit edilmiş ve sonradan değiştirilmemiş olmalı."""
    rel = [str(f.relative_to(repo_dir)) for f in files]
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", *rel], cwd=repo_dir, capture_output=True, text=True)
    if tracked.returncode != 0:
        raise SystemExit(f"DUR: {set_dir} içindeki bazı dosyalar repoya commit edilmemiş.\n{tracked.stderr.strip()[:500]}")
    status = subprocess.run(["git", "status", "--porcelain", "--", str(set_dir.relative_to(repo_dir))],
                            cwd=repo_dir, capture_output=True, text=True).stdout.strip()
    if status:
        raise SystemExit(f"DUR: {set_dir} commit'ten sonra değiştirilmiş ya da commit edilmemiş dosya var:\n{status[:500]}")


def read_notebook(path):
    """Çekim defterini okur ve değerleri kontrol eder."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    table.columns = [normalize(c) for c in table.columns]
    missing = [c for c in NOTEBOOK_COLUMNS if c not in table.columns and c != "not"]
    if missing:
        raise SystemExit(f"DUR: çekim defterinde sütun eksik: {missing}")
    table = table[table["dosya_adi"].str.strip() != ""].copy()
    table["dosya_adi"] = table["dosya_adi"].str.strip()
    table["stem"] = [Path(name).stem for name in table["dosya_adi"]]
    for column, allowed in (("isik", LIGHTS), ("mesafe", DISTANCES), ("zemin", GROUNDS)):
        table[column] = table[column].map(normalize)
        bad = table[~table[column].isin(allowed)]
        if len(bad):
            raise SystemExit(f"DUR: çekim defterinde '{column}' sütununda geçersiz değer (izin verilen: {allowed}):\n"
                             f"{bad[['dosya_adi', column]].to_string(index=False)}")
    for column in TYPE_COLUMNS:
        values = table[column].str.strip().replace("", "0")
        if not values.str.fullmatch(r"\d+").all():
            raise SystemExit(f"DUR: çekim defterinde '{column}' sütununda sayı olmayan değer var.")
        table[column] = values.astype(int)
    table["cisim"] = table[TYPE_COLUMNS].sum(axis=1)
    dup = table[table["stem"].duplicated(keep=False)]
    if len(dup):
        raise SystemExit(f"DUR: çekim defterinde aynı fotoğraf iki kez var:\n{dup['dosya_adi'].to_string(index=False)}")
    return table.sort_values("stem").reset_index(drop=True)


def read_label_file(path, n_classes):
    """YOLO etiketi: her satır 'sınıf x_merkez y_merkez genişlik yükseklik' (0-1 arası). Dosya yoksa boş."""
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                raise SystemExit(f"DUR: {path.name} içinde bozuk satır: {line!r}")
            c, values = int(float(parts[0])), [float(v) for v in parts[1:]]
            if not 0 <= c < n_classes:
                raise SystemExit(f"DUR: {path.name} içinde geçersiz sınıf numarası: {c}")
            if not all(-0.01 <= v <= 1.01 for v in values):
                raise SystemExit(f"DUR: {path.name} içinde 0-1 aralığı dışında değer: {line!r}")
            rows.append([c] + values)
    return np.array(rows, dtype=float).reshape(-1, 5)


def check_inputs(table, photo_dir, label_dir, classes):
    """Fotoğraflar, etiketler ve çekim defteri birbiriyle tutarlı mı? Fotoğraf yolu sözlüğü döner."""
    photos = {}
    for f in sorted(photo_dir.iterdir()):
        if f.suffix.lower() in IMAGE_EXT:
            photos[f.stem] = f
        elif f.suffix.lower() == ".heic":
            raise SystemExit(f"DUR: {f.name} HEIC biçiminde. Telefonu JPEG'e ayarla ya da JPEG'e çevir.")
    stems = set(table["stem"])
    if set(photos) != stems:
        raise SystemExit("DUR: fotoğraflar ile çekim defteri uyuşmuyor.\n"
                         f"  defterde olup klasörde olmayan : {sorted(stems - set(photos))[:20]}\n"
                         f"  klasörde olup defterde olmayan : {sorted(set(photos) - stems)[:20]}")
    extra_labels = sorted({f.stem for f in label_dir.glob("*.txt")} - stems)
    if extra_labels:
        raise SystemExit(f"DUR: fotoğrafı olmayan etiket dosyası var: {extra_labels[:20]}")

    problems = []
    for row in table.itertuples():
        labels = read_label_file(label_dir / f"{row.stem}.txt", len(classes))
        got = np.bincount(labels[:, 0].astype(int), minlength=len(classes)) if len(labels) else np.zeros(len(classes), int)
        expected = [getattr(row, name) for name in classes]
        if list(got) != expected:
            problems.append(f"  {row.dosya_adi}: defter {dict(zip(classes, expected))}  etiket {dict(zip(classes, got.tolist()))}")
    if problems:
        raise SystemExit("DUR: etiketlerdeki kutu sayısı çekim defteriyle uyuşmuyor (her tür için):\n" + "\n".join(problems))
    return photos


# ---------------------------------------------------------------- tahmin

def load_photo(path):
    """Fotoğrafı telefonun kaydettiği yöne çevirerek okur (RGB). Ek bilgi (EXIF) taşınmaz."""
    with Image.open(path) as img:
        return np.asarray(ImageOps.exif_transpose(img).convert("RGB"))


def resize_for(img, method):
    """Fotoğrafı yöntemin ölçeğine küçültür. Döner: görüntü, x ölçeği, y ölçeği."""
    h, w = img.shape[:2]
    target = TILE_LONG if method == "parca" else int(method)
    if max(h, w) <= target:
        return img, 1.0, 1.0
    scale = target / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    small = np.asarray(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
    return small, new_w / w, new_h / h


def intersection(box, others):
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    return np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)


def merge_boxes(boxes, scores):
    """Parçalardan gelen kutuları birleştirir: en emin kutudan başlar; tekrar kutuları ve parça kutuları ayıklar."""
    keep = []
    for i in np.argsort(-scores):
        if keep:
            kept = boxes[keep]
            if box_iou(boxes[i], kept).max() >= MERGE_IOU:
                continue
            area = max(float((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])), 1e-9)
            if (intersection(boxes[i], kept) / area).max() >= CONTAIN:
                continue
        keep.append(int(i))
    return boxes[keep], scores[keep]


def tile_starts(length):
    if length <= TILE:
        return [0]
    starts = list(range(0, length - TILE, TILE - TILE_OVERLAP))
    starts.append(length - TILE)  # son parça kenara hizalanır
    return sorted(set(starts))


def predict_photo(model, img, method, conf):
    """Bir fotoğrafta modelin kutuları (yöntemin ölçeğinde). Döner: kutular, puanlar, x ölçeği, y ölçeği."""
    small, sx, sy = resize_for(img, method)
    bgr = np.ascontiguousarray(small[:, :, ::-1])  # Ultralytics numpy görüntüyü BGR bekler
    if method != "parca":
        r = model.predict(bgr, imgsz=int(method), conf=conf, device="cpu", verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), sx, sy

    h, w = bgr.shape[:2]
    tiles, offsets = [], []
    for y in tile_starts(h):
        for x in tile_starts(w):
            tiles.append(np.ascontiguousarray(bgr[y:y + TILE, x:x + TILE]))
            offsets.append((x, y))
    all_boxes, all_scores = [], []
    for start in range(0, len(tiles), TILE_BATCH):
        results = model.predict(tiles[start:start + TILE_BATCH], imgsz=TILE, conf=conf, device="cpu", verbose=False)
        for r, (x, y) in zip(results, offsets[start:start + TILE_BATCH]):
            all_boxes.append(r.boxes.xyxy.cpu().numpy() + np.array([x, y, x, y], dtype=float))
            all_scores.append(r.boxes.conf.cpu().numpy())
    boxes = np.concatenate(all_boxes).reshape(-1, 4) if all_boxes else np.zeros((0, 4))
    scores = np.concatenate(all_scores) if all_scores else np.zeros(0)
    merged_boxes, merged_scores = merge_boxes(boxes, scores)
    return merged_boxes, merged_scores, sx, sy


def collect(model, table, photos, label_dir, n_classes, methods, timing_conf=None):
    """Her fotoğraf ve yöntem için modelin bütün kutuları (düşük eşikle) ve gerçek kutular (aynı ölçekte).

    timing_conf verilirse her yöntemin fotoğraf başına süresi de ölçülür (sabit eşikle, ayrı bir geçişte).
    """
    preds = {m: {} for m in methods}
    times = {m: [] for m in methods}
    for k, row in enumerate(table.itertuples()):
        img = load_photo(photos[row.stem])
        h, w = img.shape[:2]
        labels = read_label_file(label_dir / f"{row.stem}.txt", n_classes)
        gt = np.stack([(labels[:, 1] - labels[:, 3] / 2) * w, (labels[:, 2] - labels[:, 4] / 2) * h,
                       (labels[:, 1] + labels[:, 3] / 2) * w, (labels[:, 2] + labels[:, 4] / 2) * h], axis=1).reshape(-1, 4)
        for m in methods:
            boxes, scores, sx, sy = predict_photo(model, img, m, LOW_CONF)
            preds[m][row.stem] = {"pred": boxes, "scores": scores, "gt": gt * np.array([sx, sy, sx, sy]),
                                  "gt_class": labels[:, 0].astype(int)}
            if timing_conf is not None:
                t0 = time.perf_counter()
                predict_photo(model, img, m, timing_conf)
                if k >= WARMUP:
                    times[m].append(1000 * (time.perf_counter() - t0))
        if (k + 1) % 25 == 0 or k + 1 == len(table):
            print(f"  {k + 1}/{len(table)} fotoğraf")
    return preds, times


# ---------------------------------------------------------------- ölçümler

def average_precision(preds, iou_thr):
    """Tek sınıf için AP (101 noktalı, COCO tarzı). preds: fotoğraf -> {pred, scores, gt}."""
    n_gt = sum(len(p["gt"]) for p in preds.values())
    if n_gt == 0:
        return float("nan")
    items = sorted(((s, stem, k) for stem, p in preds.items() for k, s in enumerate(p["scores"])), key=lambda t: -t[0])
    used = {stem: np.zeros(len(p["gt"]), dtype=bool) for stem, p in preds.items()}
    tp = np.zeros(len(items))
    for n, (_, stem, k) in enumerate(items):
        gt = preds[stem]["gt"]
        if len(gt) == 0:
            continue
        ious = box_iou(preds[stem]["pred"][k], gt)
        ious[used[stem]] = -1
        j = int(ious.argmax())
        if ious[j] >= iou_thr:
            used[stem][j] = True
            tp[n] = 1
    if len(items) == 0:
        return 0.0
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    recall = ctp / n_gt
    precision = np.maximum.accumulate((ctp / (ctp + cfp))[::-1])[::-1]  # zarf
    points = []
    for r in np.linspace(0, 1, 101):
        idx = int(np.searchsorted(recall, r, side="left"))
        points.append(precision[idx] if idx < len(precision) else 0.0)
    return float(np.mean(points))


def map_scores(preds):
    return average_precision(preds, 0.5), float(np.mean([average_precision(preds, t) for t in np.arange(0.5, 0.96, 0.05)]))


def detection_row(bolum, kirilim, grup, stems, preds, matched):
    sub = {s: preds[s] for s in stems}
    c = count(matched, stems)
    _, f2 = f_scores(c["precision"], c["recall"])
    n_box = sum(len(preds[s]["gt"]) for s in stems)
    map50, map50_95 = map_scores(sub)
    with_objects = [s for s in stems if len(preds[s]["gt"])]
    row = {"bolum": bolum, "kirilim": kirilim, "grup": grup, "foto": len(stems), "kutu": n_box,
           "mAP50": map50, "mAP50_95": map50_95, "precision": c["precision"], "recall": c["recall"], "F2": f2,
           **{k: c[k] for k in COUNTS}}
    if bolum == "cisimli":
        row["foto_hepsi_bulundu"] = float(np.mean([matched[s]["gt_found"].all() for s in with_objects]))
        row["foto_en_az_biri_bulundu"] = float(np.mean([matched[s]["gt_found"].any() for s in with_objects]))
    row["not"] = "az veri" if n_box < MIN_N else ""
    return row


def recall_rows(kirilim, groups, matched):
    """Kutu bazında kırılım (tür, boyut): yalnızca recall ve kaçırılan cisim."""
    rows = []
    for grup, items in groups:
        if not items:
            continue
        found = sum(bool(matched[s]["gt_found"][k]) for s, k in items)
        rows.append({"bolum": "cisimli", "kirilim": kirilim, "grup": grup, "foto": len({s for s, _ in items}),
                     "kutu": len(items), "recall": found / len(items), "kacirilan_cisim": len(items) - found,
                     "not": "az veri" if len(items) < MIN_N else ""})
    return rows


def empty_row(kirilim, grup, stems, matched):
    fa = [int((~matched[s]["pred_ok"]).sum()) for s in stems]
    with_fa = sum(v > 0 for v in fa)
    return {"bolum": "bos", "kirilim": kirilim, "grup": grup, "foto": len(stems), "kutu": 0,
            "fazladan_kutu": sum(fa), "fa_olan_foto": with_fa, "fa_olan_foto_orani": with_fa / len(stems),
            "foto_basina_fa": sum(fa) / len(stems), "not": "az veri" if len(stems) < MIN_N else ""}


def breakdown(table, preds, matched, classes):
    rows = []
    objects = table[table["cisim"] > 0]
    empties = table[table["cisim"] == 0]

    rows.append(detection_row("tumu", "tümü (cisimli + boş)", "tümü", list(table["stem"]), preds, matched))
    if len(objects):
        rows.append(detection_row("cisimli", "tümü", "tümü", list(objects["stem"]), preds, matched))
        for column, order in (("mesafe", DISTANCES), ("isik", LIGHTS)):
            for value in order:
                sub = objects[objects[column] == value]
                if len(sub):
                    rows.append(detection_row("cisimli", column, value, list(sub["stem"]), preds, matched))
        boxes = [(s, k, int(c), max(b[2] - b[0], b[3] - b[1]))
                 for s in objects["stem"] for k, (b, c) in enumerate(zip(preds[s]["gt"], preds[s]["gt_class"]))]
        type_groups = [(name + (" (eğitimde yok)" if name == "vida" else ""), [(s, k) for s, k, c, _ in boxes if c == i])
                       for i, name in enumerate(classes)]
        rows += recall_rows("cisim türü", type_groups, matched)
        sizes = pd.cut([b[3] for b in boxes], bins=SIZE_BINS, labels=SIZE_LABELS, right=False).astype(str)
        size_groups = [(label, [(s, k) for (s, k, _, _), z in zip(boxes, sizes) if z == label]) for label in SIZE_LABELS]
        rows += recall_rows("kutu boyutu (modele giren ölçekte)", size_groups, matched)
    if len(empties):
        rows.append(empty_row("tümü", "tümü", list(empties["stem"]), matched))
        for column, order in (("zemin", GROUNDS), ("isik", LIGHTS), ("mesafe", DISTANCES)):
            for value in order:
                sub = empties[empties[column] == value]
                if len(sub):
                    rows.append(empty_row(column, value, list(sub["stem"]), matched))

    columns = ["bolum", "kirilim", "grup", "foto", "kutu", "mAP50", "mAP50_95", "precision", "recall", "F2"] + COUNTS + [
        "foto_hepsi_bulundu", "foto_en_az_biri_bulundu", "fa_olan_foto", "fa_olan_foto_orani", "foto_basina_fa", "not"]
    out = pd.DataFrame(rows).reindex(columns=columns)
    numeric = ["mAP50", "mAP50_95", "precision", "recall", "F2", "foto_hepsi_bulundu", "foto_en_az_biri_bulundu",
               "fa_olan_foto_orani", "foto_basina_fa"]
    out[numeric] = out[numeric].astype(float).round(4)
    return out


# ---------------------------------------------------------------- hata görüntüleri

def draw_errors(table, photos, matched, method, out_dir, conf):
    """Kutuları çizilmiş örnek hata görüntüleri. Görüntü yeniden çizildiği için EXIF (konum vb.) taşınmaz."""
    if out_dir.exists():
        for f in out_dir.iterdir():
            f.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    kinds = [
        ("kacirilan_cisim", lambda s: (~matched[s]["gt_found"]).any(), table["cisim"] > 0),
        ("cisimli_fotoda_zeminde_alarm", lambda s: (~matched[s]["pred_ok"] & ~matched[s]["fp_near"]).any(), table["cisim"] > 0),
        ("cisimli_fotoda_cisimde_alarm", lambda s: (~matched[s]["pred_ok"] & matched[s]["fp_near"]).any(), table["cisim"] > 0),
        ("bos_zeminde_alarm", lambda s: (~matched[s]["pred_ok"]).any(), table["cisim"] == 0),
    ]
    used, n = set(), 0
    for kind, has_error, mask in kinds:
        candidates = [s for s in table.loc[mask, "stem"] if has_error(s) and s not in used]
        if len(candidates) > EXAMPLES_PER_KIND:
            candidates = [candidates[i] for i in np.linspace(0, len(candidates) - 1, EXAMPLES_PER_KIND).round().astype(int)]
        for s in candidates:
            small, _, _ = resize_for(load_photo(photos[s]), method)
            f = DISPLAY_LONG / max(small.shape[:2])
            img = Image.fromarray(small).resize((round(small.shape[1] * f), round(small.shape[0] * f)))
            draw = ImageDraw.Draw(img)
            e = matched[s]
            for box, found in zip(e["gt"], e["gt_found"]):
                draw.rectangle([float(v) * f for v in box], outline=(0, 200, 0) if found else (255, 210, 0), width=3)
            for box, score, ok, near in zip(e["pred"], e["scores"], e["pred_ok"], e["fp_near"]):
                color = (40, 120, 255) if ok else ((255, 0, 255) if near else (255, 0, 0))
                scaled = [float(v) * f for v in box]
                draw.rectangle(scaled, outline=color, width=2)
                draw.text((scaled[0] + 3, scaled[1] + 3), f"{score:.2f}", fill=color)
            img.save(out_dir / f"{kind}_{s}.jpg", quality=90)  # exif verilmediği için kaydedilmez
            used.add(s)
            n += 1
    (out_dir / "ACIKLAMA.txt").write_text(
        f"Hata örnekleri, yöntem '{method}', güven eşiği {conf}. Görüntüler modele giren ölçekten {DISPLAY_LONG} piksele büyütüldü/küçültüldü.\n"
        "Yeşil kutu  : gerçek cisim, model buldu\n"
        "Sarı kutu   : gerçek cisim, model KAÇIRDI\n"
        "Mavi kutu   : modelin doğru tespiti (yanındaki sayı güven puanı)\n"
        "Mor kutu    : CİSİMDE fazladan kutu (cismin parçası ya da ikinci kutu)\n"
        "Kırmızı kutu: ZEMİNDE fazladan kutu (yanlış alarm)\n"
        f"Doğru sayılma: gerçek kutuyla en az %{100 * MATCH_IOU:.0f} örtüşme. 'Cisimde': gerçek kutuya {NEAR_PX} pikselden yakın.\n"
        "Dosya adı: <hata türü>_<fotoğraf adı>.jpg\n", encoding="utf-8")
    return n


# ---------------------------------------------------------------- yardımcılar

def cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "bilinmiyor"


def summary_row(results_root, model_name, split, method, conf, overall, note):
    info = json.loads((results_root / model_name / "train_info.json").read_text(encoding="utf-8"))
    test = json.loads((results_root / model_name / "test_metrics.json").read_text(encoding="utf-8"))
    return {"exp_id": model_name, "date": datetime.now().strftime("%Y-%m-%d"), "model": info.get("model", ""),
            "imgsz": method, "epochs_max": info.get("epochs", ""), "epochs_run": test.get("epochs_run", ""),
            "best_epoch": test.get("best_epoch", ""), "split": split, "conf": conf, "esik_kurali": "F2",
            "mAP50": round(overall["mAP50"], 4), "mAP50_95": round(overall["mAP50_95"], 4),
            "precision": round(overall["precision"], 4), "recall": round(overall["recall"], 4), "notes": note}


def log_has(log_file, name):
    return log_file.exists() and name in set(pd.read_csv(log_file)["exp_id"])


def main():
    parser = argparse.ArgumentParser(description="FOD: telefon fotoğraflarıyla gerçek dünya testi")
    parser.add_argument("--set", required=True, choices=["pilot", "test"])
    parser.add_argument("--photo-dir", required=True, help="Drive'daki fotoğraf klasörü")
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--model-name", default="exp002_nano")
    parser.add_argument("--final-test", action="store_true", help="Test setini değerlendirmek için zorunlu onay (bir kez)")
    parser.add_argument("--hata-goruntusu", action="store_true", help="Test'in hata görüntüleri (test sayıları commit edildikten sonra)")
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    results_root = Path(args.results_dir)
    log_file = results_root / "telefon_test_log.csv"
    set_dir = repo_dir / "telefon" / args.set
    label_dir = set_dir / "etiket"
    notebook_path = set_dir / "cekim_defteri.csv"
    classes = [normalize(c) for c in (repo_dir / "telefon" / "siniflar.txt").read_text(encoding="utf-8").split()]
    if classes != TYPE_COLUMNS:
        raise SystemExit(f"DUR: telefon/siniflar.txt beklenen sırada değil: {classes} (beklenen {TYPE_COLUMNS})")

    # Hangi iş? Test korumaları
    if args.set == "pilot":
        if args.final_test or args.hata_goruntusu:
            raise SystemExit("DUR: --final-test ve --hata-goruntusu yalnızca test seti içindir.")
        if log_has(log_file, args.model_name):
            raise SystemExit("DUR: telefon testi yapılmış; deneme seçimi artık değiştirilemez.")
    elif args.hata_goruntusu:
        if not log_has(log_file, args.model_name):
            raise SystemExit("DUR: telefon testi henüz yapılmadı; hata görüntüleri testten sonra çıkarılır.")
        committed = subprocess.run(["git", "ls-files", "--error-unmatch", "results/telefon_test/test_metrics.json"],
                                   cwd=repo_dir, capture_output=True)
        if committed.returncode != 0:
            raise SystemExit("DUR: test sayıları henüz repoya commit edilmedi. Önce sonuçlar commit edilmeli (README kuralı).")
    else:
        if not args.final_test:
            raise SystemExit("DUR: Test seti yalnızca bir kez değerlendirilir. Emin isen --final-test ekle.")
        if log_has(log_file, args.model_name):
            raise SystemExit(f"DUR: telefon testi zaten yapılmış (bkz. {log_file}). İkinci kez yapılmaz.")

    if not notebook_path.exists():
        raise SystemExit(f"DUR: {notebook_path} yok. Çekim defteri ve etiketler önce repoya commit edilmeli.")
    label_files = sorted(label_dir.glob("*.txt"))
    check_committed(repo_dir, set_dir, [notebook_path] + label_files)
    table = read_notebook(notebook_path)
    photos = check_inputs(table, Path(args.photo_dir), label_dir, classes)
    print(f"Set: {args.set}   fotoğraf: {len(table)} (cisimli {int((table['cisim'] > 0).sum())}, boş {int((table['cisim'] == 0).sum())})"
          f"   kutu: {int(table['cisim'].sum())}")
    print("OK: etiketler çekim defteriyle uyuşuyor ve repoya commit edilmiş.")

    conf = float(json.loads((results_root / args.model_name / "conf_threshold.json").read_text(encoding="utf-8"))["conf_f2"])
    model = YOLO(str(Path(args.runs_dir) / args.model_name / "weights" / "best.pt"))
    date = datetime.now().strftime("%Y-%m-%d %H:%M")
    commit = git_commit()

    if args.set == "pilot":
        out_dir = results_root / "telefon_pilot"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nDört yöntem ölçülüyor (işlemci: {cpu_name()}, iş parçacığı: {torch.get_num_threads()}), eşik {conf}...")
        preds, times = collect(model, table, photos, label_dir, len(classes), METHODS, timing_conf=conf)
        rows, results = [], {}
        for m in METHODS:
            matched = {s: match(p, conf) for s, p in preds[m].items()}
            result = breakdown(table, preds[m], matched, classes)
            results[m] = (result, matched)
            overall = result.iloc[0]
            empty = result[(result["bolum"] == "bos") & (result["kirilim"] == "tümü")]
            rows.append({"yontem": m, "F2_tumu": round(float(overall["F2"]), 4), "F2_yuvarlanmis": round(float(overall["F2"]), 3),
                         "precision": overall["precision"], "recall": overall["recall"],
                         "fazladan_kutu": int(overall["fazladan_kutu"]), "kacirilan_cisim": int(overall["kacirilan_cisim"]),
                         "bos_fotoda_fa_olan": int(empty["fa_olan_foto"].iloc[0]) if len(empty) else "",
                         "ortanca_ms_foto": round(float(np.median(times[m])), 1) if times[m] else float("nan"),
                         "olculen_foto": len(times[m]), "islemci": cpu_name(), "is_parcacigi": torch.get_num_threads()})
        options = pd.DataFrame(rows)
        best = options["F2_yuvarlanmis"].max()
        chosen = options[options["F2_yuvarlanmis"] == best].sort_values("ortanca_ms_foto").iloc[0]
        method = str(chosen["yontem"])
        options["secildi"] = options["yontem"] == method
        options.to_csv(out_dir / "secenekler.csv", index=False)
        fastest_single = options[options["yontem"] != "parca"]["ortanca_ms_foto"].min()
        selection = {
            "yontem": method, "tarih": date, "git_commit": commit, "esik": conf,
            "kural": "deneme setinin tamamında F2 en yüksek; 3 ondalıkta eşitlikte daha hızlı olan (README)",
            "F2": float(chosen["F2_tumu"]), "ortanca_ms_foto": float(chosen["ortanca_ms_foto"]),
            "parca_ayarlari": {"uzun_kenar": TILE_LONG, "parca": TILE, "ortusme": TILE_OVERLAP,
                               "birlestirme_iou": MERGE_IOU, "icinde_kalma": CONTAIN},
        }
        if method == "parca":
            selection["parca_hiz_notu"] = (f"parca fotoğraf başına {chosen['ortanca_ms_foto']:.0f} ms; tek seferlik yöntemlerin "
                                           f"en hızlısı {fastest_single:.0f} ms ({chosen['ortanca_ms_foto'] / fastest_single:.1f} kat yavaş)")
        (out_dir / "secim.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
        result, matched = results[method]
        result.to_csv(out_dir / "pilot_breakdown.csv", index=False)
        n_err = draw_errors(table, photos, matched, method, out_dir / "hata_ornekleri", conf)
        cisimli = result[(result["bolum"] == "cisimli") & (result["kirilim"] == "tümü")]
        if len(cisimli):
            update_summary(results_root, summary_row(results_root, args.model_name, "telefon_pilot", method, conf,
                                                     cisimli.iloc[0], "telefon deneme seti, cisimli fotograflar"))
        print("\nDört yöntem (deneme setinin tamamı):")
        print(options.drop(columns=["islemci"]).to_string(index=False))
        print(f"\nSEÇİLEN YÖNTEM: {method}")
        if "parca_hiz_notu" in selection:
            print("Hız notu:", selection["parca_hiz_notu"])
        print(f"\nSeçilen yöntemle kırılım:\n{result.to_string(index=False)}")
        print(f"\nHata örneği: {n_err}   Sonuçlar: {out_dir}")
        return

    # Test seti
    selection_path = results_root / "telefon_pilot" / "secim.json"
    if not selection_path.exists():
        raise SystemExit("DUR: deneme seçimi yok (results/telefon_pilot/secim.json). Önce deneme seti değerlendirilmeli.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    method = selection["yontem"]
    out_dir = results_root / "telefon_test"
    print(f"\nYöntem (denemede seçildi): {method}   eşik: {conf}")
    preds, _ = collect(model, table, photos, label_dir, len(classes), [method])
    matched = {s: match(p, conf) for s, p in preds[method].items()}

    if args.hata_goruntusu:
        n_err = draw_errors(table, photos, matched, method, out_dir / "hata_ornekleri", conf)
        print(f"Hata örneği: {n_err}   Klasör: {out_dir / 'hata_ornekleri'}")
        return

    result = breakdown(table, preds[method], matched, classes)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "test_breakdown.csv", index=False)
    pilot_dates = set()
    pilot_notebook = repo_dir / "telefon" / "pilot" / "cekim_defteri.csv"
    if pilot_notebook.exists():
        pilot_dates = set(read_notebook(pilot_notebook)["tarih"].str.strip())
    same_day = sorted(set(table["tarih"].str.strip()) & pilot_dates - {""})
    fod_a = json.loads((results_root / args.model_name / "test_metrics.json").read_text(encoding="utf-8"))
    cisimli = result[(result["bolum"] == "cisimli") & (result["kirilim"] == "tümü")].iloc[0]
    empty = result[(result["bolum"] == "bos") & (result["kirilim"] == "tümü")]
    metrics = {
        "exp_id": args.model_name, "set": "telefon_test", "tarih": date, "git_commit": commit,
        "yontem": method, "esik": conf, "secim": selection,
        "foto": len(table), "cisimli_foto": int((table["cisim"] > 0).sum()), "bos_foto": int((table["cisim"] == 0).sum()),
        "kutu": int(table["cisim"].sum()),
        "cisimli": {k: (None if pd.isna(v) else v) for k, v in cisimli.drop(["bolum", "kirilim", "grup", "not"]).items()},
        "bos": ({k: (None if pd.isna(v) else v) for k, v in empty.iloc[0].drop(["bolum", "kirilim", "grup", "not"]).items()}
                if len(empty) else None),
        "fod_a_test": {k: fod_a[k] for k in ("mAP50", "mAP50_95", "precision", "recall")},
        "deneme_ile_ayni_gun": same_day,
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    pd.DataFrame([{"exp_id": args.model_name, "date": date, "yontem": method, "foto": len(table), "git_commit": commit}]).to_csv(
        log_file, mode="a", header=not log_file.exists(), index=False)
    update_summary(results_root, summary_row(results_root, args.model_name, "telefon_test", method, conf, cisimli,
                                             "telefon test seti, cisimli fotograflar"))
    if same_day:
        print(f"UYARI: deneme ile aynı günde çekilmiş test fotoğrafı var ({same_day}). Raporda belirtilecek.")
    print(f"\nTelefon testi ({method}, eşik {conf}):\n{result.to_string(index=False)}")
    print(f"\nKayıt: {log_file}   Sonuçlar: {out_dir}")


if __name__ == "__main__":
    main()
