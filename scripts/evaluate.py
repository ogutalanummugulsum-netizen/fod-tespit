"""Eğitilmiş bir modeli val ya da test üzerinde ölçer ve kırılımlarıyla raporlar.

Kullanım (val):
    python scripts/evaluate.py --name exp002_nano --split val \
        --runs-dir /content/drive/MyDrive/fod_proje/runs \
        --results-dir /content/drive/MyDrive/fod_proje/results

Kullanım (test, yalnızca nihai model için ve bir kez):
    python scripts/evaluate.py --name <deney> --split test --final-test --runs-dir ... --results-dir ...

Ölçümler: mAP50, mAP50-95, precision, recall ve hata sayıları.
Kırılımlar: tümü, sınıf grubu (eğitimde var / yok), ışık koşulu, kutu boyutu; val'de ayrıca video parçası.
Her satırda kaç kareye, kaç kutuya ve kaç video parçasına dayandığı yazar.

Güven eşiği (README, "Güven eşiği kuralı"):
    - Val değerlendirmesinde eşik tablosu çıkarılır ve val'in tamamında F2'yi en yüksek yapan
      eşik seçilir (adaylar: 0.05-0.60 arası 0.05 adımla + F1'i en yüksek yapan eşik; eşitlikte
      yüksek olan). Bu eşik conf_threshold.json dosyasına yazılır.
    - Test'te eşik yeniden hesaplanmaz; val'de yazılan F2 eşiği kullanılır.
    - mAP50 ve mAP50-95 eşikten bağımsızdır (Ultralytics ile ölçülür).
    - Precision, recall ve hata sayıları basit eşleştirmeyle hesaplanır: model kutusu, gerçek kutuyla
      en az %50 örtüşürse (IoU >= 0.5) doğru sayılır.

Yanlış alarmlar (fazladan kutular) yalnızca raporlama için ikiye ayrılır, eşik kuralını etkilemez:
    - cisimde: gerçek bir kutuyla kesişen ya da ona 16 pikselden yakın olan (cismin parçası, ikinci kutu)
    - zeminde: hiçbir gerçek kutuya yakın olmayan (boş zeminde alarm)

Test koruması: test için --final-test gerekir, her test değerlendirmesi
<results-dir>/test_log.csv dosyasına yazılır ve aynı deney için ikinci kez çalışmaz.

Sızıntılı ayrım deneyi (README, Deney sırası 6. adım) başka bir veri klasörüyle çalışır:
    --data-dir data/yolo_sizintili --split-label sizintili_test --test-log test_log_sizintili.csv
Bizim veri klasörü (data/yolo) dışındaki bir klasörde test için bu iki ayar zorunludur; böylece
sonuçlar ve log, bizim test sonuçlarıyla karışmaz.

Ürettiği dosyalar (<results-dir>/<name>/ altında):
    <etiket>_breakdown.csv  -> kırılım tablosu; etiket = --split-label, verilmezse bölüm adı (val / test)
    <etiket>_metrics.json   -> genel sonuç, eşik ve ağırlıklar
    conf_threshold.json     -> yalnızca val: F1 ve F2 eşikleri
    val_thresholds.csv      -> yalnızca val: farklı güven eşiklerinde precision, recall ve hata sayıları
    val_errors/             -> yalnızca val: kutuları çizilmiş örnek hata görüntüleri
ve <results-dir>/summary.csv dosyasına bir satır.
"""

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from ultralytics import YOLO

from fod_data import write_eval_yaml

SIZE_BINS = [0, 16, 32, 64, 96, 100000]
SIZE_LABELS = ["<16 px", "16-32 px", "32-64 px", "64-96 px", ">=96 px"]
MIN_SEGMENTS = 3  # bundan az video parçasına dayanan satır "az veri" diye işaretlenir
METRICS = ["mAP50", "mAP50_95", "precision", "recall"]
COUNTS = ["fazladan_kutu", "fazladan_kutu_cisimde", "fazladan_kutu_zeminde", "kacirilan_cisim"]
SUMMARY_COLUMNS = [
    "exp_id", "date", "model", "imgsz", "epochs_max", "epochs_run", "best_epoch", "split",
    "conf", "esik_kurali", "mAP50", "mAP50_95", "precision", "recall", "notes"]

MATCH_IOU = 0.5   # model kutusu, gerçek kutuyla en az bu kadar örtüşürse doğru sayılır
NEAR_PX = 16      # fazladan kutu, gerçek kutuya bu kadar pikselden yakınsa "cisimde" sayılır
LOW_CONF = 0.01   # modelin bütün kutuları bu eşikle toplanır, sonra istenen eşikle süzülür
THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]  # eşik tablosundaki değerler

# Hata örnekleri (yalnızca val): hangi gruplardan, her türden en fazla kaç görüntü
ERROR_GROUPS = [
    ("aydinlik", lambda t: t["isik"] == "Dry-Bright"),
    ("96px_ustu", lambda t: t["boyut"] == ">=96 px"),
    ("diger", None),
]
EXAMPLES_PER_KIND = 8


def load_image_table(data_dir):
    """Her görüntü için bir satır: bölüm, video parçası, ışık koşulu, sınıf grubu, kutu boyutu."""
    boxes = pd.read_csv(data_dir / "boxes.csv", dtype={"image_id": str})
    image_splits = pd.read_csv(data_dir / "image_splits.csv", dtype={"image_id": str})

    boxes["long_side"] = (boxes[["xmax", "ymax"]].values - boxes[["xmin", "ymin"]].values).max(axis=1)
    train_ids = set(image_splits.loc[image_splits["split"] == "train", "image_id"])
    train_classes = set(boxes.loc[boxes["image_id"].isin(train_ids), "class_name"])
    boxes["unseen"] = ~boxes["class_name"].isin(train_classes)

    per_image = boxes.groupby("image_id").agg(
        n_boxes=("class_name", "size"),
        min_side=("long_side", "min"),  # iki cisimli karede küçük olan kutu esas alınır
        unseen=("unseen", "any"),
        weather=("weather", "first"),
        light=("light", "first"),
        classes=("class_name", lambda names: "+".join(sorted(set(names)))),
    ).reset_index()

    table = image_splits.merge(per_image, on="image_id")
    table["isik"] = table["weather"] + "-" + table["light"]
    table["sinif_grubu"] = np.where(table["unseen"], "eğitimde yok", "eğitimde var")
    table["boyut"] = pd.cut(table["min_side"], bins=SIZE_BINS, labels=SIZE_LABELS, right=False).astype(str)
    return table, boxes


def make_groups(part, per_segment=False):
    """Ölçülecek grupları sıralar: [(kırılım, grup, o gruptaki görüntüler), ...]

    per_segment=True ise her video parçası da ayrı bir satır olur (yalnızca val'de kullanılır).
    """
    groups = [("tümü", "tümü", part)]
    for column, title, order in (
        ("sinif_grubu", "sınıf grubu", ["eğitimde var", "eğitimde yok"]),
        ("isik", "ışık koşulu", sorted(part["isik"].unique())),
        ("boyut", "kutu boyutu", SIZE_LABELS),
    ):
        for value in order:
            sub = part[part[column] == value]
            if len(sub):
                groups.append((title, value, sub))
    if per_segment:
        for segment_id, sub in part.groupby("segment_id", sort=True):
            classes = "+".join(sorted(set("+".join(sub["classes"]).split("+"))))
            groups.append(("video parçası", f"{segment_id} {classes} {sub['isik'].iloc[0]}", sub))
    return groups


def run_val(model, sub, data_dir, split, work_dir, tag, imgsz, batch):
    """Bir görüntü grubunu YOLO ile ölçer; mAP değerlerini ve F1 eğrisini döner."""
    image_list = work_dir / f"{tag}.txt"
    image_dir = (data_dir / "images" / split).resolve()
    image_list.write_text("\n".join(str(image_dir / f"{i}.jpg") for i in sub["image_id"]) + "\n", encoding="utf-8")
    data_yaml = write_eval_yaml(data_dir, image_list, work_dir / f"{tag}.yaml")

    # split="val": ayar dosyasındaki "val" satırı bizim görüntü listemizi gösterir
    metrics = model.val(
        data=str(data_yaml), split="val", imgsz=imgsz, batch=batch,
        plots=False, verbose=False, project=str(work_dir), name=f"val_{tag}", exist_ok=True,
    )
    box = metrics.box
    if len(np.asarray(box.f1_curve)) == 0:
        raise SystemExit(f"'{tag}' grubunda ölçüm yapılamadı: model hiç tahmin üretmedi ya da etiket bulunamadı.")
    return {
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "px": np.asarray(box.px),                  # güven eşikleri (0-1 arası 1000 nokta)
        "f1_curve": np.asarray(box.f1_curve)[0],   # her eşikte F1
    }


def box_iou(box, others):
    """Bir kutunun, bir dizi kutuyla örtüşme oranı (IoU). Kutular: xmin, ymin, xmax, ymax."""
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_others = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / (area_box + area_others - inter + 1e-9)


def near_any(box, others, margin):
    """Kutu, gerçek kutulardan biriyle kesişiyor ya da ona 'margin' pikselden yakın mı?"""
    if len(others) == 0:
        return False
    grown = others + np.array([-margin, -margin, margin, margin])
    overlap_x = (box[0] < grown[:, 2]) & (box[2] > grown[:, 0])
    overlap_y = (box[1] < grown[:, 3]) & (box[3] > grown[:, 1])
    return bool((overlap_x & overlap_y).any())


def predict_all(model, part, boxes, data_dir, split, work_dir, imgsz, batch):
    """Her görüntü için modelin bütün kutularını (çok düşük güven eşiğiyle) ve gerçek kutuları toplar."""
    image_dir = (data_dir / "images" / split).resolve()
    source = work_dir / "predict_list.txt"
    source.write_text("\n".join(str(image_dir / f"{i}.jpg") for i in part["image_id"]) + "\n", encoding="utf-8")

    in_part = boxes[boxes["image_id"].isin(set(part["image_id"]))]
    gt_boxes = {i: g[["xmin", "ymin", "xmax", "ymax"]].values.astype(float) for i, g in in_part.groupby("image_id")}

    predictions = {}
    for result in model.predict(source=str(source), conf=LOW_CONF, imgsz=imgsz, batch=batch, stream=True, verbose=False):
        image_id = Path(result.path).stem
        predictions[image_id] = {
            "pred": result.boxes.xyxy.cpu().numpy(),
            "scores": result.boxes.conf.cpu().numpy(),
            "gt": gt_boxes.get(image_id, np.zeros((0, 4))),
        }
    missing = set(part["image_id"]) - set(predictions)
    if missing:
        raise SystemExit(f"{len(missing)} görüntü için tahmin alınamadı (örnek: {sorted(missing)[:3]}).")
    return predictions


def match(prediction, conf):
    """Güven eşiğini geçen model kutularını gerçek kutularla eşleştirir.

    Fazladan kutu: hiçbir gerçek kutuyla eşleşmeyen model kutusu (yanlış alarm).
        cisimde: gerçek bir kutuyla kesişiyor ya da ona NEAR_PX pikselden yakın
        zeminde: hiçbir gerçek kutuya yakın değil
    Kaçırılan cisim: hiçbir model kutusuyla eşleşmeyen gerçek kutu.
    """
    keep = prediction["scores"] >= conf
    pred, scores, gt = prediction["pred"][keep], prediction["scores"][keep], prediction["gt"]
    gt_found = np.zeros(len(gt), dtype=bool)
    pred_ok = np.zeros(len(pred), dtype=bool)
    for k in np.argsort(-scores):  # en emin olduğu kutudan başla
        if len(gt) == 0:
            break
        ious = box_iou(pred[k], gt)
        ious[gt_found] = 0
        j = int(ious.argmax())
        if ious[j] >= MATCH_IOU:
            gt_found[j] = True
            pred_ok[k] = True
    fp_near = np.array([not ok and near_any(box, gt, NEAR_PX) for box, ok in zip(pred, pred_ok)], dtype=bool)
    return {"pred": pred, "scores": scores, "pred_ok": pred_ok, "fp_near": fp_near, "gt": gt, "gt_found": gt_found}


def count(matched, ids):
    """Bir görüntü grubu için doğru tespit, hata sayıları, precision ve recall."""
    tp = sum(int(matched[i]["pred_ok"].sum()) for i in ids)
    fp = sum(int((~matched[i]["pred_ok"]).sum()) for i in ids)
    fp_near = sum(int(matched[i]["fp_near"].sum()) for i in ids)
    fn = sum(int((~matched[i]["gt_found"]).sum()) for i in ids)
    return {
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
        "fazladan_kutu": fp,
        "fazladan_kutu_cisimde": fp_near,
        "fazladan_kutu_zeminde": fp - fp_near,
        "kacirilan_cisim": fn,
        "kacirilan_cisimli_kare": sum(int((~matched[i]["gt_found"]).any()) for i in ids),
    }


def f_scores(precision, recall):
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # F2: recall'a precision'dan iki kat ağırlık verir (cisim kaçırmak daha tehlikeli)
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    return f1, f2


def threshold_table(predictions, part, conf_f1):
    """Farklı güven eşiklerinde precision, recall ve hata sayıları (tümü ve ışık koşulu başına).

    Ayrıca kurala göre F2 eşiğini döner: "tümü"nde F2'yi en yüksek yapan eşik, eşitlikte yüksek olan.
    """
    thresholds = sorted(set(THRESHOLDS + [round(conf_f1, 3)]))
    groups = [("tümü", part)] + [(value, sub) for value, sub in part.groupby("isik", sort=True)]
    rows = []
    for t in thresholds:
        matched = {i: match(p, t) for i, p in predictions.items()}
        for name, sub in groups:
            c = count(matched, list(sub["image_id"]))
            f1, f2 = f_scores(c["precision"], c["recall"])
            rows.append({"grup": name, "esik": t, **c, "F1": f1, "F2": f2})
    table = pd.DataFrame(rows)

    overall = table[table["grup"] == "tümü"]
    best_f2 = overall["F2"].max()
    conf_f2 = float(overall.loc[overall["F2"] == best_f2, "esik"].max())  # eşitlikte yüksek eşik

    table["not"] = ""
    table.loc[table["esik"] == round(conf_f1, 3), "not"] = "F1 eşiği"
    table.loc[table["esik"] == conf_f2, "not"] = "F2 eşiği (seçilen)"
    # Sıra: önce "tümü", sonra ışık koşulları; her grubun içinde eşik küçükten büyüğe
    table["sira"] = table["grup"].map({name: k for k, (name, _) in enumerate(groups)})
    table = table.sort_values(["sira", "esik"])
    columns = ["grup", "esik", "precision", "recall", "F1", "F2"] + COUNTS + ["kacirilan_cisimli_kare", "not"]
    table = table[columns]
    table[["precision", "recall", "F1", "F2"]] = table[["precision", "recall", "F1", "F2"]].round(4)
    return table, conf_f2


def training_epochs(results_dir, train_info):
    """Eğitimin gerçekte kaç epoch sürdüğünü ve en iyi epoch'u eğitim eğrisinden bulur.

    En iyi epoch: Ultralytics'in best.pt'yi seçtiği ölçüt, yani val mAP50-95'in en yüksek olduğu epoch.
    """
    curve_path = results_dir / "training_curve.csv"
    if not curve_path.exists():
        return train_info.get("epochs_run", ""), train_info.get("best_epoch", "")
    curve = pd.read_csv(curve_path)
    best = int(curve.loc[curve["metrics/mAP50-95(B)"].idxmax(), "epoch"])
    return int(curve["epoch"].max()), best


def draw_example(image_path, error, out_path, scale=2):
    """Görüntünün üstüne gerçek kutuları ve model kutularını çizer (2 kat büyütülmüş)."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((img.width * scale, img.height * scale))
    draw = ImageDraw.Draw(img)
    for box, found in zip(error["gt"], error["gt_found"]):
        color = (0, 200, 0) if found else (255, 210, 0)  # yeşil: bulundu, sarı: kaçırıldı
        draw.rectangle([float(v) * scale for v in box], outline=color, width=3)
    for box, score, ok, near in zip(error["pred"], error["scores"], error["pred_ok"], error["fp_near"]):
        if ok:
            color = (40, 120, 255)   # mavi: doğru tespit
        elif near:
            color = (255, 0, 255)    # mor: cisimde fazladan kutu (parça / ikinci kutu)
        else:
            color = (255, 0, 0)      # kırmızı: boş zeminde fazladan kutu
        scaled = [float(v) * scale for v in box]
        draw.rectangle(scaled, outline=color, width=2)
        draw.text((scaled[0] + 3, scaled[1] + 3), f"{score:.2f}", fill=color)
    img.save(out_path, quality=90)


def save_error_examples(matched, part, data_dir, split, out_dir, conf):
    """Kutuları çizilmiş örnek hata görüntülerini kaydeder (yalnızca val)."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    part = part.copy()
    part["fazladan_kutu"] = [int((~matched[i]["pred_ok"]).sum()) for i in part["image_id"]]
    part["kacirilan_cisim"] = [int((~matched[i]["gt_found"]).sum()) for i in part["image_id"]]

    # Önce aydınlık grubu, sonra 96 piksel üstü, sonra kalanlar
    image_dir = data_dir / "images" / split
    used = set()
    for group_name, mask in ERROR_GROUPS:
        sub = part if mask is None else part[mask(part)]
        for kind in ("fazladan_kutu", "kacirilan_cisim"):
            candidates = sub[(sub[kind] > 0) & ~sub["image_id"].isin(used)].sort_values("image_id")
            if len(candidates) > EXAMPLES_PER_KIND:  # videoya yayılmış örnekler seç, art arda kareleri değil
                picks = np.linspace(0, len(candidates) - 1, EXAMPLES_PER_KIND).round().astype(int)
                candidates = candidates.iloc[picks]
            for image_id in candidates["image_id"]:
                draw_example(image_dir / f"{image_id}.jpg", matched[image_id], out_dir / f"{group_name}_{kind}_{image_id}.jpg")
                used.add(image_id)

    (out_dir / "ACIKLAMA.txt").write_text(
        f"Hata örnekleri (yalnızca val), güven eşiği {conf:.3f}. Görüntüler 2 kat büyütülmüştür.\n"
        "Yeşil kutu  : gerçek cisim, model buldu\n"
        "Sarı kutu   : gerçek cisim, model KAÇIRDI\n"
        "Mavi kutu   : modelin doğru tespiti (yanındaki sayı güven puanı)\n"
        "Mor kutu    : CİSİMDE fazladan kutu (cismin parçası ya da ikinci kutu)\n"
        "Kırmızı kutu: BOŞ ZEMİNDE fazladan kutu\n"
        f"Bir model kutusu, gerçek kutuyla en az %{100 * MATCH_IOU:.0f} örtüşürse doğru sayılır.\n"
        f"Fazladan kutu, gerçek kutuyla kesişiyor ya da ona {NEAR_PX} pikselden yakınsa 'cisimde' sayılır.\n"
        "Dosya adı: <grup>_<hata türü>_<görüntü no>.jpg\n"
        "Hata sayıları: val_breakdown.csv (her grup ve her video parçası için).\n",
        encoding="utf-8",
    )
    return len(used)


def weighted_light_average(rows, table):
    """Işık koşulu sonuçlarının, bütün verideki koşul dağılımına göre ağırlıklı ortalaması."""
    weights = table["isik"].value_counts(normalize=True)  # bütün veri: train + val + test
    light_rows = [r for r in rows if r["kirilim"] == "ışık koşulu"]
    present = {r["grup"] for r in light_rows}
    missing = sorted(set(weights.index) - present)
    total_weight = sum(weights[r["grup"]] for r in light_rows)

    row = {
        "kirilim": "ışık koşulu",
        "grup": "ağırlıklı ortalama (bütün verinin dağılımı)",
        "kare": sum(r["kare"] for r in light_rows),
        "kutu": sum(r["kutu"] for r in light_rows),
        "video_parcasi": sum(r["video_parcasi"] for r in light_rows),
        "not": "kaba tahmin" + (f"; bu bölümde olmayan koşul: {', '.join(missing)}" if missing else ""),
    }
    for m in METRICS:
        row[m] = sum(weights[r["grup"]] * r[m] for r in light_rows) / total_weight
    return row, {k: float(v) for k, v in weights.items()}


def update_summary(results_root, row):
    """summary.csv dosyasına deneyin satırını yazar (aynı deney + bölüm varsa yenisiyle değiştirir)."""
    path = results_root / "summary.csv"
    if path.exists():
        summary = pd.read_csv(path)
        if "epochs" in summary.columns:  # eski biçim: tek "epochs" sütunu = istenen en fazla epoch
            summary = summary.rename(columns={"epochs": "epochs_max"})
        summary = summary.reindex(columns=SUMMARY_COLUMNS)
        summary["esik_kurali"] = summary["esik_kurali"].astype(object)  # yazı sütunu
        # Eski satırlarda eksikse: epoch bilgisi eğitim eğrisinden, eşik deneyin eşik dosyasından doldurulur
        for k, old in summary.iterrows():
            exp_dir = results_root / str(old["exp_id"])
            if pd.isna(old["epochs_run"]) and (exp_dir / "training_curve.csv").exists():
                summary.loc[k, ["epochs_run", "best_epoch"]] = training_epochs(exp_dir, {})
            if pd.isna(old["conf"]) and (exp_dir / "conf_threshold.json").exists():
                saved = json.loads((exp_dir / "conf_threshold.json").read_text(encoding="utf-8"))
                if "conf_f2" not in saved:  # eski sürüm: yalnızca F1 eşiği vardı
                    summary.loc[k, ["conf", "esik_kurali"]] = [round(saved["conf"], 3), "F1"]
        summary = summary[~((summary["exp_id"] == row["exp_id"]) & (summary["split"] == row["split"]))]
    else:
        summary = pd.DataFrame(columns=SUMMARY_COLUMNS)
    summary = pd.concat([summary, pd.DataFrame([row])[SUMMARY_COLUMNS]], ignore_index=True)
    for column in ("epochs_max", "epochs_run", "best_epoch"):  # 26.0 değil 26 yazılsın
        summary[column] = pd.to_numeric(summary[column], errors="coerce").astype("Int64")
    summary.to_csv(path, index=False)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "bilinmiyor"


def main():
    parser = argparse.ArgumentParser(description="FOD: model değerlendirme ve kırılım raporu")
    parser.add_argument("--name", required=True, help="Deney adı (train.py'deki --name ile aynı)")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--final-test", action="store_true", help="Test değerlendirmesi için zorunlu onay")
    parser.add_argument("--runs-dir", required=True, help="Eğitim kayıtlarının olduğu klasör")
    parser.add_argument("--weights", default=None, help="Model dosyası (varsayılan: <runs-dir>/<name>/weights/best.pt)")
    parser.add_argument("--data-dir", default="data/yolo")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--imgsz", type=int, default=None, help="Varsayılan: eğitimdeki görüntü boyutu")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--note", default="", help="summary.csv'ye yazılacak not")
    parser.add_argument("--split-label", default=None,
                        help="Dosya adlarında ve summary.csv'de bölümün adı (varsayılan: --split), örn. sizintili_test")
    parser.add_argument("--test-log", default="test_log.csv", help="Test değerlendirmelerinin log dosyası (results-dir içinde)")
    args = parser.parse_args()
    label = args.split_label or args.split

    data_dir = Path(args.data_dir).resolve()
    if args.split == "test" and data_dir.name != "yolo" and (label == "test" or args.test_log == "test_log.csv"):
        raise SystemExit(
            f"DUR: {data_dir} bizim veri klasörümüz değil. Bu klasörün test'i için --split-label ve --test-log "
            "ayrı verilmeli (örn. --split-label sizintili_test --test-log test_log_sizintili.csv)."
        )
    results_root = Path(args.results_dir)
    results_dir = results_root / args.name
    results_dir.mkdir(parents=True, exist_ok=True)
    weights = Path(args.weights) if args.weights else Path(args.runs_dir) / args.name / "weights" / "best.pt"
    if not weights.exists():
        raise SystemExit(f"Model dosyası bulunamadı: {weights}")

    # Test koruması ve test eşiği (eşik val'de belirlenmiş olmalı)
    test_log = results_root / args.test_log
    threshold_file = results_dir / "conf_threshold.json"
    if args.split == "test":
        if not args.final_test:
            raise SystemExit(
                "DUR: Test seti yalnızca nihai model için, bir kez kullanılır.\n"
                "Gerçekten nihai değerlendirmeyse komuta --final-test ekle."
            )
        if test_log.exists() and args.name in set(pd.read_csv(test_log)["exp_id"]):
            raise SystemExit(f"DUR: '{args.name}' deneyi test setinde zaten değerlendirilmiş (bkz. {test_log}).")
        saved = json.loads(threshold_file.read_text(encoding="utf-8")) if threshold_file.exists() else {}
        if "conf_f2" not in saved:
            raise SystemExit(
                f"DUR: {threshold_file} içinde F2 eşiği yok. Önce bu deneyi bu sürümle --split val ile değerlendir."
            )

    train_info = {}
    if (results_dir / "train_info.json").exists():
        train_info = json.loads((results_dir / "train_info.json").read_text(encoding="utf-8"))
    imgsz = args.imgsz or train_info.get("imgsz", 320)
    epochs_run, best_epoch = training_epochs(results_dir, train_info)

    table, boxes = load_image_table(data_dir)
    part = table[table["split"] == args.split]
    # Video parçası kırılımı, eşik tablosu ve hata örnekleri yalnızca val içindir
    groups = make_groups(part, per_segment=args.split == "val")

    model = YOLO(str(weights))
    work_dir = (data_dir / "eval_tmp" / f"{args.name}_{label}").resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    # 1) mAP: her grup Ultralytics ile ölçülür (eşikten bağımsız)
    results = []
    for k, (title, value, sub) in enumerate(groups):
        print(f"mAP ölçülüyor: {title} / {value} ({len(sub)} kare)")
        results.append(run_val(model, sub, data_dir, args.split, work_dir, f"g{k:02d}", imgsz, args.batch))

    # 2) Modelin bütün kutuları bir kez toplanır; eşik ve hata sayıları bunlardan hesaplanır
    print("Tahminler toplanıyor...")
    predictions = predict_all(model, part, boxes, data_dir, args.split, work_dir, imgsz, args.batch)

    # 3) Güven eşiği: val'de kurala göre seçilir, test'te val'deki eşik kullanılır
    if args.split == "val":
        overall = results[0]
        conf_f1 = float(overall["px"][int(overall["f1_curve"].argmax())])
        print("Eşik tablosu hesaplanıyor...")
        thresholds, conf_f2 = threshold_table(predictions, part, conf_f1)
        thresholds.to_csv(results_dir / "val_thresholds.csv", index=False)
        threshold_file.write_text(json.dumps({
            "conf_f2": conf_f2,
            "conf_f1": conf_f1,
            "kural": "val'in tamamında F2'yi en yüksek yapan eşik (README, Güven eşiği kuralı)",
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    conf = json.loads(threshold_file.read_text(encoding="utf-8"))["conf_f2"]

    # 4) Kırılım tablosu: mAP + bu eşikte precision, recall ve hata sayıları
    matched = {i: match(p, conf) for i, p in predictions.items()}
    rows = []
    for (title, value, sub), result in zip(groups, results):
        n_segments = sub["segment_id"].nunique()
        c = count(matched, list(sub["image_id"]))
        rows.append({
            "kirilim": title,
            "grup": value,
            "kare": len(sub),
            "kutu": int(sub["n_boxes"].sum()),
            "video_parcasi": n_segments,
            "mAP50": result["mAP50"],
            "mAP50_95": result["mAP50_95"],
            **{key: c[key] for key in ["precision", "recall"] + COUNTS},
            "not": "az veri" if n_segments < MIN_SEGMENTS and title != "video parçası" else "",
        })

    weighted_row, light_weights = weighted_light_average(rows, table)
    rows.append(weighted_row)

    columns = ["kirilim", "grup", "kare", "kutu", "video_parcasi"] + METRICS + COUNTS + ["not"]
    breakdown = pd.DataFrame(rows)[columns]
    breakdown[METRICS] = breakdown[METRICS].round(4)
    breakdown[COUNTS] = breakdown[COUNTS].astype("Int64")  # ağırlıklı ortalama satırında boş kalır
    breakdown.to_csv(results_dir / f"{label}_breakdown.csv", index=False)

    overall_row = rows[0]
    metrics_json = {
        "exp_id": args.name,
        "split": label,
        "data_dir": data_dir.name,
        "weights": str(weights),
        "imgsz": imgsz,
        "conf_threshold": conf,
        "esik_kurali": "F2",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "git_commit": git_commit(),
        "epochs_max": train_info.get("epochs", ""),
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "light_weights": light_weights,
    }
    metrics_json.update({m: round(overall_row[m], 4) for m in METRICS})
    metrics_json.update({key: overall_row[key] for key in COUNTS})
    (results_dir / f"{label}_metrics.json").write_text(json.dumps(metrics_json, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_row = {
        "exp_id": args.name,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "model": train_info.get("model", weights.name),
        "imgsz": imgsz,
        "epochs_max": train_info.get("epochs", ""),
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "split": label,
        "conf": conf,
        "esik_kurali": "F2",
        "notes": args.note,
    }
    summary_row.update({m: round(overall_row[m], 4) for m in METRICS})
    update_summary(results_root, summary_row)

    n_examples = 0
    if args.split == "val":
        print("Hata örnekleri çiziliyor...")
        n_examples = save_error_examples(matched, part, data_dir, args.split, results_dir / "val_errors", conf)

    if args.split == "test":
        entry = pd.DataFrame([{"exp_id": args.name, "date": metrics_json["date"], "weights": str(weights), "git_commit": metrics_json["git_commit"]}])
        entry.to_csv(test_log, mode="a", header=not test_log.exists(), index=False)

    shutil.rmtree(work_dir)

    print(f"\nBölüm: {label} ({data_dir.name})   Güven eşiği (val'de F2 kuralıyla belirlendi): {conf:.3f}   Görüntü boyutu: {imgsz}")
    print(breakdown.to_string(index=False))
    if args.split == "val":
        print(f"\nGüven eşiğine göre precision, recall ve hata sayıları (val, tümü; IoU {MATCH_IOU} ile eşleştirme):")
        print(thresholds[thresholds["grup"] == "tümü"].drop(columns="grup").to_string(index=False))
        print(f"\n{n_examples} örnek görüntü kaydedildi: {results_dir / 'val_errors'}")
    print(f"\nSonuçlar: {results_dir}")


if __name__ == "__main__":
    main()
