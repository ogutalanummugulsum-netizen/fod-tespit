"""Eğitilmiş bir modeli val ya da test üzerinde ölçer ve kırılımlarıyla raporlar.

Kullanım (val):
    python scripts/evaluate.py --name exp001_nano_deneme --split val \
        --runs-dir /content/drive/MyDrive/fod_proje/runs \
        --results-dir /content/drive/MyDrive/fod_proje/results

Kullanım (test, yalnızca nihai model için ve bir kez):
    python scripts/evaluate.py --name <deney> --split test --final-test --runs-dir ... --results-dir ...

Ölçümler: mAP50, mAP50-95, precision, recall.
Kırılımlar: tümü, sınıf grubu (eğitimde var / yok), ışık koşulu, kutu boyutu.
Her satırda kaç kareye, kaç kutuya ve kaç video parçasına dayandığı yazar.

Precision ve recall sabit bir güven eşiğiyle ölçülür. Eşik, val'in tamamında F1'i en
yüksek yapan değerdir; val değerlendirmesinde bir kez belirlenir, dosyaya yazılır ve
bütün kırılımlarda ve test'te aynısı kullanılır. Böylece gruplar birbiriyle karşılaştırılabilir.

Test koruması: test için --final-test gerekir, her test değerlendirmesi
<results-dir>/test_log.csv dosyasına yazılır ve aynı deney için ikinci kez çalışmaz.

Ürettiği dosyalar (<results-dir>/<name>/ altında):
    <split>_breakdown.csv   -> kırılım tablosu
    <split>_metrics.json    -> genel sonuç, eşik ve ağırlıklar
    conf_threshold.json     -> val'de belirlenen güven eşiği
    val_errors/             -> yalnızca val: hata sayıları ve kutuları çizilmiş örnek görüntüler
ve <results-dir>/summary.csv dosyasına bir satır.

Yalnızca val'de ek olarak: her video parçası için ayrı bir kırılım satırı ve hata örnekleri
(fazladan kutu ve kaçırılan cisim). Test raporunda bunlar yoktur.
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
SUMMARY_COLUMNS = ["exp_id", "date", "model", "imgsz", "epochs", "split", "mAP50", "mAP50_95", "precision", "recall", "notes"]

# Hata örnekleri (yalnızca val): hangi gruplardan, her türden en fazla kaç görüntü
ERROR_GROUPS = [
    ("aydinlik", lambda t: t["isik"] == "Dry-Bright"),
    ("96px_ustu", lambda t: t["boyut"] == ">=96 px"),
    ("diger", None),
]
EXAMPLES_PER_KIND = 8
MATCH_IOU = 0.5  # model kutusu, gerçek kutuyla en az bu kadar örtüşürse doğru sayılır


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


def find_errors(model, part, boxes, data_dir, split, work_dir, conf, imgsz, batch):
    """Her görüntüde modelin kutularını gerçek kutularla eşleştirir.

    Fazladan kutu: hiçbir gerçek kutuyla eşleşmeyen model kutusu (yanlış tespit).
    Kaçırılan cisim: hiçbir model kutusuyla eşleşmeyen gerçek kutu.
    """
    image_dir = (data_dir / "images" / split).resolve()
    source = work_dir / "predict_list.txt"
    source.write_text("\n".join(str(image_dir / f"{i}.jpg") for i in part["image_id"]) + "\n", encoding="utf-8")

    in_part = boxes[boxes["image_id"].isin(set(part["image_id"]))]
    gt_boxes = {i: g[["xmin", "ymin", "xmax", "ymax"]].values.astype(float) for i, g in in_part.groupby("image_id")}

    errors = {}
    for result in model.predict(source=str(source), conf=conf, imgsz=imgsz, batch=batch, stream=True, verbose=False):
        image_id = Path(result.path).stem
        pred = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        gt = gt_boxes.get(image_id, np.zeros((0, 4)))
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
        errors[image_id] = {"pred": pred, "scores": scores, "pred_ok": pred_ok, "gt": gt, "gt_found": gt_found}
    return errors


def draw_example(image_path, error, out_path, scale=2):
    """Görüntünün üstüne gerçek kutuları ve model kutularını çizer (2 kat büyütülmüş)."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((img.width * scale, img.height * scale))
    draw = ImageDraw.Draw(img)
    for box, found in zip(error["gt"], error["gt_found"]):
        color = (0, 200, 0) if found else (255, 210, 0)  # yeşil: bulundu, sarı: kaçırıldı
        draw.rectangle([float(v) * scale for v in box], outline=color, width=3)
    for box, score, ok in zip(error["pred"], error["scores"], error["pred_ok"]):
        color = (40, 120, 255) if ok else (255, 0, 0)  # mavi: doğru tespit, kırmızı: fazladan kutu
        scaled = [float(v) * scale for v in box]
        draw.rectangle(scaled, outline=color, width=2)
        draw.text((scaled[0] + 3, scaled[1] + 3), f"{score:.2f}", fill=color)
    img.save(out_path, quality=90)


def save_error_examples(errors, part, data_dir, split, out_dir):
    """Hata sayılarını ve örnek görüntüleri kaydeder."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    part = part.copy()
    part["fazladan_kutu"] = [int((~errors[i]["pred_ok"]).sum()) if i in errors else 0 for i in part["image_id"]]
    part["kacirilan_cisim"] = [int((~errors[i]["gt_found"]).sum()) if i in errors else 0 for i in part["image_id"]]

    # Hata sayıları: kırılım kırılım ve video parçası bazında
    count_rows = []
    for column, title in (("isik", "ışık koşulu"), ("boyut", "kutu boyutu"), ("segment_id", "video parçası")):
        for value, sub in part.groupby(column, sort=True):
            label = f"{value} {sub['classes'].iloc[0]} {sub['isik'].iloc[0]}" if column == "segment_id" else value
            count_rows.append({
                "kirilim": title,
                "grup": label,
                "kare": len(sub),
                "fazladan_kutulu_kare": int((sub["fazladan_kutu"] > 0).sum()),
                "kacirilan_cisimli_kare": int((sub["kacirilan_cisim"] > 0).sum()),
                "fazladan_kutu": int(sub["fazladan_kutu"].sum()),
                "kacirilan_cisim": int(sub["kacirilan_cisim"].sum()),
            })
    counts = pd.DataFrame(count_rows)
    counts.to_csv(out_dir / "error_counts.csv", index=False)

    # Örnek görüntüler: önce aydınlık grubu, sonra 96 piksel üstü, sonra kalanlar
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
                draw_example(image_dir / f"{image_id}.jpg", errors[image_id], out_dir / f"{group_name}_{kind}_{image_id}.jpg")
                used.add(image_id)

    (out_dir / "ACIKLAMA.txt").write_text(
        "Hata örnekleri (yalnızca val). Görüntüler 2 kat büyütülmüştür.\n"
        "Yeşil kutu  : gerçek cisim, model buldu\n"
        "Sarı kutu   : gerçek cisim, model KAÇIRDI\n"
        "Mavi kutu   : modelin doğru tespiti (yanındaki sayı güven puanı)\n"
        "Kırmızı kutu: modelin FAZLADAN kutusu (yanlış tespit)\n"
        f"Bir model kutusu, gerçek kutuyla en az %{100 * MATCH_IOU:.0f} örtüşürse doğru sayılır.\n"
        "Dosya adı: <grup>_<hata türü>_<görüntü no>.jpg\n"
        "error_counts.csv: her grupta ve her video parçasında kaç hata olduğu.\n",
        encoding="utf-8",
    )
    return counts, len(used)


def run_val(model, sub, data_dir, split, work_dir, tag, imgsz, batch):
    """Bir görüntü grubunu YOLO ile ölçer; mAP değerlerini ve güven eşiği eğrilerini döner."""
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
    if len(np.asarray(box.p_curve)) == 0:
        raise SystemExit(f"'{tag}' grubunda ölçüm yapılamadı: model hiç tahmin üretmedi ya da etiket bulunamadı.")
    return {
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "px": np.asarray(box.px),                  # güven eşikleri (0-1 arası 1000 nokta)
        "p_curve": np.asarray(box.p_curve)[0],     # her eşikte precision
        "r_curve": np.asarray(box.r_curve)[0],     # her eşikte recall
        "f1_curve": np.asarray(box.f1_curve)[0],   # her eşikte F1
    }


def at_threshold(result, conf):
    """Verilen güven eşiğindeki precision ve recall."""
    k = int(np.abs(result["px"] - conf).argmin())
    return float(result["p_curve"][k]), float(result["r_curve"][k])


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
        summary = summary[~((summary["exp_id"] == row["exp_id"]) & (summary["split"] == row["split"]))]
    else:
        summary = pd.DataFrame(columns=SUMMARY_COLUMNS)
    summary = pd.concat([summary, pd.DataFrame([row])[SUMMARY_COLUMNS]], ignore_index=True)
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
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    results_root = Path(args.results_dir)
    results_dir = results_root / args.name
    results_dir.mkdir(parents=True, exist_ok=True)
    weights = Path(args.weights) if args.weights else Path(args.runs_dir) / args.name / "weights" / "best.pt"
    if not weights.exists():
        raise SystemExit(f"Model dosyası bulunamadı: {weights}")

    # Test koruması
    test_log = results_root / "test_log.csv"
    if args.split == "test":
        if not args.final_test:
            raise SystemExit(
                "DUR: Test seti yalnızca nihai model için, bir kez kullanılır.\n"
                "Gerçekten nihai değerlendirmeyse komuta --final-test ekle."
            )
        if test_log.exists() and args.name in set(pd.read_csv(test_log)["exp_id"]):
            raise SystemExit(f"DUR: '{args.name}' deneyi test setinde zaten değerlendirilmiş (bkz. {test_log}).")

    train_info = {}
    if (results_dir / "train_info.json").exists():
        train_info = json.loads((results_dir / "train_info.json").read_text(encoding="utf-8"))
    imgsz = args.imgsz or train_info.get("imgsz", 320)

    table, boxes = load_image_table(data_dir)
    part = table[table["split"] == args.split]
    # Video parçası kırılımı ve hata örnekleri yalnızca val içindir; test raporu README'deki kırılımlarla sınırlıdır
    groups = make_groups(part, per_segment=args.split == "val")

    model = YOLO(str(weights))
    work_dir = (data_dir / "eval_tmp" / f"{args.name}_{args.split}").resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    results = []
    for k, (title, value, sub) in enumerate(groups):
        print(f"Ölçülüyor: {title} / {value} ({len(sub)} kare)")
        results.append(run_val(model, sub, data_dir, args.split, work_dir, f"g{k:02d}", imgsz, args.batch))

    # Güven eşiği: val'de belirlenir, test'te aynısı kullanılır
    threshold_file = results_dir / "conf_threshold.json"
    if args.split == "val":
        overall = results[0]
        conf = float(overall["px"][int(overall["f1_curve"].argmax())])
        threshold_file.write_text(json.dumps({"conf": conf, "kaynak": "val'in tamamında F1'i en yüksek yapan eşik"}, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        if not threshold_file.exists():
            raise SystemExit(f"Güven eşiği bulunamadı: {threshold_file}. Önce bu deneyi --split val ile değerlendir.")
        conf = json.loads(threshold_file.read_text(encoding="utf-8"))["conf"]

    rows = []
    for (title, value, sub), result in zip(groups, results):
        precision, recall = at_threshold(result, conf)
        n_segments = sub["segment_id"].nunique()
        rows.append({
            "kirilim": title,
            "grup": value,
            "kare": len(sub),
            "kutu": int(sub["n_boxes"].sum()),
            "video_parcasi": n_segments,
            "mAP50": result["mAP50"],
            "mAP50_95": result["mAP50_95"],
            "precision": precision,
            "recall": recall,
            "not": "az veri" if n_segments < MIN_SEGMENTS and title != "video parçası" else "",
        })

    weighted_row, light_weights = weighted_light_average(rows, table)
    rows.append(weighted_row)

    columns = ["kirilim", "grup", "kare", "kutu", "video_parcasi"] + METRICS + ["not"]
    breakdown = pd.DataFrame(rows)[columns]
    breakdown[METRICS] = breakdown[METRICS].round(4)
    breakdown.to_csv(results_dir / f"{args.split}_breakdown.csv", index=False)

    overall_row = rows[0]
    metrics_json = {
        "exp_id": args.name,
        "split": args.split,
        "weights": str(weights),
        "imgsz": imgsz,
        "conf_threshold": conf,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "git_commit": git_commit(),
        "light_weights": light_weights,
    }
    metrics_json.update({m: round(overall_row[m], 4) for m in METRICS})
    (results_dir / f"{args.split}_metrics.json").write_text(json.dumps(metrics_json, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_row = {
        "exp_id": args.name,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "model": train_info.get("model", weights.name),
        "imgsz": imgsz,
        "epochs": train_info.get("epochs", ""),
        "split": args.split,
        "notes": args.note,
    }
    summary_row.update({m: round(overall_row[m], 4) for m in METRICS})
    update_summary(results_root, summary_row)

    n_examples = 0
    if args.split == "val":
        print("Hata örnekleri çıkarılıyor...")
        errors = find_errors(model, part, boxes, data_dir, args.split, work_dir, conf, imgsz, args.batch)
        error_counts, n_examples = save_error_examples(errors, part, data_dir, args.split, results_dir / "val_errors")

    if args.split == "test":
        entry = pd.DataFrame([{"exp_id": args.name, "date": metrics_json["date"], "weights": str(weights), "git_commit": metrics_json["git_commit"]}])
        entry.to_csv(test_log, mode="a", header=not test_log.exists(), index=False)

    shutil.rmtree(work_dir)

    print(f"\nBölüm: {args.split}   Güven eşiği (val'de belirlendi): {conf:.3f}   Görüntü boyutu: {imgsz}")
    print(breakdown.to_string(index=False))
    if args.split == "val":
        print(f"\nHata sayıları (güven eşiği {conf:.3f}, eşleşme IoU {MATCH_IOU}):")
        print(error_counts.to_string(index=False))
        print(f"\n{n_examples} örnek görüntü kaydedildi: {results_dir / 'val_errors'}")
    print(f"\nSonuçlar: {results_dir}")


if __name__ == "__main__":
    main()
