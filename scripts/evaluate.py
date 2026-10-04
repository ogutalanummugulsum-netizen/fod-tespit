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
from ultralytics import YOLO

from fod_data import write_eval_yaml

SIZE_BINS = [0, 16, 32, 64, 96, 100000]
SIZE_LABELS = ["<16 px", "16-32 px", "32-64 px", "64-96 px", ">=96 px"]
MIN_SEGMENTS = 3  # bundan az video parçasına dayanan satır "az veri" diye işaretlenir
METRICS = ["mAP50", "mAP50_95", "precision", "recall"]
SUMMARY_COLUMNS = ["exp_id", "date", "model", "imgsz", "epochs", "split", "mAP50", "mAP50_95", "precision", "recall", "notes"]


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
    ).reset_index()

    table = image_splits.merge(per_image, on="image_id")
    table["isik"] = table["weather"] + "-" + table["light"]
    table["sinif_grubu"] = np.where(table["unseen"], "eğitimde yok", "eğitimde var")
    table["boyut"] = pd.cut(table["min_side"], bins=SIZE_BINS, labels=SIZE_LABELS, right=False).astype(str)
    return table


def make_groups(part):
    """Ölçülecek grupları sıralar: [(kırılım, grup, o gruptaki görüntüler), ...]"""
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
    return groups


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

    table = load_image_table(data_dir)
    part = table[table["split"] == args.split]
    groups = make_groups(part)

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
            "not": "az veri" if n_segments < MIN_SEGMENTS else "",
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

    if args.split == "test":
        entry = pd.DataFrame([{"exp_id": args.name, "date": metrics_json["date"], "weights": str(weights), "git_commit": metrics_json["git_commit"]}])
        entry.to_csv(test_log, mode="a", header=not test_log.exists(), index=False)

    shutil.rmtree(work_dir)

    print(f"\nBölüm: {args.split}   Güven eşiği (val'de belirlendi): {conf:.3f}   Görüntü boyutu: {imgsz}")
    print(breakdown.to_string(index=False))
    print(f"\nSonuçlar: {results_dir}")


if __name__ == "__main__":
    main()
