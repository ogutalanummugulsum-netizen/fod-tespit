"""Ortak kare karşılaştırması: sızıntının etkisini AYNI karelerde ölçer (README, "Sızıntılı ayrım karşılaştırması").

Kullanım (yalnızca bir kez; önce iki modelin kendi test değerlendirmeleri bitmiş olmalı):
    python scripts/compare_leak.py --final-test-subset \
        --runs-dir /content/drive/MyDrive/fod_proje/runs \
        --results-dir /content/drive/MyDrive/fod_proje/results

Ortak kareler: hem hazır test'te hem bizim test setimizde olan kareler (make_leaky_split.py yazar).
    - exp002_nano (dürüst): bu karelerin video parçalarını hiç görmedi.
    - exp004_nano_sizintili (sızıntılı): aynı video parçalarının komşu karelerini eğitimde gördü.
Aynı karelerdeki fark doğrudan sızıntının etkisidir.

Test kuralına istisna (README'de sonuç görülmeden yazıldı): exp002 bizim test setinde yalnızca bu
ortak karelerde yeniden çalıştırılır. Model ve eşik sabittir, sonuç hiçbir seçimi değiştirmez.
Korumalar:
    - --final-test-subset zorunlu.
    - İki modelin kendi test değerlendirmesi önceden yapılmış olmalı (test_log.csv, test_log_sizintili.csv).
    - Yalnızca bir kez çalışır (results/leak_compare_log.csv).
    - Video parçası kırılımı, hata görüntüsü ve eşik tablosu çıkarılmaz; eşikler val'de belirlenmiş olanlardır.

Ürettiği dosyalar (<results-dir>/leak_compare/):
    ortak_kare_breakdown.csv -> her model için: tümü, sınıf grubu, ışık koşulu, kutu boyutu
    ortak_kare_eslesme.csv   -> aynı cisim için: ikisi de buldu / yalnız dürüst / yalnız sızıntılı / ikisi de kaçırdı
    ortak_kare_metrics.json  -> özet
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from evaluate import COUNTS, METRICS, MIN_SEGMENTS, count, git_commit, load_image_table, make_groups, match, predict_all, run_val


def main():
    parser = argparse.ArgumentParser(description="FOD: ortak karelerde dürüst ve sızıntılı model karşılaştırması")
    parser.add_argument("--final-test-subset", action="store_true", help="Zorunlu onay: bizim test setinin ortak karelerine dokunur")
    parser.add_argument("--honest", default="exp002_nano")
    parser.add_argument("--leaky", default="exp004_nano_sizintili")
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--data-dir", default="data/yolo", help="Bizim veri klasörümüz")
    parser.add_argument("--leaky-data-dir", default="data/yolo_sizintili")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    if not args.final_test_subset:
        raise SystemExit("DUR: Bu karşılaştırma bizim test setinin ortak karelerine dokunur. Bir kez yapılır; --final-test-subset ekle.")

    results_root = Path(args.results_dir)
    out_dir = results_root / "leak_compare"
    log_file = results_root / "leak_compare_log.csv"
    if log_file.exists() and len(pd.read_csv(log_file)):
        raise SystemExit(f"DUR: ortak kare karşılaştırması zaten yapılmış (bkz. {log_file}).")

    # Önce iki modelin kendi test değerlendirmesi bitmiş olmalı (modeller ve eşikler artık sabit)
    for name, log_name in ((args.honest, "test_log.csv"), (args.leaky, "test_log_sizintili.csv")):
        path = results_root / log_name
        if not path.exists() or name not in set(pd.read_csv(path)["exp_id"]):
            raise SystemExit(f"DUR: {name} için {log_name} kaydı yok. Önce modelin kendi test değerlendirmesi yapılmalı.")

    data_dir = Path(args.data_dir).resolve()
    common_ids = set((Path(args.leaky_data_dir) / "ortak_kareler.txt").read_text(encoding="utf-8").split())
    table, boxes = load_image_table(data_dir)
    part = table[(table["split"] == "test") & table["image_id"].isin(common_ids)]
    if len(part) != len(common_ids):
        raise SystemExit(f"DUR: {len(common_ids)} ortak karenin {len(part)} tanesi bizim test setinde bulundu; ikisi aynı olmalı.")
    print(f"Ortak kare: {len(part)} ({part['segment_id'].nunique()} video parçası)")
    groups = make_groups(part)  # video parçası kırılımı YOK

    work_root = (data_dir / "eval_tmp" / "leak_compare").resolve()
    if work_root.exists():
        shutil.rmtree(work_root)
    rows, matched_by_model, confs = [], {}, {}
    for role, name in (("dürüst", args.honest), ("sızıntılı", args.leaky)):
        conf = json.loads((results_root / name / "conf_threshold.json").read_text(encoding="utf-8"))["conf_f2"]
        confs[name] = conf
        model = YOLO(str(Path(args.runs_dir) / name / "weights" / "best.pt"))
        work_dir = work_root / name
        work_dir.mkdir(parents=True)
        print(f"\n===== {name} ({role}), güven eşiği {conf} =====")
        results = []
        for k, (title, value, sub) in enumerate(groups):
            print(f"mAP ölçülüyor: {title} / {value} ({len(sub)} kare)")
            results.append(run_val(model, sub, data_dir, "test", work_dir, f"g{k:02d}", args.imgsz, args.batch))
        print("Tahminler toplanıyor...")
        predictions = predict_all(model, part, boxes, data_dir, "test", work_dir, args.imgsz, args.batch)
        matched = {i: match(p, conf) for i, p in predictions.items()}
        matched_by_model[name] = matched
        for (title, value, sub), result in zip(groups, results):
            n_segments = sub["segment_id"].nunique()
            c = count(matched, list(sub["image_id"]))
            rows.append({
                "model": name, "rol": role, "esik": conf,
                "kirilim": title, "grup": value, "kare": len(sub), "kutu": int(sub["n_boxes"].sum()),
                "video_parcasi": n_segments,
                "mAP50": result["mAP50"], "mAP50_95": result["mAP50_95"],
                **{key: c[key] for key in ["precision", "recall"] + COUNTS},
                "not": "az veri" if n_segments < MIN_SEGMENTS else "",
            })

    breakdown = pd.DataFrame(rows)
    breakdown[METRICS] = breakdown[METRICS].round(4)

    # Aynı cisim için iki modelin sonucu (eşleştirilmiş karşılaştırma)
    honest, leaky = matched_by_model[args.honest], matched_by_model[args.leaky]
    pair_rows = []
    for title, value, sub in [g for g in groups if g[0] in ("tümü", "ışık koşulu", "kutu boyutu")]:
        both = only_h = only_l = neither = 0
        for i in sub["image_id"]:
            for h, l in zip(honest[i]["gt_found"], leaky[i]["gt_found"]):
                both += int(h and l)
                only_h += int(h and not l)
                only_l += int(l and not h)
                neither += int(not h and not l)
        pair_rows.append({"kirilim": title, "grup": value, "kutu": both + only_h + only_l + neither,
                          "ikisi_de_buldu": both, f"yalniz_{args.honest}": only_h,
                          f"yalniz_{args.leaky}": only_l, "ikisi_de_kacirdi": neither})
    pairs = pd.DataFrame(pair_rows)

    # Tutarlılık: dürüst modelin ortak karelerdeki hataları, bütün test'teki hatalarından fazla olamaz
    overall_test = json.loads((results_root / args.honest / "test_metrics.json").read_text(encoding="utf-8"))
    h_row = breakdown[(breakdown["model"] == args.honest) & (breakdown["kirilim"] == "tümü")].iloc[0]
    consistent = all(int(h_row[key]) <= int(overall_test[key]) for key in COUNTS)
    if not consistent:
        print("UYARI: dürüst modelin ortak karelerdeki hata sayısı, bütün test'tekinden fazla. Çıktıyı gönder.")

    out_dir.mkdir(parents=True, exist_ok=True)
    breakdown.to_csv(out_dir / "ortak_kare_breakdown.csv", index=False)
    pairs.to_csv(out_dir / "ortak_kare_eslesme.csv", index=False)
    date = datetime.now().strftime("%Y-%m-%d %H:%M")
    summary = {
        "tarih": date, "git_commit": git_commit(), "ortak_kare": len(part),
        "video_parcasi": int(part["segment_id"].nunique()), "esikler": confs,
        "durust_hatalari_butun_testle_tutarli": consistent,
        "tumu": breakdown[breakdown["kirilim"] == "tümü"].drop(columns=["kirilim", "grup"]).to_dict(orient="records"),
    }
    (out_dir / "ortak_kare_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    pd.DataFrame([{"tarih": date, "durust": args.honest, "sizintili": args.leaky, "ortak_kare": len(part),
                   "git_commit": summary["git_commit"]}]).to_csv(log_file, mode="a", header=not log_file.exists(), index=False)
    shutil.rmtree(work_root)

    columns = ["model", "kirilim", "grup", "kare", "video_parcasi"] + METRICS + COUNTS + ["not"]
    print("\nOrtak karelerde iki model (her biri kendi val F2 eşiğiyle):")
    print(breakdown[columns].to_string(index=False))
    print("\nAynı cisim için iki modelin sonucu:")
    print(pairs.to_string(index=False))
    print(f"\nSonuçlar: {out_dir}")


if __name__ == "__main__":
    main()
