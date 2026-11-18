"""Modellerin yalnızca işlemciyle (ekran kartı olmadan) hızını ölçer.

Kullanım:
    python scripts/benchmark_cpu.py --models exp002_nano exp003_small --where colab_cpu \
        --runs-dir /content/drive/MyDrive/fod_proje/runs \
        --results-dir /content/drive/MyDrive/fod_proje/results

Nasıl ölçülür:
    - Val'den sabit, eşit aralıklı görüntüler seçilir (iki model için aynısı). Test setine dokunulmaz.
    - İlk --warmup görüntü ısınma içindir, ölçüme katılmaz.
    - Görüntüler tek tek işlenir (canlı kamerada olduğu gibi), görüntü boyutu eğitimdeki gibi (320).
    - Model, her deneyin val'de F2 kuralıyla belirlenen güven eşiğiyle çalıştırılır.
    - Her model, --threads ile verilen her iş parçacığı sayısıyla ayrı ayrı ölçülür (örn. 1 ve 2).
      Ultralytics işlemcide bu sayıyı kendisi "çekirdek sayısı - 1" yapar; bu yüzden sayı her
      tahminden önce yeniden ayarlanır ve tahminden sonra okunarak doğrulanır.
    - Model süresi = hazırlık + model + kutu ayıklama (Ultralytics'in ölçtüğü süreler, diskten okuma hariç).
    - Uçtan uca süre = görüntüyü diskten okuma dahil, bir çağrının toplam süresi.

Ayrıca her model için dosya boyutu (MB) ve parametre sayısı kaydedilir.

Sonuç: <results-dir>/speed_cpu.csv (yalnızca aynı deney, aynı ölçüm yeri, aynı işlemci ve aynı iş parçacığı sayısı için eski
satır yenisiyle değişir; başka bir işlemcide yapılmış ölçümler korunur. Colab her oturumda farklı
bir işlemci verebilir ve hız işlemciye göre belirgin biçimde değişir.)
"""

import argparse
import json
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import ultralytics
from ultralytics import YOLO


def cpu_name():
    """İşlemcinin adı (Linux'ta /proc/cpuinfo, diğerlerinde platform bilgisi)."""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine() or "bilinmiyor"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "bilinmiyor"


def pick_images(data_dir, n):
    """Val'den eşit aralıklı n görüntü seçer (her çalıştırmada aynı görüntüler)."""
    image_splits = pd.read_csv(data_dir / "image_splits.csv", dtype={"image_id": str})
    val_ids = sorted(image_splits.loc[image_splits["split"] == "val", "image_id"])
    if n > len(val_ids):
        raise SystemExit(f"Val'de yalnızca {len(val_ids)} görüntü var, {n} istendi.")
    picks = np.linspace(0, len(val_ids) - 1, n).round().astype(int)
    return [str(data_dir / "images" / "val" / f"{val_ids[k]}.jpg") for k in picks]


def predict(model, path, threads, imgsz, conf):
    """Tek görüntü tahmini. İş parçacığı sayısı her çağrıdan önce yeniden ayarlanır, çünkü Ultralytics
    modeli ilk kez işlemcide çalıştırırken bu sayıyı kendisi değiştirir ("çekirdek sayısı - 1")."""
    torch.set_num_threads(threads)
    return model.predict(path, device="cpu", imgsz=imgsz, conf=conf, verbose=False)[0]


def benchmark(exp_id, runs_dir, results_root, images, warmup, imgsz, threads):
    """Bir modeli işlemcide, verilen iş parçacığı sayısıyla ölçer ve sonuç satırını döner."""
    weights = runs_dir / exp_id / "weights" / "best.pt"
    if not weights.exists():
        raise SystemExit(f"Model dosyası bulunamadı: {weights}")
    threshold_file = results_root / exp_id / "conf_threshold.json"
    if not threshold_file.exists():
        raise SystemExit(f"Güven eşiği bulunamadı: {threshold_file}. Önce val değerlendirmesini çalıştır.")
    conf = json.loads(threshold_file.read_text(encoding="utf-8"))["conf_f2"]
    train_info_file = results_root / exp_id / "train_info.json"
    model_name = json.loads(train_info_file.read_text(encoding="utf-8"))["model"] if train_info_file.exists() else ""

    model = YOLO(str(weights))
    n_params = sum(p.numel() for p in model.model.parameters())  # tahminden önce sayılır
    size_mb = weights.stat().st_size / (1024 * 1024)

    print(f"\n{exp_id}, {threads} iş parçacığı: {n_params / 1e6:.2f} milyon parametre, {size_mb:.1f} MB, güven eşiği {conf}")
    print(f"  ısınma: {warmup} görüntü...")
    for path in images[:warmup]:
        predict(model, path, threads, imgsz, conf)

    pre, inf, post, wall = [], [], [], []
    measured = images[warmup:]
    used_threads = set()
    for k, path in enumerate(measured):
        start = time.perf_counter()
        result = predict(model, path, threads, imgsz, conf)
        wall.append((time.perf_counter() - start) * 1000)
        used_threads.add(torch.get_num_threads())  # tahminden SONRA okunur: gerçekten kullanılan sayı
        pre.append(result.speed["preprocess"])
        inf.append(result.speed["inference"])
        post.append(result.speed["postprocess"])
        if (k + 1) % 100 == 0:
            print(f"  ölçülen: {k + 1}/{len(measured)}")

    if used_threads != {threads}:
        raise SystemExit(f"İş parçacığı sayısı istenen {threads} değil, ölçüm sırasında {sorted(used_threads)} oldu.")

    total = np.array(pre) + np.array(inf) + np.array(post)
    wall = np.array(wall)
    return {
        "exp_id": exp_id,
        "is_parcacigi": threads,
        "model": model_name,
        "parametre_sayisi": n_params,
        "dosya_boyutu_mb": round(size_mb, 2),
        "imgsz": imgsz,
        "conf": conf,
        "olculen_goruntu": len(measured),
        "hazirlik_ms": round(float(np.median(pre)), 2),
        "model_ms": round(float(np.median(inf)), 2),
        "ayiklama_ms": round(float(np.median(post)), 2),
        "toplam_ortanca_ms": round(float(np.median(total)), 2),
        "toplam_p90_ms": round(float(np.percentile(total, 90)), 2),
        "fps": round(1000 / float(np.median(total)), 1),
        "uctan_uca_ortanca_ms": round(float(np.median(wall)), 2),
        "uctan_uca_fps": round(1000 / float(np.median(wall)), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="FOD: yalnızca işlemciyle hız ölçümü")
    parser.add_argument("--models", nargs="+", required=True, help="Ölçülecek deneyler, örn. exp002_nano exp003_small")
    parser.add_argument("--where", required=True, help="Ölçüm yeri, örn. colab_cpu ya da kendi_bilgisayarim")
    parser.add_argument("--runs-dir", required=True, help="Eğitim kayıtlarının (best.pt) olduğu klasör")
    parser.add_argument("--data-dir", default="data/yolo")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--n-images", type=int, default=300, help="Ölçülecek görüntü sayısı")
    parser.add_argument("--warmup", type=int, default=20, help="Ölçüme katılmayan ısınma görüntüsü sayısı")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--threads", type=int, nargs="+", default=None,
                        help="Denenecek iş parçacığı sayıları, örn. 1 2 (varsayılan: 1 ve bütün çekirdekler)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    runs_dir = Path(args.runs_dir)
    results_root = Path(args.results_dir)
    images = pick_images(data_dir, args.n_images + args.warmup)
    thread_counts = sorted(set(args.threads or [1, os.cpu_count() or 1]))

    machine = {
        "olcum_yeri": args.where,
        "islemci": cpu_name(),
        "cekirdek": os.cpu_count(),
        "tarih": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "git_commit": git_commit(),
    }
    print("Ölçüm yeri    :", machine["olcum_yeri"])
    print("İşlemci       :", machine["islemci"], f"({machine['cekirdek']} çekirdek)")
    print("İş parçacığı  :", ", ".join(map(str, thread_counts)), "(her biri ayrı satır)")
    print("Ekran kartı kullanılmıyor: bütün tahminler device='cpu' ile yapılır.")

    rows = [
        {**machine, **benchmark(exp_id, runs_dir, results_root, images, args.warmup, args.imgsz, threads)}
        for exp_id in args.models
        for threads in thread_counts
    ]
    new = pd.DataFrame(rows)

    path = results_root / "speed_cpu.csv"
    if path.exists():
        old = pd.read_csv(path)
        if "is_parcacigi" not in old.columns:  # eski biçim: tek ölçüm, iş parçacığı sayısı başka sütunda
            old["is_parcacigi"] = old.get("torch_is_parcacigi", 1)
            old = old.drop(columns=["torch_is_parcacigi"], errors="ignore")
        same = (
            old["exp_id"].isin(new["exp_id"])
            & (old["olcum_yeri"] == args.where)
            & (old["islemci"] == machine["islemci"])  # başka işlemcide yapılmış ölçüm korunur
            & old["is_parcacigi"].isin(thread_counts)
        )
        new = pd.concat([old[~same], new], ignore_index=True)
    results_root.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)

    columns = ["exp_id", "is_parcacigi", "parametre_sayisi", "dosya_boyutu_mb", "model_ms", "toplam_ortanca_ms",
               "toplam_p90_ms", "fps", "uctan_uca_ortanca_ms", "uctan_uca_fps"]
    print(f"\nSonuç ({args.where}, {machine['islemci']}):")
    print(pd.DataFrame(rows)[columns].to_string(index=False))
    print(f"\nKaydedildi: {path}")


if __name__ == "__main__":
    main()
