"""Bir YOLO modelini train setiyle eğitir. Test setini hiç kullanmaz.

Kullanım:
    python scripts/train.py --name exp001_nano_deneme --model yolo11n.pt --epochs 3 --imgsz 320 \
        --runs-dir /content/drive/MyDrive/fod_proje/runs \
        --results-dir /content/drive/MyDrive/fod_proje/results

Ara kayıtlar (checkpoint): her epoch sonunda <runs-dir>/<name>/weights/last.pt dosyasına
yazılır. runs-dir Google Drive'da olduğu için Colab oturumu kopsa da kayıt silinmez.

Kaldığı yerden devam: aynı komutu aynı --name ile yeniden çalıştırmak yeterlidir.
    - last.pt varsa ve eğitim yarım kalmışsa: kaldığı epoch'tan devam eder.
    - last.pt varsa ve eğitim bitmişse: yeniden eğitmez, bunu söyleyip çıkar.
    - last.pt yoksa: eğitime baştan başlar.

Erken durdurma: --patience 10 verilirse, val puanı 10 epoch boyunca iyileşmezse eğitim
--epochs sayısına ulaşmadan durur. En iyi model yine best.pt olarak saklanır.

Sonuçların ölçümü bu script'te değil, evaluate.py'dedir.
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

from fod_data import write_train_yaml


def git_commit():
    """Kodun hangi commit ile çalıştığını döner (bulunamazsa 'bilinmiyor')."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "bilinmiyor"


def main():
    parser = argparse.ArgumentParser(description="FOD: YOLO eğitimi")
    parser.add_argument("--name", required=True, help="Deney adı, örn. exp001_nano_deneme")
    parser.add_argument("--model", default="yolo11n.pt", help="Başlangıç modeli (yolo11n.pt = nano, yolo11s.pt = small)")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=100, help="Val puanı bu kadar epoch iyileşmezse eğitim erken durur")
    parser.add_argument("--data-dir", default="data/yolo")
    parser.add_argument("--runs-dir", required=True, help="Ara kayıtların yazılacağı klasör (Drive'da olmalı)")
    parser.add_argument("--results-dir", default="results", help="Deney sonuçlarının yazılacağı klasör")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    runs_dir = Path(args.runs_dir).resolve()
    run_dir = runs_dir / args.name
    last = run_dir / "weights" / "last.pt"
    results_dir = Path(args.results_dir) / args.name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Eğitim ayarı her seferinde yeniden yazılır; içinde test yolu yoktur
    data_yaml = write_train_yaml(data_dir, data_dir / "fod_train.yaml")

    resumed = False
    trained = True  # bu çalıştırmada eğitim yapıldı mı
    if last.exists():
        model = YOLO(str(last))
        ckpt = model.ckpt or {}
        if ckpt.get("epoch", -1) >= 0 and ckpt.get("optimizer") is not None:
            print(f"Yarım kalmış eğitim bulundu ({ckpt['epoch'] + 1} epoch bitmiş). Kaldığı yerden devam ediyor.")
            model.train(resume=True, data=str(data_yaml))
            resumed = True
        else:
            print(f"'{args.name}' deneyinin eğitimi zaten bitmiş: {run_dir}")
            print("Yeniden eğitmedim. Farklı ayarla eğitmek için yeni bir --name ver.")
            trained = False
    else:
        model = YOLO(args.model)
        model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            seed=args.seed,
            patience=args.patience,
            project=str(runs_dir),
            name=args.name,
            exist_ok=True,
        )

    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"Eğitim bitti ama en iyi model bulunamadı: {best}")

    # Deney kaydı: ayarlar, ortam ve epoch epoch eğitim eğrisi
    info = {
        "name": args.name,
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "patience": args.patience,
        "resumed": resumed,
        "run_dir": str(run_dir),
        "best_weights": str(best),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "git_commit": git_commit(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    info_path = results_dir / "train_info.json"
    if trained or not info_path.exists():  # bitmiş bir deneyin kaydı üzerine yazılmaz
        info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    if (run_dir / "results.csv").exists():
        shutil.copy(run_dir / "results.csv", results_dir / "training_curve.csv")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    (results_dir / "environment.txt").write_text(freeze, encoding="utf-8")

    print(f"\nEn iyi model: {best}")
    print(f"Deney kaydı : {results_dir}")


if __name__ == "__main__":
    main()
