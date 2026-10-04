"""train.py ve evaluate.py'nin ortak kullandığı yardımcılar: YOLO veri ayar dosyalarını yazar."""

from pathlib import Path

import yaml

CONFIG_FILE = Path(__file__).resolve().parent.parent / "configs" / "fod.yaml"


def write_train_yaml(data_dir, out_path):
    """configs/fod.yaml dosyasına veri klasörünün tam yolunu ekleyip eğitim için yazar.

    İçinde yalnızca train ve val vardır; test yolu hiç yazılmaz.
    """
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    if "test" in config:
        raise SystemExit(f"DUR: {CONFIG_FILE} içinde test yolu var. Eğitim ayarında test olmamalı.")
    config["path"] = str(Path(data_dir).resolve())
    out_path = Path(out_path)
    out_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out_path


def write_eval_yaml(data_dir, image_list, out_path):
    """Belirli bir görüntü listesini ölçmek için veri ayarı yazar.

    image_list: her satırında bir görüntünün tam yolu olan .txt dosyası.
    YOLO ayar dosyasında "train" de ister; ölçümde kullanılmaz, aynı liste yazılır.
    """
    names = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))["names"]
    config = {
        "path": str(Path(data_dir).resolve()),
        "train": str(Path(image_list).resolve()),
        "val": str(Path(image_list).resolve()),
        "names": names,
    }
    out_path = Path(out_path)
    out_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out_path
