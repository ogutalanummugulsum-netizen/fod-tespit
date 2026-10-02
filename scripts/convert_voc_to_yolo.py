"""FOD-A etiketlerini Pascal VOC (XML) formatından YOLO (txt) formatına çevirir.

Kullanım:
    python scripts/convert_voc_to_yolo.py \
        --voc-dir data/raw/FODPascalVOCFormat-V.2.1/VOC2007 \
        --out-dir data/yolo

Ürettiği dosyalar (out-dir altında):
    all/labels_fod/000000.txt  -> tek sınıflı etiket (her cisim sınıf 0 = "FOD"). Ana model bunu kullanır.
    all/labels_31/000000.txt   -> 31 sınıflı etiket. İleride 31 sınıflı deney için saklanır.
    boxes.csv                  -> her kutu için bir satır: orijinal sınıf adı, kutu, hava, ışık.

Bu script görüntüleri kopyalamaz ve train/val/test ayrımı yapmaz.
O işi make_splits.py yapar. data/raw içindeki hiçbir dosya değiştirilmez.
"""

import argparse
import csv
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

# category_information.txt dosyasındaki sıraya göre
WEATHER_NAMES = ["Dry", "Wet"]
LIGHT_NAMES = ["Bright", "Dim", "Dark"]


def read_class_names(path):
    """Sınıf adlarını sabit sırayla okur. Satır numarası = 31 sınıflı etiketteki sınıf no."""
    names = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    return [n for n in names if n]


def read_weather_light(csv_path):
    """Her görüntünün hava ve ışık bilgisini okur: {"000000": ("Wet", "Dim"), ...}"""
    info = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_id = Path(row["File"]).stem
            info[image_id] = (WEATHER_NAMES[int(row["Weather"])], LIGHT_NAMES[int(row["Light"])])
    return info


def voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """Köşe koordinatlarını (piksel) YOLO'nun istediği biçime çevirir.

    YOLO biçimi: kutunun merkezi (x, y) ve genişlik/yükseklik, hepsi 0-1 arasında.
    """
    x_center = (xmin + xmax) / 2 / img_w
    y_center = (ymin + ymax) / 2 / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height


def parse_xml(xml_path):
    """Bir XML dosyasından görüntü boyutunu ve kutuları okur."""
    root = ET.parse(xml_path).getroot()
    img_w = float(root.find("size/width").text)
    img_h = float(root.find("size/height").text)
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        box = obj.find("bndbox")
        xmin, ymin, xmax, ymax = (float(box.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        # Görüntü dışına taşan kutuyu görüntü sınırına çek
        xmin, xmax = max(0.0, xmin), min(img_w, xmax)
        ymin, ymax = max(0.0, ymin), min(img_h, ymax)
        boxes.append((name, xmin, ymin, xmax, ymax))
    return img_w, img_h, boxes


def main():
    parser = argparse.ArgumentParser(description="FOD-A: Pascal VOC -> YOLO dönüşümü")
    parser.add_argument("--voc-dir", required=True, help="VOC2007 klasörü (içinde Annotations ve JPEGImages var)")
    parser.add_argument("--out-dir", default="data/yolo", help="Çıktı klasörü")
    parser.add_argument("--classes", default="configs/classes_31.txt", help="31 sınıf adının sabit listesi")
    args = parser.parse_args()

    voc_dir = Path(args.voc_dir)
    out_dir = Path(args.out_dir)
    class_names = read_class_names(args.classes)
    class_ids = {name: i for i, name in enumerate(class_names)}
    weather_light = read_weather_light(
        voc_dir / "ImageSets" / "Main" / "CategorizationData" / "FOD_categorization_annotations.csv"
    )

    labels_fod_dir = out_dir / "all" / "labels_fod"
    labels_31_dir = out_dir / "all" / "labels_31"
    labels_fod_dir.mkdir(parents=True, exist_ok=True)
    labels_31_dir.mkdir(parents=True, exist_ok=True)

    xml_paths = sorted((voc_dir / "Annotations").glob("*.xml"))
    if not xml_paths:
        raise SystemExit(f"XML bulunamadı: {voc_dir / 'Annotations'}")

    class_counts = Counter()
    n_boxes = 0
    n_skipped = 0  # alanı sıfır olan (bozuk) kutular
    n_no_image = 0
    n_empty = 0  # içinde hiç kutu olmayan görüntüler

    with open(out_dir / "boxes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_id", "class_name", "class_id_31",
            "xmin", "ymin", "xmax", "ymax", "img_w", "img_h",
            "x_center", "y_center", "width", "height",
            "weather", "light",
        ])

        for xml_path in xml_paths:
            image_id = xml_path.stem
            if not (voc_dir / "JPEGImages" / f"{image_id}.jpg").exists():
                n_no_image += 1
                continue

            img_w, img_h, boxes = parse_xml(xml_path)
            weather, light = weather_light.get(image_id, ("", ""))
            lines_fod = []
            lines_31 = []

            for name, xmin, ymin, xmax, ymax in boxes:
                if name not in class_ids:
                    raise SystemExit(f"Bilinmeyen sınıf '{name}' ({xml_path.name}). {args.classes} dosyasını kontrol et.")
                if xmax <= xmin or ymax <= ymin:
                    n_skipped += 1
                    continue
                x, y, w, h = voc_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
                coords = f"{x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                lines_fod.append(f"0 {coords}")
                lines_31.append(f"{class_ids[name]} {coords}")
                writer.writerow([
                    image_id, name, class_ids[name],
                    xmin, ymin, xmax, ymax, int(img_w), int(img_h),
                    f"{x:.6f}", f"{y:.6f}", f"{w:.6f}", f"{h:.6f}",
                    weather, light,
                ])
                class_counts[name] += 1
                n_boxes += 1

            if not lines_fod:
                n_empty += 1
            # Kutu yoksa boş dosya yazılır: YOLO bunu "bu görüntüde cisim yok" diye okur
            (labels_fod_dir / f"{image_id}.txt").write_text("\n".join(lines_fod) + ("\n" if lines_fod else ""))
            (labels_31_dir / f"{image_id}.txt").write_text("\n".join(lines_31) + ("\n" if lines_31 else ""))

    print(f"Görüntü sayısı        : {len(xml_paths) - n_no_image}")
    print(f"Kutu sayısı           : {n_boxes}")
    print(f"Sınıf sayısı          : {len(class_counts)} (listede {len(class_names)})")
    print(f"Kutusuz görüntü       : {n_empty}")
    print(f"Atlanan bozuk kutu    : {n_skipped}")
    print(f"Görüntüsü olmayan XML : {n_no_image}")
    print("Sınıf başına kutu sayısı:")
    for name, count in class_counts.most_common():
        print(f"  {name:<18}{count}")
    print(f"Çıktılar: {out_dir}")


if __name__ == "__main__":
    main()
