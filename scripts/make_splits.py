"""Görüntüleri video parçalarına ayırır ve parçaları bütün olarak train / val / test'e dağıtır.

Kullanım (önce convert_voc_to_yolo.py çalışmış olmalı):
    python scripts/make_splits.py \
        --voc-dir data/raw/FODPascalVOCFormat-V.2.1/VOC2007 \
        --data-dir data/yolo

Video parçası kuralı: görüntüler numara sırasıyla gezilir. Sınıf değişirse ya da
hava/ışık değişirse yeni parça başlar. Kutunun yer değiştirmesi (sıçrama) parça
sınırı sayılmaz. FOD-A VOC 2.1 verisinde bu kural 116 parça üretir.

Bölme kuralları:
    - Bir parça asla ikiye bölünmez.
    - Val ve test, toplam karelerin yaklaşık %10'u olur (%9 - %11 arası kabul edilir).
    - Tek bir parça, val ya da test'in yarısından fazlasını oluşturamaz.
    - Test'te (sonra val'de) olabildiğince çok farklı sınıf bulunur.
    - Seed sabittir: aynı komut her zaman aynı bölmeyi üretir.

Ürettiği dosyalar:
    configs/splits.csv              -> hangi parça hangi bölümde (GitHub'da saklanır)
    data/yolo/image_splits.csv      -> hangi görüntü hangi parçada ve bölümde
    data/yolo/images/{train,val,test}/ ve data/yolo/labels/{train,val,test}/
    results/split_report.txt        -> bölme raporu (sınıf sınıf kare sayıları)
"""

import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path

SPLITS = ["train", "val", "test"]
SPLITS_HEADER = ["segment_id", "first_image", "last_image", "n_images", "classes", "weather", "light", "split"]


def read_images(boxes_csv):
    """boxes.csv dosyasından her görüntünün sınıflarını, hava ve ışık bilgisini okur."""
    images = {}
    with open(boxes_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            info = images.setdefault(
                row["image_id"], {"classes": set(), "weather": row["weather"], "light": row["light"]}
            )
            info["classes"].add(row["class_name"])
    # Görüntü numaraları video sırasını izler, o yüzden numaraya göre sıralıyoruz
    return dict(sorted(images.items()))


def find_segments(images):
    """Ardışık görüntüleri video parçalarına ayırır."""
    segments = []
    prev = None
    for image_id, info in images.items():
        new_segment = (
            prev is None
            or not (info["classes"] & prev["classes"])  # ortak sınıf yok -> başka cisim
            or (info["weather"], info["light"]) != (prev["weather"], prev["light"])
        )
        if new_segment:
            segments.append({
                "segment_id": f"seg_{len(segments):03d}",
                "image_ids": [],
                "classes": set(),
                "weather": info["weather"],
                "light": info["light"],
            })
        segments[-1]["image_ids"].append(image_id)
        segments[-1]["classes"] |= info["classes"]
        prev = info
    for seg in segments:
        seg["n_images"] = len(seg["image_ids"])
    return segments


def pick_segments(available, segments, rng, target, tolerance, max_share):
    """Bir bölüm (val ya da test) için rastgele bir parça seçimi dener.

    Önce bölüme yeni sınıf getiren parçaları, sonra kalanları ekler.
    Kurallar sağlanmazsa None döner.
    """
    order = sorted(available, key=lambda i: rng.random())  # rastgele sıra
    chosen = []
    total = 0
    classes = set()
    for must_add_class in (True, False):
        for i in order:
            if total >= target:
                break
            seg = segments[i]
            if i in chosen:
                continue
            if must_add_class and seg["classes"] <= classes:
                continue
            if total + seg["n_images"] > target * (1 + tolerance):
                continue
            chosen.append(i)
            total += seg["n_images"]
            classes |= seg["classes"]

    if total < target * (1 - tolerance):
        return None
    if max(segments[i]["n_images"] for i in chosen) > max_share * total:
        return None
    return chosen


def split_stats(indices, segments):
    total = sum(segments[i]["n_images"] for i in indices)
    classes = set()
    for i in indices:
        classes |= segments[i]["classes"]
    return total, classes


def assign_splits(segments, seed, val_ratio, test_ratio, tries, tolerance=0.10, max_share=0.5):
    """Birçok rastgele bölme dener, kurallara uyanlar içinden en iyisini seçer."""
    rng = random.Random(seed)
    n_total = sum(seg["n_images"] for seg in segments)
    test_target = n_total * test_ratio
    val_target = n_total * val_ratio
    all_indices = list(range(len(segments)))

    best = None
    for _ in range(tries):
        test = pick_segments(all_indices, segments, rng, test_target, tolerance, max_share)
        if test is None:
            continue
        rest = [i for i in all_indices if i not in test]
        val = pick_segments(rest, segments, rng, val_target, tolerance, max_share)
        if val is None:
            continue
        test_total, test_classes = split_stats(test, segments)
        val_total, val_classes = split_stats(val, segments)
        # Önce test'teki sınıf sayısı, sonra val'deki sınıf sayısı, sonra hedef orana yakınlık
        score = (
            len(test_classes),
            len(val_classes),
            -(abs(test_total - test_target) + abs(val_total - val_target)),
        )
        if best is None or score > best[0]:
            best = (score, test, val)

    if best is None:
        raise SystemExit("Kurallara uyan bir bölme bulunamadı. --tries değerini artırmayı dene.")

    _, test, val = best
    assignment = {}
    for i, seg in enumerate(segments):
        assignment[seg["segment_id"]] = "test" if i in test else "val" if i in val else "train"
    return assignment


def splits_rows(segments, assignment):
    rows = []
    for seg in segments:
        rows.append([
            seg["segment_id"], seg["image_ids"][0], seg["image_ids"][-1], str(seg["n_images"]),
            "+".join(sorted(seg["classes"])), seg["weather"], seg["light"], assignment[seg["segment_id"]],
        ])
    return rows


def save_splits_file(path, rows, overwrite):
    """configs/splits.csv dosyasını yazar. Dosya varsa ve bölme farklıysa durur (test seti donmuş kalsın)."""
    path = Path(path)
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            old_rows = list(csv.reader(f))[1:]
        if old_rows == rows:
            print(f"Bölme, kayıtlı {path} dosyasıyla aynı.")
            return
        if not overwrite:
            raise SystemExit(
                f"DUR: Yeni bölme, kayıtlı {path} dosyasından farklı.\n"
                "Test seti değişmesin diye üzerine yazmadım. Gerçekten değiştirmek istiyorsan --overwrite ekle."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SPLITS_HEADER)
        writer.writerows(rows)
    print(f"Yazıldı: {path}")


def build_folders(segments, assignment, voc_dir, data_dir, label_set):
    """Görüntüleri ve etiketleri train / val / test klasörlerine kopyalar."""
    for split in SPLITS:
        for kind in ("images", "labels"):
            folder = data_dir / kind / split
            if folder.exists():
                shutil.rmtree(folder)  # eski bölmeden dosya kalmasın
            folder.mkdir(parents=True)

    labels_all = data_dir / "all" / f"labels_{label_set}"
    with open(data_dir / "image_splits.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "segment_id", "split"])
        for seg in segments:
            split = assignment[seg["segment_id"]]
            for image_id in seg["image_ids"]:
                shutil.copy(voc_dir / "JPEGImages" / f"{image_id}.jpg", data_dir / "images" / split / f"{image_id}.jpg")
                shutil.copy(labels_all / f"{image_id}.txt", data_dir / "labels" / split / f"{image_id}.txt")
                writer.writerow([image_id, seg["segment_id"], split])


def make_report(segments, assignment, images, seed):
    """Bölme raporunu metin olarak hazırlar."""
    n_total = sum(seg["n_images"] for seg in segments)
    lines = [f"Seed: {seed}", f"Video parçası sayısı: {len(segments)}", f"Toplam kare: {n_total}", ""]

    lines.append(f"{'bölüm':<7}{'kare':>8}{'oran':>8}{'parça':>7}{'sınıf':>7}{'en büyük parçanın payı':>26}")
    for split in SPLITS:
        segs = [seg for seg in segments if assignment[seg["segment_id"]] == split]
        total = sum(seg["n_images"] for seg in segs)
        classes = set()
        for seg in segs:
            classes |= seg["classes"]
        biggest = max(seg["n_images"] for seg in segs)
        lines.append(
            f"{split:<7}{total:>8}{100 * total / n_total:>7.1f}%{len(segs):>7}{len(classes):>7}{100 * biggest / total:>25.1f}%"
        )

    # Sınıf başına, bölüm başına kare sayısı
    counts = {split: Counter() for split in SPLITS}
    for seg in segments:
        split = assignment[seg["segment_id"]]
        for image_id in seg["image_ids"]:
            for name in images[image_id]["classes"]:
                counts[split][name] += 1
    all_classes = sorted(set().union(*[set(c) for c in counts.values()]))

    lines += ["", "Sınıf başına kare sayısı:", f"{'sınıf':<18}{'train':>8}{'val':>8}{'test':>8}  not"]
    for name in all_classes:
        tr, va, te = (counts[s][name] for s in SPLITS)
        note = ""
        if tr == 0:
            note = "EĞİTİMDE YOK"
        elif te == 0 and va == 0:
            note = "yalnızca train'de"
        lines.append(f"{name:<18}{tr:>8}{va:>8}{te:>8}  {note}")

    test_seen = [n for n in all_classes if counts["test"][n] and counts["train"][n]]
    test_unseen = [n for n in all_classes if counts["test"][n] and not counts["train"][n]]
    test_missing = [n for n in all_classes if not counts["test"][n]]
    val_unseen = [n for n in all_classes if counts["val"][n] and not counts["train"][n]]
    lines += [
        "",
        f"Test'te olan ve eğitimde de görülen sınıflar ({len(test_seen)}): {', '.join(test_seen)}",
        f"Test'te olan ama eğitimde hiç olmayan sınıflar ({len(test_unseen)}): {', '.join(test_unseen) or '-'}",
        f"Test'te hiç olmayan sınıflar ({len(test_missing)}): {', '.join(test_missing) or '-'}",
        f"Val'de olan ama eğitimde hiç olmayan sınıflar ({len(val_unseen)}): {', '.join(val_unseen) or '-'}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="FOD-A: video parçası bazında train / val / test bölmesi")
    parser.add_argument("--voc-dir", required=True, help="VOC2007 klasörü (JPEGImages burada)")
    parser.add_argument("--data-dir", default="data/yolo", help="convert_voc_to_yolo.py çıktısının olduğu klasör")
    parser.add_argument("--splits-file", default="configs/splits.csv", help="Parça -> bölüm kaydı")
    parser.add_argument("--report", default="results/split_report.txt", help="Bölme raporu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--tries", type=int, default=2000, help="Denenecek rastgele bölme sayısı")
    parser.add_argument("--label-set", default="fod", choices=["fod", "31"], help="fod = tek sınıf, 31 = 31 sınıf")
    parser.add_argument("--overwrite", action="store_true", help="Kayıtlı bölme farklıysa üzerine yaz")
    args = parser.parse_args()

    voc_dir = Path(args.voc_dir)
    data_dir = Path(args.data_dir)

    images = read_images(data_dir / "boxes.csv")
    segments = find_segments(images)
    assignment = assign_splits(segments, args.seed, args.val_ratio, args.test_ratio, args.tries)

    save_splits_file(args.splits_file, splits_rows(segments, assignment), args.overwrite)
    build_folders(segments, assignment, voc_dir, data_dir, args.label_set)

    report = make_report(segments, assignment, images, args.seed)
    print(report)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nRapor: {report_path}")


if __name__ == "__main__":
    main()
