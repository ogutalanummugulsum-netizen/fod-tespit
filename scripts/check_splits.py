"""Train / val / test bölümleri arasında sızıntı var mı kontrol eder.

Kullanım (önce make_splits.py çalışmış olmalı):
    python scripts/check_splits.py --data-dir data/yolo

İki tür kontrol yapar:
    1) Kesin kontroller: aynı görüntü iki bölümde mi, bir video parçası iki bölüme
       dağılmış mı, her görüntünün etiketi var mı. Bunlardan biri bozuksa script hata ile biter.
    2) Benzerlik kontrolü: her görüntünün 64 bitlik bir "parmak izi" (perceptual hash)
       çıkarılır. İki parmak izi arasındaki farklı bit sayısı eşikten küçük ya da eşitse
       o iki görüntü "benzer" sayılır. Train-val, train-test ve val-test arasında
       benzer çiftler sayılır ve raporlanır.

Benzerlik kontrolü kesin değildir: düz beton zeminli farklı sahneler de birbirine
benzeyebilir. O yüzden rapor, eşiği anlamlandırmak için aynı videodaki komşu karelerin
birbirine ne kadar benzediğini de gösterir.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

SPLITS = ["train", "val", "test"]
# 0-255 arası her sayının içindeki 1 bitlerinin sayısı
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def read_image_splits(path):
    """image_splits.csv -> [(image_id, segment_id, split), ...] (görüntü sırasıyla)"""
    with open(path, newline="", encoding="utf-8") as f:
        return [(row["image_id"], row["segment_id"], row["split"]) for row in csv.DictReader(f)]


def exact_checks(data_dir, rows):
    """Kesin kontroller. Bulduğu sorunların listesini döner (boş liste = sorun yok)."""
    problems = []

    on_disk = {split: {p.stem for p in (data_dir / "images" / split).glob("*.jpg")} for split in SPLITS}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        common = on_disk[a] & on_disk[b]
        if common:
            problems.append(f"{len(common)} görüntü hem {a} hem {b} klasöründe (örnek: {sorted(common)[:3]})")

    expected = {split: set() for split in SPLITS}
    segment_splits = defaultdict(set)
    for image_id, segment_id, split in rows:
        expected[split].add(image_id)
        segment_splits[segment_id].add(split)

    for split in SPLITS:
        if on_disk[split] != expected[split]:
            problems.append(
                f"{split} klasörü image_splits.csv ile uyuşmuyor "
                f"(klasörde {len(on_disk[split])}, listede {len(expected[split])})"
            )
        labels = {p.stem for p in (data_dir / "labels" / split).glob("*.txt")}
        if labels != on_disk[split]:
            problems.append(f"{split}: {len(on_disk[split] ^ labels)} görüntü ile etiket eşleşmiyor")

    divided = [seg for seg, splits in segment_splits.items() if len(splits) > 1]
    if divided:
        problems.append(f"{len(divided)} video parçası birden fazla bölüme dağılmış (örnek: {divided[:3]})")

    return problems, {split: len(on_disk[split]) for split in SPLITS}


def compute_hashes(data_dir, rows):
    """Her görüntünün parmak izini hesaplar: (görüntü sayısı, 8) boyutlu uint8 dizi."""
    hashes = np.zeros((len(rows), 8), dtype=np.uint8)
    for k, (image_id, _, split) in enumerate(rows):
        with Image.open(data_dir / "images" / split / f"{image_id}.jpg") as img:
            hashes[k] = np.packbits(imagehash.phash(img).hash.flatten())
        if (k + 1) % 5000 == 0:
            print(f"  parmak izi: {k + 1}/{len(rows)}")
    return hashes


def distances(a, b):
    """a ve b'deki her parmak izi çifti arasındaki farklı bit sayısı: (len(a), len(b))"""
    return POPCOUNT[a[:, None, :] ^ b[None, :, :]].sum(axis=2)


def compare(hashes_a, ids_a, hashes_b, ids_b, threshold, chunk=256):
    """a'daki her görüntü için b'deki en yakın görüntüyü bulur ve benzer çiftleri sayar."""
    n_pairs = 0
    nearest_dist = np.zeros(len(ids_a), dtype=np.int64)
    nearest_id = []
    for start in range(0, len(ids_a), chunk):
        d = distances(hashes_a[start:start + chunk], hashes_b)
        n_pairs += int((d <= threshold).sum())
        best = d.argmin(axis=1)
        nearest_dist[start:start + chunk] = d[np.arange(len(best)), best]
        nearest_id += [ids_b[j] for j in best]
    return n_pairs, nearest_dist, nearest_id


def main():
    parser = argparse.ArgumentParser(description="Bölümler arası sızıntı kontrolü")
    parser.add_argument("--data-dir", default="data/yolo")
    parser.add_argument("--threshold", type=int, default=6, help="Benzerlik eşiği: 64 bitten en fazla kaç bit farklı olabilir")
    parser.add_argument("--report", default="results/split_check.txt")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    rows = read_image_splits(data_dir / "image_splits.csv")
    lines = []

    # 1) Kesin kontroller
    problems, n_on_disk = exact_checks(data_dir, rows)
    lines.append("1) Kesin kontroller")
    lines.append("   Klasördeki görüntü sayıları: " + ", ".join(f"{s}={n}" for s, n in n_on_disk.items()))
    if problems:
        # Klasörler bozukken benzerlik kontrolü anlamsız, önce bunlar düzelmeli
        lines += [f"   HATA: {p}" for p in problems]
        print("\n".join(lines))
        sys.exit(1)
    lines.append("   Sorun yok: hiçbir görüntü iki bölümde değil, hiçbir video parçası bölünmemiş.")

    # 2) Benzerlik kontrolü
    print("Parmak izleri hesaplanıyor...")
    hashes = compute_hashes(data_dir, rows)
    ids = [r[0] for r in rows]
    segs = np.array([r[1] for r in rows])
    splits = np.array([r[2] for r in rows])

    lines += ["", f"2) Benzerlik kontrolü (eşik: 64 bitten en fazla {args.threshold} bit farklı)"]

    # Eşiği anlamlandırmak için: aynı video parçasındaki komşu kareler ne kadar benzer?
    same_segment = segs[1:] == segs[:-1]
    neighbour_dist = POPCOUNT[hashes[1:] ^ hashes[:-1]].sum(axis=1)[same_segment]
    lines.append(
        "   Karşılaştırma için, aynı videodaki komşu kareler: "
        f"ortanca fark {int(np.median(neighbour_dist))} bit, "
        f"%90'ı en fazla {int(np.percentile(neighbour_dist, 90))} bit, "
        f"%{100 * (neighbour_dist <= args.threshold).mean():.1f}'i eşiğin altında"
    )

    for a, b in (("val", "train"), ("test", "train"), ("test", "val")):
        mask_a, mask_b = splits == a, splits == b
        ids_a = [i for i, m in zip(ids, mask_a) if m]
        ids_b = [i for i, m in zip(ids, mask_b) if m]
        n_pairs, nearest_dist, nearest_id = compare(hashes[mask_a], ids_a, hashes[mask_b], ids_b, args.threshold)
        n_images = int((nearest_dist <= args.threshold).sum())

        lines += [
            "",
            f"   {a} <-> {b}",
            f"     Benzer çift sayısı: {n_pairs}",
            f"     {b} içinde benzeri olan {a} görüntüsü: {n_images} / {len(ids_a)} (%{100 * n_images / len(ids_a):.1f})",
            f"     {a} görüntülerinin {b} içindeki en yakın görüntüye farkı: "
            f"en az {int(nearest_dist.min())}, ortanca {int(np.median(nearest_dist))} bit",
        ]
        closest = np.argsort(nearest_dist, kind="stable")[:5]
        lines.append("     En benzer 5 çift (gözle bakmak için): " + ", ".join(
            f"{ids_a[k]}~{nearest_id[k]} ({int(nearest_dist[k])} bit)" for k in closest
        ))

    report = "\n".join(lines)
    print(report)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nRapor: {report_path}")


if __name__ == "__main__":
    main()
