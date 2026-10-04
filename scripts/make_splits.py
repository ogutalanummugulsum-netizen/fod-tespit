"""Görüntüleri video parçalarına ayırır ve parçaları bütün olarak train / val / test'e dağıtır.

Kullanım (önce convert_voc_to_yolo.py çalışmış olmalı):
    python scripts/make_splits.py \
        --voc-dir data/raw/FODPascalVOCFormat-V.2.1/VOC2007 \
        --data-dir data/yolo

Video parçası kuralı: görüntüler numara sırasıyla gezilir. Sınıf değişirse ya da
hava/ışık değişirse yeni parça başlar. Kutunun yer değiştirmesi (sıçrama) parça
sınırı sayılmaz. FOD-A VOC 2.1 verisinde bu kural 116 parça üretir.

Bölme kuralları (hepsi aynı anda sağlanmalı):
    1. Bir parça asla ikiye bölünmez.
    2. Val ve test, toplam karelerin %9 - %11'i olur.
    3. Tek bir parça, val ya da test'in yarısından fazlasını oluşturamaz.
    4. Test'te Bolt, Nut ve Rock'tan en az birer parça bulunur.
    5. Val'de en az 12 parça bulunur ve tek bir sınıf val karelerinin %25'ini geçemez.
    6. 32 pikselden küçük kutuların oranı val ve test'te train'e yakındır (en fazla 5 puan fark).
    7. Test'te, eğitimde hiç görülmeyen 2 - 4 sınıf bulunur.
Kurallara uyan bölmeler içinden, test'te en çok farklı sınıf olan seçilir
(eşitlikte: val'de daha çok parça, sonra val'de daha çok sınıf, sonra %10'a yakınlık).
Seed sabittir: aynı komut her zaman aynı bölmeyi üretir.

Ürettiği dosyalar:
    configs/splits.csv              -> hangi parça hangi bölümde (GitHub'da saklanır)
    data/yolo/image_splits.csv      -> hangi görüntü hangi parçada ve bölümde
    data/yolo/images/{train,val,test}/ ve data/yolo/labels/{train,val,test}/
    results/split_report.txt        -> bölme raporu (kural kontrolü, sınıf sınıf kare sayıları)
"""

import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path

SPLITS = ["train", "val", "test"]
SPLITS_HEADER = ["segment_id", "first_image", "last_image", "n_images", "classes", "weather", "light", "split"]

# Bölme kuralları
TOLERANCE = 0.10                    # val/test hedef oranından en fazla %10 sapabilir (%9 - %11)
MAX_SEGMENT_SHARE = 0.5             # tek parça, val ya da test'in en fazla yarısı
TEST_REQUIRED_CLASSES = ["Bolt", "Nut", "Rock"]
VAL_MIN_SEGMENTS = 12
VAL_MAX_CLASS_SHARE = 0.25
SMALL_BOX_PX = 32                   # uzun kenarı bundan kısa olan kutu "küçük" sayılır
SMALL_BOX_MAX_DIFF = 5.0            # küçük kutu oranında train ile en fazla fark (yüzde puanı)
TEST_UNSEEN_CLASSES = (2, 4)        # test'te olup eğitimde hiç olmayan sınıf sayısı (en az, en çok)


def read_images(boxes_csv):
    """boxes.csv dosyasından her görüntünün sınıflarını, hava/ışık bilgisini ve kutu sayılarını okur."""
    images = {}
    with open(boxes_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            info = images.setdefault(row["image_id"], {
                "classes": set(), "weather": row["weather"], "light": row["light"], "n_boxes": 0, "n_small": 0,
            })
            info["classes"].add(row["class_name"])
            long_side = max(float(row["xmax"]) - float(row["xmin"]), float(row["ymax"]) - float(row["ymin"]))
            info["n_boxes"] += 1
            if long_side < SMALL_BOX_PX:
                info["n_small"] += 1
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
                "class_frames": Counter(),  # sınıf -> o sınıfı içeren kare sayısı
                "n_boxes": 0,
                "n_small": 0,
                "weather": info["weather"],
                "light": info["light"],
            })
        seg = segments[-1]
        seg["image_ids"].append(image_id)
        seg["classes"] |= info["classes"]
        seg["class_frames"].update(info["classes"])
        seg["n_boxes"] += info["n_boxes"]
        seg["n_small"] += info["n_small"]
        prev = info
    for seg in segments:
        seg["n_images"] = len(seg["image_ids"])
    return segments


def split_stats(indices, segments):
    """Bir bölümün özet sayıları."""
    class_frames = Counter()
    for i in indices:
        class_frames.update(segments[i]["class_frames"])
    n_boxes = sum(segments[i]["n_boxes"] for i in indices)
    return {
        "n_images": sum(segments[i]["n_images"] for i in indices),
        "n_segments": len(indices),
        "biggest": max(segments[i]["n_images"] for i in indices),
        "class_frames": class_frames,
        "small_share": 100 * sum(segments[i]["n_small"] for i in indices) / n_boxes,
    }


def fill(chosen, order, segments, target, class_cap=None):
    """Bir bölümü hedef kare sayısına kadar doldurur.

    Önce bölüme yeni sınıf getiren parçaları, sonra kalanları ekler.
    class_cap verilirse hiçbir sınıfın kare sayısı bu sınırı geçemez.
    """
    total = sum(segments[i]["n_images"] for i in chosen)
    class_frames = Counter()
    for i in chosen:
        class_frames.update(segments[i]["class_frames"])

    for must_add_class in (True, False):
        for i in order:
            if total >= target:
                break
            seg = segments[i]
            if i in chosen:
                continue
            if must_add_class and all(name in class_frames for name in seg["classes"]):
                continue
            if total + seg["n_images"] > target * (1 + TOLERANCE):
                continue
            if class_cap is not None and any(
                class_frames[name] + n > class_cap for name, n in seg["class_frames"].items()
            ):
                continue
            chosen.append(i)
            total += seg["n_images"]
            class_frames.update(seg["class_frames"])
    return chosen


def draw_candidate(segments, rng, test_target, val_target):
    """Rastgele bir bölme adayı üretir: (test parçaları, val parçaları)."""
    all_indices = list(range(len(segments)))

    # Test: önce zorunlu sınıflardan (Bolt, Nut, Rock) birer rastgele parça
    test = []
    for name in TEST_REQUIRED_CLASSES:
        options = [i for i in all_indices if name in segments[i]["classes"] and i not in test]
        test.append(options[int(rng.random() * len(options))])
    order = sorted(all_indices, key=lambda i: rng.random())  # rastgele sıra
    fill(test, order, segments, test_target)

    # Val: kalan parçalardan. Sınıf sınırı, val en küçük boyutunda kalsa bile %25 kuralını sağlar.
    rest = [i for i in all_indices if i not in test]
    order = sorted(rest, key=lambda i: rng.random())
    class_cap = VAL_MAX_CLASS_SHARE * val_target * (1 - TOLERANCE)
    val = fill([], order, segments, val_target, class_cap)
    return test, val


def check_rules(test, val, segments, test_target, val_target):
    """Bölme kurallarını kontrol eder. [(kural, tutuyor mu, açıklama), ...] ve istatistikleri döner."""
    train = [i for i in range(len(segments)) if i not in test and i not in val]
    stats = {"train": split_stats(train, segments), "test": split_stats(test, segments)}
    if not val:
        return [("Val boş", False, "val'e hiç parça seçilemedi")], stats
    stats["val"] = split_stats(val, segments)

    rules = []
    for name, target in (("val", val_target), ("test", test_target)):
        s = stats[name]
        rules.append((
            f"{name} boyutu hedefin %{100 * TOLERANCE:.0f} yakınında",
            target * (1 - TOLERANCE) <= s["n_images"] <= target * (1 + TOLERANCE),
            f"{s['n_images']} kare (hedef {target:.0f})",
        ))
        rules.append((
            f"{name}: tek parça yarıyı geçmiyor",
            s["biggest"] <= MAX_SEGMENT_SHARE * s["n_images"],
            f"en büyük parça %{100 * s['biggest'] / s['n_images']:.1f}",
        ))

    for name in TEST_REQUIRED_CLASSES:
        n = stats["test"]["class_frames"][name]
        rules.append((f"test'te {name} var", n > 0, f"{n} kare"))

    v = stats["val"]
    rules.append((f"val'de en az {VAL_MIN_SEGMENTS} parça", v["n_segments"] >= VAL_MIN_SEGMENTS, f"{v['n_segments']} parça"))
    top_class, top_frames = v["class_frames"].most_common(1)[0]
    rules.append((
        f"val'de tek sınıf en fazla %{100 * VAL_MAX_CLASS_SHARE:.0f}",
        top_frames <= VAL_MAX_CLASS_SHARE * v["n_images"],
        f"en büyük sınıf {top_class} %{100 * top_frames / v['n_images']:.1f}",
    ))

    train_small = stats["train"]["small_share"]
    for name in ("val", "test"):
        diff = stats[name]["small_share"] - train_small
        rules.append((
            f"{name}: küçük kutu oranı train'e yakın (en fazla {SMALL_BOX_MAX_DIFF:.0f} puan)",
            abs(diff) <= SMALL_BOX_MAX_DIFF,
            f"{name} %{stats[name]['small_share']:.1f}, train %{train_small:.1f}, fark {diff:+.1f} puan",
        ))

    unseen = sorted(c for c in stats["test"]["class_frames"] if c not in stats["train"]["class_frames"])
    stats["test_unseen"] = unseen
    low, high = TEST_UNSEEN_CLASSES
    rules.append((
        f"test'te eğitimde olmayan {low}-{high} sınıf",
        low <= len(unseen) <= high,
        f"{len(unseen)} sınıf: {', '.join(unseen) or '-'}",
    ))
    return rules, stats


def assign_splits(segments, seed, val_ratio, test_ratio, tries):
    """Birçok rastgele bölme dener, bütün kurallara uyanlar içinden en iyisini seçer."""
    rng = random.Random(seed)
    n_total = sum(seg["n_images"] for seg in segments)
    test_target = n_total * test_ratio
    val_target = n_total * val_ratio

    best = None
    n_ok = 0
    failures = Counter()  # hangi kural kaç adayda tutmadı
    for _ in range(tries):
        test, val = draw_candidate(segments, rng, test_target, val_target)
        rules, stats = check_rules(test, val, segments, test_target, val_target)
        failed = [rule for rule, ok, _ in rules if not ok]
        if failed:
            failures.update(failed)
            continue
        n_ok += 1
        score = (
            len(stats["test"]["class_frames"]),
            stats["val"]["n_segments"],
            len(stats["val"]["class_frames"]),
            -(abs(stats["test"]["n_images"] - test_target) + abs(stats["val"]["n_images"] - val_target)),
        )
        if best is None or score > best[0]:
            best = (score, test, val, rules)

    if best is None:
        # Zorla uydurmuyoruz: hangi kuralın ne sıklıkta tutmadığını göster ve dur
        lines = [f"{tries} denemede bütün kurallara uyan bir bölme bulunamadı.", "Tutmayan kurallar (kaç adayda):"]
        lines += [f"  {count:>6}  {rule}" for rule, count in failures.most_common()]
        raise SystemExit("\n".join(lines))

    _, test, val, rules = best
    assignment = {}
    for i, seg in enumerate(segments):
        assignment[seg["segment_id"]] = "test" if i in test else "val" if i in val else "train"
    return assignment, rules, n_ok


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


def make_report(segments, assignment, rules, seed, tries, n_ok):
    """Bölme raporunu metin olarak hazırlar."""
    n_total = sum(seg["n_images"] for seg in segments)
    lines = [
        f"Seed: {seed}",
        f"Video parçası sayısı: {len(segments)}",
        f"Toplam kare: {n_total}",
        f"Denenen bölme: {tries}, bütün kurallara uyan: {n_ok}",
        "",
    ]

    by_split = {split: [i for i, seg in enumerate(segments) if assignment[seg["segment_id"]] == split] for split in SPLITS}
    stats = {split: split_stats(by_split[split], segments) for split in SPLITS}

    lines.append(f"{'bölüm':<7}{'kare':>8}{'oran':>8}{'parça':>7}{'sınıf':>7}{'en büyük parça':>16}{'küçük kutu':>12}")
    for split in SPLITS:
        s = stats[split]
        lines.append(
            f"{split:<7}{s['n_images']:>8}{100 * s['n_images'] / n_total:>7.1f}%{s['n_segments']:>7}"
            f"{len(s['class_frames']):>7}{100 * s['biggest'] / s['n_images']:>15.1f}%{s['small_share']:>11.1f}%"
        )
    lines.append(f"(küçük kutu: uzun kenarı {SMALL_BOX_PX} pikselden kısa olan kutuların oranı)")

    lines += ["", "Kural kontrolü:"]
    for rule, ok, detail in rules:
        lines.append(f"  {'OK  ' if ok else 'HATA'}  {rule}: {detail}")

    counts = {split: stats[split]["class_frames"] for split in SPLITS}
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
    parser.add_argument("--tries", type=int, default=30000, help="Denenecek rastgele bölme sayısı")
    parser.add_argument("--label-set", default="fod", choices=["fod", "31"], help="fod = tek sınıf, 31 = 31 sınıf")
    parser.add_argument("--overwrite", action="store_true", help="Kayıtlı bölme farklıysa üzerine yaz")
    args = parser.parse_args()

    voc_dir = Path(args.voc_dir)
    data_dir = Path(args.data_dir)

    images = read_images(data_dir / "boxes.csv")
    segments = find_segments(images)
    assignment, rules, n_ok = assign_splits(segments, args.seed, args.val_ratio, args.test_ratio, args.tries)

    save_splits_file(args.splits_file, splits_rows(segments, assignment), args.overwrite)
    build_folders(segments, assignment, voc_dir, data_dir, args.label_set)

    report = make_report(segments, assignment, rules, args.seed, args.tries, n_ok)
    print(report)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nRapor: {report_path}")


if __name__ == "__main__":
    main()
