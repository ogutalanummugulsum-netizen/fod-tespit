"""Hazır (sızıntılı) trainval / test ayrımıyla AYRI bir veri klasörü hazırlar (README, Deney sırası 6. adım).

Kullanım (önce convert_voc_to_yolo.py ve make_splits.py çalışmış olmalı):
    python scripts/make_leaky_split.py --voc-dir data/raw/FODPascalVOCFormat-V.2.1/VOC2007

Ne yapar:
    - Veri setinin hazır ayrımını okur: ImageSets/Main/trainval.txt ve test.txt.
      Listelerin bilinen sorunları: tekrar eden numaralar tek kareye indirilir; iki listede birden olan
      görüntüler yalnızca test'te bırakılır; hiçbir listede olmayan görüntüler kullanılmaz (sayıları raporda).
    - trainval'den rastgele %10 kareyi (seed 42, kare bazında) val olarak ayırır. Bu da bilerek
      sızıntılıdır: amaç, yaygın "rastgele kare bölme" yönteminin verdiği puanı görmek.
    - data/yolo_sizintili/ altına images/{train,val,test}, labels/{train,val,test}, boxes.csv ve
      image_splits.csv yazar. Bizim bölmemize (data/yolo, configs/splits.csv) DOKUNMAZ.
    - image_splits.csv'de her karenin video parçası ve bizim bölmedeki bölümü (bizim_bolum) de yazar.
    - Sızıntı raporu: results/leaky_split_report.txt
      (hazır test'teki karelerin kaçının video parçası eğitimde de var, kaçının bir komşu karesi
      eğitimde var; bizim test setimizle ortak kare sayısı).
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ["train", "val", "test"]


def read_ids(path):
    """Listeyi okur. Hazır listelerde bazı numaralar birden çok kez yazılmış; bunlar tek kareye indirilir.

    Döner: (tekil numaralar, satır sayısı, birden çok kez yazılmış numaralar)
    """
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = pd.Series(lines).value_counts()
    return set(lines), len(lines), sorted(counts[counts > 1].index)


def segment_table(splits_csv):
    """configs/splits.csv'deki parça aralıklarından her kare için video parçası ve bizim bölüm."""
    splits = pd.read_csv(splits_csv, dtype={"first_image": str, "last_image": str})
    rows = []
    for seg in splits.itertuples():
        for number in range(int(seg.first_image), int(seg.last_image) + 1):
            rows.append({"image_id": f"{number:06d}", "segment_id": seg.segment_id, "bizim_bolum": seg.split})
    return pd.DataFrame(rows)


def leak_lines(name, part, train):
    """Bir bölümdeki karelerin eğitim kareleriyle ne kadar yakın olduğunu sayar."""
    train_segments = set(train["segment_id"])
    train_numbers = set(train["image_id"].astype(int))
    numbers = part["image_id"].astype(int)
    seg_in_train = part["segment_id"].isin(train_segments)
    # komşu kare: aynı video parçasında numarası bir önce ya da bir sonra olan kare
    seg_of = dict(zip(pd.concat([part, train])["image_id"].astype(int), pd.concat([part, train])["segment_id"]))
    neighbor = [
        any(n2 in train_numbers and seg_of.get(n2) == seg for n2 in (n - 1, n + 1))
        for n, seg in zip(numbers, part["segment_id"])
    ]
    n = len(part)
    return [
        f"{name}: {n} kare, {part['segment_id'].nunique()} video parçası",
        f"  video parçası eğitimde de olan kare : {int(seg_in_train.sum())} (%{100 * seg_in_train.mean():.1f})",
        f"  bir komşu karesi eğitimde olan kare : {int(sum(neighbor))} (%{100 * np.mean(neighbor):.1f})",
        f"  video parçası eğitimde HİÇ olmayan  : {int((~seg_in_train).sum())} kare",
    ]


def main():
    parser = argparse.ArgumentParser(description="FOD: hazır (sızıntılı) ayrımla ayrı veri klasörü")
    parser.add_argument("--voc-dir", required=True)
    parser.add_argument("--src-dir", default="data/yolo", help="convert_voc_to_yolo.py çıktısı (all/labels_fod, boxes.csv)")
    parser.add_argument("--out-dir", default="data/yolo_sizintili")
    parser.add_argument("--splits-csv", default="configs/splits.csv")
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default="results/leaky_split_report.txt")
    args = parser.parse_args()

    voc_dir, src_dir, out_dir = Path(args.voc_dir), Path(args.src_dir), Path(args.out_dir)
    if out_dir.resolve() == src_dir.resolve():
        raise SystemExit("DUR: çıktı klasörü bizim veri klasörümüzle aynı olamaz.")

    # Hazır listelerin bilinen sorunları (README, "Hazır listelerin sorunları"):
    #   - bazı numaralar bir listede birden çok kez yazılmış -> tek kareye indirilir
    #   - bazı görüntüler iki listede birden var -> trainval'den çıkarılır, yalnızca test'te kalır
    #   - bazı görüntüler hiçbir listede yok -> kullanılmaz
    trainval, n_trainval_lines, dup_trainval = read_ids(voc_dir / "ImageSets" / "Main" / "trainval.txt")
    test, n_test_lines, dup_test = read_ids(voc_dir / "ImageSets" / "Main" / "test.txt")
    in_both = sorted(trainval & test)
    trainval = trainval - test

    boxes = pd.read_csv(src_dir / "boxes.csv", dtype={"image_id": str})
    all_ids = set(boxes["image_id"])
    unknown = (trainval | test) - all_ids
    if unknown:
        raise SystemExit(f"DUR: listelerde veri setinde olmayan {len(unknown)} numara var (örnek: {sorted(unknown)[:3]}).")
    unused = sorted(all_ids - trainval - test)

    rng = np.random.default_rng(args.seed)
    trainval_sorted = sorted(trainval)
    n_val = round(args.val_frac * len(trainval_sorted))
    val = set(rng.choice(trainval_sorted, size=n_val, replace=False))

    table = segment_table(args.splits_csv)
    if set(table["image_id"]) != all_ids:
        raise SystemExit("DUR: configs/splits.csv bütün kareleri kapsamıyor.")
    split_of = {i: "test" for i in test}
    split_of.update({i: ("val" if i in val else "train") for i in trainval})
    table["split"] = table["image_id"].map(split_of)
    unused_table = table[table["split"].isna()]
    table = table.dropna(subset=["split"]).sort_values("image_id").reset_index(drop=True)
    if set(table["split"]) != set(SPLITS) or len(unused_table) != len(unused):
        raise SystemExit("DUR: bölüm ataması beklenmedik bir sonuç verdi.")

    # Klasörler
    for split in SPLITS:
        for kind in ("images", "labels"):
            folder = out_dir / kind / split
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True)
    labels_all = src_dir / "all" / "labels_fod"
    for row in table.itertuples():
        shutil.copy(voc_dir / "JPEGImages" / f"{row.image_id}.jpg", out_dir / "images" / row.split / f"{row.image_id}.jpg")
        shutil.copy(labels_all / f"{row.image_id}.txt", out_dir / "labels" / row.split / f"{row.image_id}.txt")
    table[["image_id", "segment_id", "split", "bizim_bolum"]].to_csv(out_dir / "image_splits.csv", index=False)
    shutil.copy(src_dir / "boxes.csv", out_dir / "boxes.csv")

    # Ortak kareler: hazır test'te ve bizim test setimizde olan kareler
    common = table[(table["split"] == "test") & (table["bizim_bolum"] == "test")]
    (out_dir / "ortak_kareler.txt").write_text("\n".join(common["image_id"]) + "\n", encoding="utf-8")

    # Rapor
    train = table[table["split"] == "train"]
    light = boxes.drop_duplicates("image_id").set_index("image_id")
    common_light = (light.loc[common["image_id"], "weather"] + "-" + light.loc[common["image_id"], "light"]).value_counts()
    lines = [
        "Hazır (sızıntılı) ayrım, README Deney sırası 6. adım",
        f"Seed: {args.seed}, val oranı (trainval içinden, kare bazında rastgele): {args.val_frac}",
        "",
        "Hazır listelerin sorunları ve nasıl ele alındığı:",
        f"  trainval.txt: {n_trainval_lines} satır, {n_trainval_lines - len(trainval | set(in_both))} tekrar satırı;"
        f" birden çok kez yazılan numara: {len(dup_trainval)} ({', '.join(dup_trainval) or '-'}) -> tek kareye indirildi",
        f"  test.txt    : {n_test_lines} satır, {n_test_lines - len(test)} tekrar satırı;"
        f" birden çok kez yazılan numara: {len(dup_test)} ({', '.join(dup_test) or '-'}) -> tek kareye indirildi",
        f"  iki listede birden olan görüntü: {len(in_both)} ({', '.join(in_both) or '-'}) -> trainval'den çıkarıldı, yalnızca test'te",
        f"  hiçbir listede olmayan görüntü : {len(unused)} -> kullanılmadı"
        f" (bizim bölmede: {', '.join(f'{k} {v}' for k, v in unused_table['bizim_bolum'].value_counts().items()) or '-'})",
        "",
        "Kare sayıları:",
    ]
    lines += [f"  {split:<6}{int((table['split'] == split).sum()):>7}" for split in SPLITS]
    lines += ["", "Hazır ayrım x bizim bölme (kare):"]
    lines += ["  " + line for line in pd.crosstab(table["split"], table["bizim_bolum"]).reindex(index=SPLITS, columns=SPLITS).to_string().splitlines()]
    lines += ["", "Sızıntı (eğitim = hazır train):"]
    lines += leak_lines("hazır test", table[table["split"] == "test"], train)
    lines += leak_lines("hazır val", table[table["split"] == "val"], train)
    lines += ["", "Ortak kareler (hazır test VE bizim test):"]
    lines += leak_lines("ortak", common, train)
    lines += ["  ışık koşulu: " + ", ".join(f"{k} {v}" for k, v in common_light.items())]
    lines += ["", "exp002_nano bu ortak karelerin video parçalarını hiç görmedi (parçalar bizim test'te).",
              "exp004_nano_sizintili ise aynı parçaların başka karelerini eğitimde görüyor."]
    report = "\n".join(lines) + "\n"
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(report, encoding="utf-8")
    print(report)
    print(f"Veri klasörü: {out_dir}   Rapor: {args.report}")


if __name__ == "__main__":
    main()
