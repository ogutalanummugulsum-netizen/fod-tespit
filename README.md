# FOD Tespiti (YOLO)

Havalimanı pist görüntülerinde vida, somun, taş gibi küçük yabancı cisimleri (FOD) tespit eden model.
Ana hedef: radar ve özel kamera kullanan pahalı ticari FOD sistemlerine karşı düşük maliyetli bir alternatif.
Veri seti: [FOD-A](https://github.com/FOD-UNOmaha/FOD-data), Pascal VOC sürümü (MIT lisansı).

## Klasörler
| Klasör | İçerik |
|---|---|
| `configs/` | YOLO veri ayar dosyası (`fod.yaml`) |
| `scripts/` | Veri dönüştürme, bölme, kontrol, eğitim ve değerlendirme script'leri |
| `notebooks/` | Colab defteri (`colab_train.ipynb`) |
| `data/raw/` | İndirilen orijinal veri. **Hiç değiştirilmez.** |
| `data/yolo/` | YOLO formatına çevrilmiş veri: `images/` ve `labels/` altında `train`, `val`, `test`. `all/` altında bölünmemiş etiketler, `boxes.csv` içinde tüm kutuların listesi |
| `results/` | Her deneyin sonuçları. `summary.csv` tüm deneylerin özetidir. |

`data/` GitHub'a yüklenmez, Google Drive'da tutulur.

## Kararlar
- **Ana model tek sınıflıdır:** 31 sınıfın hepsi "FOD" sınıfında birleşir (`data/yolo/all/labels_fod/`).
- **Orijinal 31 sınıf saklanır:** sınıf adları `configs/classes_31.txt` içinde, 31 sınıflı etiketler `data/yolo/all/labels_31/` içinde, her kutunun orijinal sınıfı `data/yolo/boxes.csv` içinde.
- **Hazır `trainval.txt` / `test.txt` ayrımı kullanılmaz.** Art arda gelen video karelerini rastgele dağıttığı için sızıntılıdır.
- **Bölme video parçası bazında yapılır:** bir video parçası bütün olarak train, val ya da test'e gider. Tek videolu sınıflar tek bir bölüme düşebilir.
- **Test raporu iki grup için ayrı hesaplanır:** eğitimde görülen sınıflar ve eğitimde hiç olmayan sınıflar.

## İleride yapılacaklar
- Karşılaştırma için hazır (sızıntılı) `trainval.txt` / `test.txt` ayrımıyla da bir eğitim yapılacak. Amaç, sızıntının puanı ne kadar şişirdiğini göstermek.
- 31 sınıflı deney.

### Düşük maliyet hedefi için deneyler
Ana hedef: radar ve özel kamera kullanan pahalı ticari FOD sistemlerine karşı düşük maliyetli bir alternatif.
- **Nano ve small karşılaştırması:** YOLO'nun nano ve small sürümleri aynı veriyle eğitilecek; doğruluk (mAP, precision, recall) ve hız yan yana karşılaştırılacak.
- **Yalnızca işlemciyle hız ölçümü:** Modelin ekran kartı olmadan saniyede kaç kare işlediği ölçülecek. Bu ölçüm ucuz cihazı temsil eder.
- **Donanım maliyet tablosu:** Kullanılacak donanımın (kamera, küçük bilgisayar vb.) maliyeti tablo halinde çıkarılacak.

Fiyat kuralı: ticari sistemlerin fiyatları tahminle yazılmaz. Kaynağı olmayan fiyat tabloya konmaz, yerine "bilinmiyor" yazılır.

## Kurallar
- Train, val ve test görüntüleri asla karışmaz. Test seti proje sonuna kadar eğitimde kullanılmaz.
- Her deneyin mAP, precision ve recall değerleri `results/` klasörüne kaydedilir.
