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
- **Bölme video parçası bazında yapılır:** bir video parçası bütün olarak train, val ya da test'e gider. Tek videolu sınıflar tek bir bölüme düşebilir. Ayrıntılı kurallar aşağıda.
- **Test raporu iki grup için ayrı hesaplanır:** eğitimde görülen sınıflar ve eğitimde hiç olmayan sınıflar.

## Bölme kuralları (2. sürüm, son)
Video parçası: görüntüler numara sırasıyla gezilir; sınıf ya da hava/ışık değişince yeni parça başlar. Veri setinde 116 parça var. Bölmeyi `scripts/make_splits.py` üretir (seed 42), kayıt `configs/splits.csv` içindedir.

1. Bir parça asla ikiye bölünmez.
2. Val ve test, toplam karelerin %9-%11'idir.
3. Tek bir parça, val ya da test'in yarısını geçemez.
4. Test'te Bolt, Nut ve Rock'tan en az birer parça bulunur.
5. Val'de en az 12 parça bulunur ve tek bir sınıf val karelerinin %25'ini geçemez.
6. 32 pikselden küçük kutuların oranı val ve test'te train'e yakındır (en fazla 5 puan fark).
7. Test'te, eğitimde hiç görülmeyen 2-4 sınıf bulunur.

Kurallara uyan bölmeler içinden test'te en çok farklı sınıf olan seçilir.

### Neden revize edildi
İlk bölme (1. sürüm, kayıtları `results/split_v1/` içinde) yalnızca 1-3. kurallarla üretildi ve şu sorunlar çıktı:
- Test'te hiç Bolt yoktu (hedef cisimlerden biri).
- Val zayıftı: 10 parça, 10 sınıf; karelerin %45'i Bolt ve BoltWasher'dı. En iyi model val'e göre seçildiği için seçim bu iki sınıfa göre yapılmış olacaktı.
- En küçük kutular test'te azdı: 32 pikselden küçük kutular train'de %20,1, test'te %11,1. Gerçek pistte cisimler küçük görünür; test en zor durumu az ölçecekti.

Revizyon, hiçbir model eğitilmeden ve hiçbir test sonucu görülmeden yapıldı; bu yüzden test setini dondurma kuralına aykırı değildir. **Bu son değişikliktir: ilk eğitimden sonra test seti donar** ve `make_splits.py` kayıtlı bölmeden farklı bir bölme üretirse durur.

## Sınırlar
- **Benzerlik kontrolü bu veride ayırt edici değil.** `check_splits.py` görüntülerin 64 bitlik parmak izlerini (perceptual hash) karşılaştırır. Düz beton üstündeki küçük cisimler bu yönteme hep aynı görünür: gözle bakılan en benzer çiftler farklı cisim ve farklı sahne çıktı (örneğin cıvata-pul ile taş, 0 bit fark). Aynı videodaki komşu karelerin de yalnızca %83,8'i eşiğin altında kalır. Bu yüzden "benzer çift" sayıları sızıntı kanıtı sayılmaz; sızıntıya karşı asıl güvence, video parçalarının bölünmemesidir.
- **Aynı cisim farklı ışıkta farklı bölümlere düşebilir.** Parçalar sınıf ve hava/ışık değişimine göre ayrılır. Aynı cismin kuru-aydınlık videosu train'de, loş ya da karanlık videosu test'te olabilir. Bu durumda test, "hiç görülmemiş cisim" değil "görülmüş cismin başka koşuldaki hali" ölçer. Eğitimde hiç olmayan sınıflar bu yüzden ayrıca raporlanır.
- **Video parçası sınırları tahmindir.** Dosya adlarında video bilgisi yok; sınırlar etiketlerden çıkarıldı.
- **Veri gerçek pist kamerasına benzemiyor.** Görüntüler 300×300 yakın plan çekimler, kutuların yaklaşık %40'ının uzun kenarı 96 piksel ve üzeri. Cisimsiz (boş) kare yok; yanlış alarm oranı bu veriyle ölçülemez.

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
