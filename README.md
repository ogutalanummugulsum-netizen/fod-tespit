# FOD Tespiti (YOLO)

Havalimanı pist görüntülerinde vida, somun, taş gibi küçük yabancı cisimleri (FOD) tespit eden model.
Ana hedef: radar ve özel kamera kullanan pahalı ticari FOD sistemlerine karşı düşük maliyetli bir alternatif.
Proje hedefi: TEKNOFEST Üniversite Öğrencileri Araştırma Proje Yarışması'na ("Savunma, Uzay ve Havacılık" kategorisi) sunmak ve ileride ticari bir ürüne dönüştürmek.
Veri seti: [FOD-A](https://github.com/FOD-UNOmaha/FOD-data), Pascal VOC sürümü (MIT lisansı).

## Klasörler
| Klasör | İçerik |
|---|---|
| `configs/` | YOLO veri ayar dosyası (`fod.yaml`) |
| `scripts/` | Veri dönüştürme, bölme, kontrol, eğitim ve değerlendirme script'leri |
| `notebooks/` | Colab defteri (`colab_train.ipynb`) |
| `data/raw/` | İndirilen orijinal veri. **Hiç değiştirilmez.** |
| `data/yolo/` | YOLO formatına çevrilmiş veri: `images/` ve `labels/` altında `train`, `val`, `test`. `all/` altında bölünmemiş etiketler, `boxes.csv` içinde tüm kutuların listesi |
| `data/yolo_sizintili/` | Yalnızca sızıntı karşılaştırması için: hazır (sızıntılı) ayrımla aynı veri. Asıl deneylerde kullanılmaz |
| `results/` | Her deneyin sonuçları. `summary.csv` tüm deneylerin özetidir. |

`data/` GitHub'a yüklenmez, Google Drive'da tutulur.

## Kararlar
- **Ana model tek sınıflıdır:** 31 sınıfın hepsi "FOD" sınıfında birleşir (`data/yolo/all/labels_fod/`).
- **Orijinal 31 sınıf saklanır:** sınıf adları `configs/classes_31.txt` içinde, 31 sınıflı etiketler `data/yolo/all/labels_31/` içinde, her kutunun orijinal sınıfı `data/yolo/boxes.csv` içinde.
- **Hazır `trainval.txt` / `test.txt` ayrımı kullanılmaz.** Art arda gelen video karelerini rastgele dağıttığı için sızıntılıdır.
- **Bölme video parçası bazında yapılır:** bir video parçası bütün olarak train, val ya da test'e gider. Tek videolu sınıflar tek bir bölüme düşebilir. Ayrıntılı kurallar aşağıda.
- **Test raporu iki grup için ayrı hesaplanır:** eğitimde görülen sınıflar ve eğitimde hiç olmayan sınıflar.

## Bölme donduruldu
**4 Ekim 2026'da bölme onaylandı ve donduruldu.** Geçerli bölme, `41a85b89e` numaralı commit'teki `configs/splits.csv` dosyasıdır: train 26.906, val 3.448, test 3.439 kare (81 / 16 / 19 video parçası). Bölme artık değişmeyecek. `make_splits.py` kayıtlı bölmeden farklı bir sonuç üretirse durur. Bölmenin raporları `results/split_report.txt`, `results/split_check.txt` ve `results/box_size_report.csv` içindedir.

## Test kuralı (sonuçlar görülmeden, 4 Ekim 2026'da yazıldı)
Bu kural, hiçbir model eğitilmeden ve hiçbir test sonucu görülmeden kararlaştırıldı.

- **Test edilecek modeller:** iki nihai model. YOLO11 nano'nun nihai modeli ve YOLO11 small'un nihai modeli.
- **Kaç kez:** her nihai model test setinde **bir kez** değerlendirilir. Toplam iki test değerlendirmesi.
- **Neye göre seçim:** hangi modelin daha iyi olduğuna **val** sonuçlarına bakılarak karar verilir. Epoch sayısı, görüntü boyutu ve diğer bütün ayarlar da val'e göre seçilir. Test sonucu hiçbir seçimde kullanılmaz, yalnızca son raporda yer alır.
- **Deneme eğitimleri test edilmez.** Kısa deneme (`exp001_nano_deneme`) ve ara deneyler yalnızca val üzerinde ölçülür.
- **Nasıl korunur:** `configs/fod.yaml` içinde test yolu yoktur, eğitim test klasörünü bilmez. `scripts/evaluate.py` test için `--final-test` ister, her test değerlendirmesini `results/test_log.csv` dosyasına yazar ve aynı deney için ikinci kez çalışmaz.
- **Rapor:** test sonucu dört ölçümle (mAP50, mAP50-95, precision, recall) ve üç kırılımla verilir: eğitimde görülen / görülmeyen sınıflar, ışık koşulu, kutu boyutu. Her satırda kaç kareye ve kaç video parçasına dayandığı yazılır. Işık koşulu sonuçlarının, bütün verideki koşul dağılımına göre ağırlıklı ortalaması da verilir.

### Güven eşiği kuralı (eşik tablosu görülmeden, 16 Kasım 2026'da yazıldı)
Bu kural, val eşik tablosu (`val_thresholds.csv`) hiç görülmeden kararlaştırıldı.

- **Kural:** precision ve recall, **val'de F2'yi en yüksek yapan güven eşiğiyle** ölçülür. F2, recall'a precision'dan iki kat ağırlık verir. Gerekçe: pist güvenliğinde cisim kaçırmak, yanlış alarmdan daha tehlikelidir.
- **Nasıl hesaplanır:**
  - F2, bütün val üzerinden hesaplanır ("tümü" satırları), her nihai model için ayrı ayrı.
  - Adaylar, `val_thresholds.csv` tablosundaki eşiklerdir: 0,05'ten 0,60'a 0,05 adımla, artı val'de F1'i en yüksek yapan eşik.
  - Model kutusu, gerçek kutuyla en az %50 örtüşürse doğru sayılır.
  - İki eşiğin F2'si eşitse yüksek olan seçilir, çünkü daha az yanlış alarm verir.
- **Test'te:** bu eşik yalnızca val'e bakılarak belirlenir ve test değerlendirmesinde değiştirilmeden kullanılır. mAP50 ve mAP50-95 eşikten bağımsızdır.
- **Önceki eşik:** exp001 ve exp002'nin ilk val raporlarında precision ve recall, F1'i en yüksek yapan eşikle verildi (exp001: 0,60, exp002: 0,61). exp001 raporu olduğu gibi kalır. exp002, nihai model adayı olduğu için F2 kuralıyla yeniden değerlendirilir; F1'li ilk raporu git geçmişinde durur (commit `131fa4c70`). `summary.csv` içindeki `esik_kurali` sütunu her satırın hangi kuralla ölçüldüğünü gösterir.
- **Ölçüm yöntemi (16 Kasım 2026'dan itibaren):** precision, recall ve hata sayıları basit eşleştirmeyle hesaplanır: model kutusu, gerçek kutuyla en az %50 örtüşürse (IoU ≥ 0,5) doğru sayılır. Önceki raporlarda precision ve recall, Ultralytics'in kendi eğrilerinden okunuyordu; iki yöntem aynı eşikte farklı sonuç verebilir. Örneğin exp002'de 0,61 eşiğinde Ultralytics precision 0,9716 / recall 0,9649 verirken basit eşleştirme 0,9849 / 0,9632 veriyor. Bu yüzden eski ve yeni raporların precision ve recall değerleri doğrudan karşılaştırılmamalı. mAP50 ve mAP50-95 yine Ultralytics ile ölçülür.

### Yanlış alarmların ayrımı (yalnızca raporlama için)
Yanlış alarmlar (fazladan kutular) iki gruba ayrılır. Bu ayrım eşik kuralını değiştirmez; val'de ve en sonda test'te aynı şekilde uygulanır.
- **Cisimde:** gerçek bir cismin kutusuyla kesişen ya da ona 16 pikselden yakın olan fazladan kutu. Örnek: cismin bir parçasına çizilen kutu, aynı cisme çizilen ikinci kutu.
- **Zeminde:** hiçbir gerçek cisme yakın olmayan fazladan kutu. Boş zeminde verilen alarm.

### Raporlama ilkesi
- **Asıl sonuçlar her zaman bütün val üzerinden verilir.** En sonda da bütün test üzerinden verilir.
- Bir videonun ya da grubun çıkarıldığı hesaplar yalnızca açıklayıcı bilgidir, asıl sonuç olarak sunulmaz. Örneğin exp002'de, F1 eşiğinde (0,61) val'de kaçırılan 127 cismin 88'i tek bir PaintChip videosundan geliyor; o video çıkarılınca recall yaklaşık 0,99 olur (39 / 3.271 kaçırma). Bu bilgi yalnızca hatanın nereden geldiğini açıklamak için verilir. exp002'nin o eşikteki asıl val recall'ı 0,9649'du; F2 kuralıyla (0,25) asıl val recall'ı 0,9936'dır.

### Model karşılaştırma ölçütü (17 Kasım 2026)
**Bu ölçüt, nano ve small'un val sonuçları görüldükten sonra, test sonucu görülmeden belirlendi.** Daha önce README'de yalnızca "hangisinin daha iyi olduğuna val'e bakılarak karar verilir" yazıyordu; hangi ölçüyle karar verileceği tanımlanmamıştı.

- **Ölçüt:** her modelin kendi F2 eşiğinde, bütün val üzerinden hesaplanan **F2**. Eşik kuralıyla tutarlıdır ve "cisim kaçırmak yanlış alarmdan tehlikelidir" ilkesine uyar.
- **Sonuç ve önerilen model: nano.** Val'de nano (exp002, eşik 0,25) F2 = 0,9806; small (exp003, eşik 0,40) F2 = 0,9732. Nano aynı zamanda daha küçük ve düşük maliyet hedefine daha uygun.
- **Test kuralı değişmiyor:** iki nihai model de (exp002_nano ve exp003_small) test setinde birer kez değerlendirilir. Bu ölçüt yalnızca hangisinin önerilen model olacağını belirler; test sonucu bu seçimi değiştirmez.
- **Not:** iki model arasındaki farklar küçük ve büyük ölçüde birkaç videodan (PaintChip, Wire, Hammer) geliyor. small eşikten bağımsız mAP50-95'te biraz önde (0,834'e karşı 0,828).

### Nihai modeller (17 Kasım 2026)
Test'ten önce başka model denemesi yapılmayacak. Nihai modeller **exp002_nano** ve **exp003_small**'dur. Aynı cisme çizilen ikinci kutular modelde değil, uyarı sistemindeki zaman filtresiyle ele alınacak (bkz. "İleride yapılacaklar").

### Test raporunda verilecekler (17 Kasım 2026, test çalıştırılmadan önce yazıldı)
Bu liste test sonucu görülmeden yazıldı. **Sonuç ne çıkarsa çıksın, aşağıdakilerin hepsi raporlanacak.** Test sonucuna göre model, eşik, kırılım ya da gruplar değiştirilmeyecek; bir satır iyi ya da kötü çıktığı için rapordan çıkarılmayacak.

**Neler, kaç kez, hangi eşikle**
- İki nihai model: exp002_nano ve exp003_small. Her biri test setinde **bir kez**, aynı çalıştırmada değerlendirilir.
- Her model **kendi val F2 eşiğiyle** ölçülür: nano 0,25, small 0,40. Eşikler test'te yeniden hesaplanmaz.
- mAP50 ve mAP50-95 eşikten bağımsızdır. Precision, recall ve hata sayıları basit eşleştirmeyle (IoU ≥ 0,5) hesaplanır.
- Test seti: 3.439 kare, 19 video parçası (`configs/splits.csv`, commit `41a85b89e`).

**Her tabloda, her satır için verilecekler**
- Ölçümler: mAP50, mAP50-95, precision, recall.
- Hatalar: fazladan kutu (yanlış alarm) toplamı, bunun **cisimde** ve **zeminde** olanları, kaçırılan cisim.
- Dayanak: kare, kutu ve video parçası sayısı. 3'ten az video parçasına dayanan satır "az veri" diye işaretlenir.

**Tablolar**
1. **Genel:** bütün test seti. Asıl sonuç budur.
2. **Sınıf grubu:** eğitimde görülen sınıflar ve eğitimde hiç görülmeyen sınıflar (test'te Screw ve Tape) ayrı ayrı.
3. **Işık koşulu:** kuru-aydınlık, kuru-loş, kuru-karanlık, ıslak-loş ayrı ayrı. Ayrıca bu dört sonucun, bütün verideki koşul dağılımına göre ağırlıklı ortalaması (aydınlık %50,1, ıslak-loş %21,4, loş %15,5, karanlık %13,0). Ağırlıklı ortalama "kaba tahmin" olarak işaretlenir.
4. **Kutu boyutu:** uzun kenarı 16 pikselden küçük, 16-32, 32-64, 64-96, 96 piksel ve üzeri.
5. **Karşılaştırmalar:** nano ve small'un test sonuçları yan yana; her modelin val ve test sonuçları yan yana.

**Raporda ayrıca**
- İşlemci hız ölçümü (`results/speed_cpu.csv`): parametre sayısı, dosya boyutu, iş parçacığı sayısıyla birlikte süre ve FPS. Hız ölçümü val görüntüleriyle yapılır, test görüntüleri kullanılmaz.
- "Sınırlar" bölümündeki uyarılar, özellikle test'in yaklaşık yarısının karanlık çekim olduğu, test'teki Bolt'un tek kısa video olduğu ve 16 pikselden küçük kutuların test'te az olduğu.

**Test'te yapılmayacaklar:** video parçası bazında kırılım, hata örneği görüntüleri ve eşik tablosu yalnızca val içindir; test'te çıkarılmaz.

**Test'ten sonra:** modelde, eşikte ya da son işlemede (NMS vb.) yapılacak her değişiklik yeni bir sürümdür ve ayrı raporlanır. Yeni sürüm bu test setinde değerlendirilmiş sayılmaz.

## Test sonuçları (18 Kasım 2026)
İki nihai model, test setinde birer kez değerlendirildi (`results/test_log.csv`; ağırlıklar Drive'daki `best.pt`, kod commit `504f8cd`, bölme commit `41a85b89e`). Her model kendi val F2 eşiğiyle ölçüldü: nano 0,25, small 0,40. Rapor, "Test raporunda verilecekler" listesine göre hazırlandı; hiçbir satır çıkarılmadı. Ham sonuçlar: `results/exp002_nano/test_breakdown.csv`, `results/exp003_small/test_breakdown.csv`, `test_metrics.json` dosyaları ve `results/summary.csv`.

Kısaltmalar: P = precision, R = recall, FA = fazladan kutu (yanlış alarm), cisimde / zeminde = FA'nın ayrımı, Kaç. = kaçırılan cisim. Her satırda kare sayısı = kutu sayısı (her karede bir cisim var).

### 1. Genel (asıl sonuç): bütün test, 3.439 kare, 19 video parçası

| Model | Eşik | mAP50 | mAP50-95 | P | R | FA | cisimde | zeminde | Kaç. |
|---|---|---|---|---|---|---|---|---|---|
| exp002_nano | 0,25 | **0,9932** | **0,7811** | **0,9506** | **0,9788** | **175** | 107 | 68 | **73** |
| exp003_small | 0,40 | 0,9846 | 0,7711 | 0,9436 | 0,9529 | 196 | 187 | 9 | 162 |

### 2. Sınıf grubu

| Grup | Kare | Parça | Model | mAP50 | mAP50-95 | P | R | FA | cisimde | zeminde | Kaç. | Not |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Eğitimde var | 3.155 | 17 | nano | 0,9930 | 0,7835 | 0,9513 | 0,9778 | 158 | 95 | 63 | 70 | |
| | | | small | 0,9833 | 0,7745 | 0,9426 | 0,9528 | 183 | 176 | 7 | 149 | |
| Eğitimde yok (Screw, Tape) | 284 | 2 | nano | 0,9950 | 0,7639 | 0,9430 | 0,9894 | 17 | 12 | 5 | 3 | az veri |
| | | | small | 0,9950 | 0,7414 | 0,9542 | 0,9542 | 13 | 11 | 2 | 13 | az veri |

### 3. Işık koşulu

| Koşul | Kare | Parça | Model | mAP50 | mAP50-95 | P | R | FA | cisimde | zeminde | Kaç. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Kuru-aydınlık (Dry-Bright) | 448 | 3 | nano | 0,9599 | 0,3938 | 0,9324 | 0,9241 | 30 | 29 | 1 | 34 |
| | | | small | 0,9211 | 0,3217 | 0,8981 | 0,8460 | 43 | 42 | 1 | 69 |
| Kuru-loş (Dry-Dim) | 498 | 3 | nano | 0,9946 | 0,8324 | 0,9447 | 0,9940 | 29 | 29 | 0 | 3 |
| | | | small | 0,9800 | 0,7943 | 0,8721 | 0,9719 | 71 | 71 | 0 | 14 |
| Kuru-karanlık (Dry-Dark) | 1.562 | 8 | nano | 0,9936 | 0,8093 | 0,9351 | 0,9776 | 106 | 41 | 65 | 35 |
| | | | small | 0,9923 | 0,8193 | 0,9722 | 0,9622 | 43 | 37 | 6 | 59 |
| Islak-loş (Wet-Dim) | 931 | 5 | nano | 0,9947 | 0,8780 | 0,9894 | 0,9989 | 10 | 8 | 2 | 1 |
| | | | small | 0,9929 | 0,8870 | 0,9589 | 0,9785 | 39 | 37 | 2 | 20 |
| Ağırlıklı ortalama (bütün verinin dağılımı), **kaba tahmin** | 3.439 | 19 | nano | 0,9771 | 0,6192 | 0,9468 | 0,9579 | - | - | - | - |
| | | | small | 0,9548 | 0,5804 | 0,9167 | 0,9089 | - | - | - | - |

Ağırlıklar: aydınlık %50,1, ıslak-loş %21,4, loş %15,5, karanlık %13,0. Ağırlıklı ortalama genel sonuçtan düşük, çünkü test'te en kötü çıkan koşul (aydınlık) bütün veride en sık koşul.

### 4. Kutu boyutu (uzun kenar)

| Boyut | Kare | Parça | Model | mAP50 | mAP50-95 | P | R | FA | cisimde | zeminde | Kaç. | Not |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| <16 px | 193 | 2 | nano | 0,8105 | 0,2290 | 0,7347 | 0,7461 | 52 | 30 | 22 | 49 | az veri |
| | | | small | 0,7425 | 0,2006 | 0,8170 | 0,6477 | 28 | 28 | 0 | 68 | az veri |
| 16-32 px | 367 | 8 | nano | 0,9882 | 0,4512 | 0,9747 | 0,9455 | 9 | 8 | 1 | 20 | |
| | | | small | 0,9804 | 0,4187 | 0,9462 | 0,9101 | 19 | 18 | 1 | 33 | |
| 32-64 px | 603 | 13 | nano | 0,9949 | 0,7812 | 0,9726 | 1,0000 | 17 | 17 | 0 | 0 | |
| | | | small | 0,9949 | 0,7811 | 0,9723 | 0,9884 | 17 | 15 | 2 | 7 | |
| 64-96 px | 699 | 15 | nano | 0,9950 | 0,7968 | 0,9762 | 0,9957 | 17 | 12 | 5 | 3 | |
| | | | small | 0,9940 | 0,7929 | 0,9698 | 0,9642 | 21 | 19 | 2 | 25 | |
| ≥96 px | 1.577 | 16 | nano | 0,9949 | 0,8913 | 0,9517 | 0,9994 | 80 | 40 | 40 | 1 | |
| | | | small | 0,9908 | 0,8918 | 0,9331 | 0,9816 | 111 | 107 | 4 | 29 | |

### 5. Karşılaştırmalar

**Val ve test yan yana** (val: 3.448 kare, 16 parça; test: 3.439 kare, 19 parça; aynı eşik):

| Model | Bölüm | mAP50 | mAP50-95 | P | R | FA | cisimde | zeminde | Kaç. |
|---|---|---|---|---|---|---|---|---|---|
| nano | val | 0,9846 | 0,8281 | 0,9317 | 0,9936 | 251 | 220 | 31 | 22 |
| nano | test | 0,9932 | 0,7811 | 0,9506 | 0,9788 | 175 | 107 | 68 | 73 |
| small | val | 0,9835 | 0,8338 | 0,9210 | 0,9872 | 292 | 277 | 15 | 44 |
| small | test | 0,9846 | 0,7711 | 0,9436 | 0,9529 | 196 | 187 | 9 | 162 |

**Nano ve small:** yan yana karşılaştırma için 1-4. tablolara bakın. Karşılaştırma ölçütü olan F2 (val'de belirlendi), test'in P ve R değerlerinden hesaplanınca: nano 0,973, small 0,951. Sıralama val ile aynı.

**İşlemci hızı:** bkz. "Deney sırası", 4. madde (iki Colab işlemcisi, 1 ve 2 iş parçacığı). Hız val görüntüleriyle ölçüldü; test görüntüleri kullanılmadı.

### Sonuçların özeti
- **Nano, test'te small'dan daha iyi:** mAP50, mAP50-95, P ve R'nin dördünde de önde, daha az yanlış alarm veriyor ve yarıdan az cisim kaçırıyor (73'e karşı 162). Ayrıca yaklaşık 2 kat hızlı ve 3,5 kat küçük. Val'e göre yapılan "nano önerilir" seçimi test'te de geçerli.
- **Val'den test'e recall düştü** (nano 0,994'ten 0,979'a, small 0,987'den 0,953'e) ve mAP50-95 düştü (kutu yerinin tam bulunması). mAP50 düşmedi.
- **En zayıf noktalar:** 16 pikselden küçük kutular (nano R 0,746; 73 kaçırmanın 49'u bu satırda, ama satır yalnızca 2 parçaya dayanıyor) ve kuru-aydınlık çekimler (nano R 0,924, mAP50-95 0,394).
- **Eğitimde hiç görülmemiş cisim türleri** (Screw, Tape) nano'da 284 kutudan 3 kaçırmayla bulundu (2 parça, az veri).
- **Nano'nun zemindeki yanlış alarmları** val'de 31 iken test'te 68 oldu; 65'i kuru-karanlık karelerde. Val'de bu türden bazı alarmların etiketlenmemiş gerçek cisim olduğu görülmüştü; test'te hata görüntüsü çıkarılmadığı için bu kontrol edilmedi ve edilmeyecek.

### Bu sonuçları okurken (bkz. "Sınırlar")
- Test'in yaklaşık yarısı (%45) karanlık çekim; aydınlık çekimler test'in yalnızca %13'ü (3 parça).
- Test'teki Bolt tek kısa bir video (182 kare) ve içinde 32 pikselden küçük kutu yok.
- 16 pikselden küçük kutular test'te az (193 kutu, 2 parça); bu satır "az veri".
- Görüntüler 300×300 yakın plan ve her karede bir cisim var; boş kare olmadığı için cisimsiz bir pistteki yanlış alarm oranı bu testle ölçülemez. Veri gerçek pist kamerasına benzemiyor; gerçek dünya başarısı telefon kamerası testinde ayrıca ölçülecek.

## Sızıntılı ayrım karşılaştırması (plan ve yorum kuralları, 18 Kasım 2026, sonuç görülmeden yazıldı)
Amaç: veri setinin hazır `trainval.txt` / `test.txt` ayrımıyla eğitilen bir modelin puanının ne kadar şişkin çıktığını göstermek. Bu bölüm, `exp004_nano_sizintili` eğitilmeden ve hiçbir sonucu görülmeden yazıldı.

**Nasıl yapılır**
- Hazır ayrım: listelerde trainval 25.345, test 8.448 satır; aşağıdaki düzeltmelerden sonra trainval 25.294, test 8.429 kare. Trainval'den rastgele %10 kare (seed 42, kare bazında) val olarak ayrılır. Bu val de bilerek sızıntılıdır; amaç yaygın "rastgele kare bölme" yöntemini taklit etmek. Hazır test, model seçiminde ve eşik belirlemede kullanılmaz.
- Model: yalnızca nano, `exp004_nano_sizintili`. Ayarlar exp002 ile aynı: yolo11n, en fazla 50 epoch, patience 10, görüntü boyutu 320, batch 64, seed 0. Güven eşiği sızıntılı val'de F2 kuralıyla seçilir.
- Ayrı veri klasörü (`data/yolo_sizintili`, `scripts/make_leaky_split.py`). Bizim bölmemize (`data/yolo`, `configs/splits.csv`) dokunulmaz. Sızıntı raporu: `results/leaky_split_report.txt`.
- **Hazır listelerin sorunları ve nasıl ele alındığı** (18 Kasım 2026, hiçbir model eğitilmeden ve sonuç görülmeden eklendi; ilk çalıştırmada script bu sorunlar yüzünden durdu):
  - Bazı numaralar bir listede birden çok kez yazılmış: trainval'de 4 numara (47 fazla satır), test'te 4 numara (19 fazla satır). **Tek kareye indirilir.**
  - 4 görüntü iki listede birden var (014567, 015731, 016653, 016763; tekrar eden numaraların aynısı). **Trainval'den çıkarılır, yalnızca test'te kalır.**
  - 70 görüntü hiçbir listede yok. **Kullanılmaz.**
  - Her birinin sayısı `results/leaky_split_report.txt` içine yazılır. Bu 4 + 70 görüntünün hepsi bizim bölmede train'de; bu yüzden ortak kare sayısı (845) değişmez.
- exp004, hazır test'te bir kez değerlendirilir (`test_log_sizintili.csv`). Sonuçlar `sizintili_val_*` ve `sizintili_test_*` adlı dosyalara yazılır, bizim test sonuçlarıyla karışmaz.

**İki ölçüm**
1. **Ortak kare karşılaştırması: sızıntının doğrudan etkisi (asıl ölçüm).** Hem hazır test'te hem bizim test setimizde olan kareler: 845 kare, bizim test'in 19 video parçasının hepsinden (bu sayılar sonuç görülmeden, yalnızca dosya listelerinden sayıldı). Bu parçaların hepsinin başka kareleri hazır trainval'de var.
   - exp002_nano bu karelerin video parçalarını hiç görmedi (dürüst).
   - exp004 aynı parçaların komşu karelerini eğitimde gördü (sızıntılı).
   - İki model aynı karelerde, her biri kendi val F2 eşiğiyle ölçülür. Aradaki fark sızıntının doğrudan etkisi olarak raporlanır. Kırılımlar: tümü, sınıf grubu, ışık koşulu, kutu boyutu. Ayrıca aynı cisim için eşleştirilmiş tablo: ikisi de buldu / yalnız exp002 / yalnız exp004 / ikisi de kaçırdı.
2. **Raporlanan puan farkı.** exp004'ün hazır test'teki (8.429 kare) sonucu ve exp002'nin bizim test'teki (3.439 kare) sonucu yan yana. İki test seti farklı karelerden oluştuğu için bu farkın tamamı sızıntı değildir; "bir makalede raporlanacak puan" ile "dürüst puan" arasındaki fark diye verilir. Işık koşulu ağırlıklı ortalaması iki model için aynı ağırlıklarla verilir.

**Test kuralına istisna (yalnızca raporlama)**
- exp002_nano, bizim test setinde **yalnızca bu 845 ortak karede bir kez daha** çalıştırılır. Nedeni: ilk test'te kare kare tahminler kaydedilmedi.
- exp004 bizim test setinde yalnızca bu ortak karelerde değerlendirilir. Bu kareler zaten kendi hazır test'inin parçası.
- Bu bir model seçimi ya da ayar değişikliği değildir. Model ve eşikler sabittir (exp002: 0,25). Sonuç ne çıkarsa çıksın nihai model, eşik ve bizim test sonuçları değişmez. exp004 demo, model seçimi ya da başka bir amaçla kullanılmaz.
- Korumalar (`scripts/compare_leak.py`): `--final-test-subset` onayı gerekir. İki modelin kendi test değerlendirmesi önceden bitmiş olmalıdır. Karşılaştırma yalnızca bir kez çalışır (`results/leak_compare_log.csv`). Defterde `IS = "sizinti"` ile birlikte `TEST_ONAY = "evet"` gerekir.
- Ortak karelerde video parçası kırılımı, hata görüntüsü ve eşik tablosu çıkarılmaz.
- Tutarlılık kontrolü: exp002'nin ortak karelerdeki hata sayıları, bütün test'teki hata sayılarından (175 yanlış alarm, 73 kaçırma) fazla olamaz. Fazla çıkarsa bu da raporlanır.

**Yorum kuralları**
- Bütün sayılar, iyi ya da kötü, raporlanır. Fark için bir "anlamlı" eşiği konmaz; farklar sayı olarak ve dayandığı kare, kutu ve video parçası sayısıyla verilir. 3'ten az parçaya dayanan satır "az veri" diye işaretlenir.
- Sızıntılı model ortak karelerde dürüst modelden daha iyi çıkarsa, bu fark "sızıntının puanı şişirmesi" olarak yorumlanır. Fark çıkmaz ya da ters çıkarsa bu da aynen yazılır.
- Farkı etkileyebilecek başka nedenler de rapora yazılır: exp004'ün eğitim verisi daha az (yaklaşık 22.800 kareye karşı 26.906) ve farklı karelerden oluşuyor. exp004, bizim test'te "eğitimde yok" sayılan Screw ve Tape'i eğitimde gördü; bu da sızıntının bir parçasıdır. Ayrıca exp002'nin yeniden çalıştırılmasında, ilk test'e göre (işlemci ile ekran kartı farkından) çok küçük sayısal farklar olabilir.

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
- **Test ve val, ışık koşulu olarak veri setinin geneline benzemiyor.** Test karelerinin yaklaşık yarısı karanlık çekim (%45; bütün veride %13), val karelerinin yarıdan fazlası ıslak zemin (%56; bütün veride %21). Aydınlık çekimler bütün verinin %50'si, ama test'in %13'ü ve val'in %10'u. Test puanı ortalama koşuldan daha zor bir durumu ölçer.
- **Test'teki Bolt tek bir kısa video (182 kare) ve içinde 32 pikselden küçük kutu yok.** Test, küçük görünen Bolt'u ölçmez.
- **16 pikselden küçük kutular test'te train'e göre az:** test'te %5,6, train'de %9,3.
- **Video parçası sınırları tahmindir.** Dosya adlarında video bilgisi yok; sınırlar etiketlerden çıkarıldı.
- **Veri gerçek pist kamerasına benzemiyor.** Görüntüler 300×300 yakın plan çekimler, kutuların yaklaşık %40'ının uzun kenarı 96 piksel ve üzeri. Cisimsiz (boş) kare yok; yanlış alarm oranı bu veriyle ölçülemez.

## Deney sırası
1. `exp001_nano_deneme`: kısa deneme, 3 epoch. **Bitti** (4 Ekim 2026).
2. `exp002_nano`: nano'nun asıl eğitimi. En fazla 50 epoch, val puanı 10 epoch iyileşmezse erken durur, görüntü boyutu 320. **Bitti** (16 Kasım 2026): 26 epoch'ta durdu, en iyi epoch 16.
3. `exp003_small`: small modeli, exp002 ile aynı ayarlarla. Tek fark model. **Bitti** (17 Kasım 2026): 26 epoch'ta durdu, en iyi epoch 16.
4. Val'e göre nano-small karşılaştırması (**bitti**, önerilen model nano; bkz. "Model karşılaştırma ölçütü") ve yalnızca işlemciyle hız ölçümü:
   - Önce **Colab'ın işlemcisinde** (`scripts/benchmark_cpu.py`, defterde `IS = "hiz_olc"`). Sonuç: `results/speed_cpu.csv`. Her model için hız (ms, FPS), dosya boyutu (MB) ve parametre sayısı. Her model 1 ve 2 iş parçacığıyla ayrı ayrı ölçülür; iş parçacığı sayısı her satırda yazılır. (İlk ölçümde Ultralytics iş parçacığı sayısını kendisi "çekirdek sayısı - 1" = 1 yapmıştı; script artık sayıyı her tahminden önce ayarlıyor ve sonra doğruluyor.)
   - **Demo aşamasında aynı script proje sahibinin kendi bilgisayarında da çalıştırılacak** (`--where kendi_bilgisayarim`).
   - **Colab işlemcisine göre hız yaklaşık 2 kat değişiyor.** Colab her oturumda farklı bir işlemci verebiliyor. İki ölçüm yapıldı ve ikisi de saklanıp raporlanacak:

     | Tarih | İşlemci (Colab, 2 çekirdek) | Oturum | Model | İş parçacığı | Model süresi (ortanca) | Uçtan uca (ortanca) | Uçtan uca FPS |
     |---|---|---|---|---|---|---|---|
     | 17 Kasım 2026 | Intel Xeon @ 2.00GHz | GPU'lu oturum, model işlemcide | nano | 1 | 32,73 ms | 33,66 ms | 29,7 |
     | 17 Kasım 2026 | Intel Xeon @ 2.00GHz | GPU'lu oturum, model işlemcide | small | 1 | 76,36 ms | 77,39 ms | 12,9 |
     | 18 Kasım 2026 | Intel Xeon @ 2.20GHz | GPU'suz oturum | nano | 1 | 65,20 ms | 67,15 ms | 14,9 |
     | 18 Kasım 2026 | Intel Xeon @ 2.20GHz | GPU'suz oturum | nano | 2 | 60,46 ms | 62,27 ms | 16,1 |
     | 18 Kasım 2026 | Intel Xeon @ 2.20GHz | GPU'suz oturum | small | 1 | 136,65 ms | 138,15 ms | 7,2 |
     | 18 Kasım 2026 | Intel Xeon @ 2.20GHz | GPU'suz oturum | small | 2 | 129,43 ms | 131,01 ms | 7,6 |

     Model süresi = hazırlık + model + kutu ayıklama; uçtan uca = diskten okuma dahil. 300 val görüntüsü, görüntü boyutu 320. Nano: 2.590.035 parametre, 5,18 MB; small: 9.428.179 parametre, 18,25 MB. 17 Kasım'da yalnızca 1 iş parçacığı ölçüldü.
     - Her iki işlemcide de nano, small'dan yaklaşık 2,1-2,3 kat hızlı.
     - 2 iş parçacığı bu işlemcide yalnızca %5-8 hızlandırdı (nedeni ölçülmedi).

     Saat hızı daha yüksek olan işlemci daha yavaş çıktı. Nedeni ölçülmedi; büyük olasılıkla işlemcinin nesli ve desteklediği komutlar ya da oturum türü (GPU'lu / GPU'suz) farkı. Saat hızı tek başına hızı belirlemiyor. Bu yüzden hız sonuçları her zaman işlemci adıyla birlikte verilir; tek bir FPS değeri "Colab hızı" diye raporlanmaz. Bütün satırlar `results/speed_cpu.csv` içindedir.
5. İki nihai modelin test'te birer kez değerlendirilmesi (bkz. "Test kuralı"). **Bitti** (18 Kasım 2026); bkz. "Test sonuçları".
6. **Hazır (sızıntılı) ayrımla karşılaştırma:** nano ve small bittikten sonra, hazır `trainval.txt` / `test.txt` ayrımıyla da bir eğitim yapılacak. Amaç, sızıntının puanı ne kadar şişirdiğini göstermek. Plan ve yorum kuralları: "Sızıntılı ayrım karşılaştırması" bölümü (defterde `IS = "sizinti"`).
7. **Telefon kamerasıyla gerçek dünya testi ve demo** (nano ve small eğitimlerinden sonra). Projede özel kamera yok; test ve demo telefon kamerasıyla yapılacak:
   - Boş asfalt fotoğrafları ve asfalt üzerinde cisim fotoğrafları.
   - Maket zemin üzerinde demo videosu.

## Kararlar (14 Kasım 2026)
- **Hedef:** projeyi TEKNOFEST Üniversite Öğrencileri Araştırma Proje Yarışması'na ("Savunma, Uzay ve Havacılık" kategorisi) sunmak ve ileride ticari bir ürüne dönüştürmek.
- **Lisans:** ticari seçenekleri açık tutmak için şimdilik depoya MIT gibi serbest bir lisans eklenmeyecek. Lisans kararını proje sahibi verecek.
- **Kamera:** özel kamera yok. Gerçek dünya testi ve demo telefon kamerasıyla yapılacak (bkz. Deney sırası, 7. adım).

### Lisans kararı için not: kullanılan bileşenlerin lisansları
Bu bilgiler karar için hatırlatmadır, hukuki görüş değildir.
- **Ultralytics (YOLO): AGPL-3.0.** Kapalı kaynak bir ticari üründe kullanmanın şartları ayrıca incelenmeli; Ultralytics bunun için ayrı bir ticari lisans sunuyor.
- **FOD-A veri seti: MIT.** Ticari kullanıma izin verir, telif ve lisans bildiriminin korunmasını ister.
- **imagehash: BSD (2 maddeli).**

## İleride yapılacaklar
- 31 sınıflı deney.
- **NMS ve görüntü boyutu denemeleri (ertelendi):** aynı cisme çizilen ikinci kutuları azaltmak için kutu ayıklama (NMS) örtüşme eşiğini değiştirmek; küçük cisimler için görüntü boyutunu büyütmek. Bunlar nihai modellerin test değerlendirmesinden sonra yapılırsa **yeni sürüm** olarak ayrı raporlanır, test sonuçlarının yerine geçmez.
- **ONNX ile hız ölçümü (ertelendi):** modelleri işlemciye uygun ONNX biçimine çevirip yeniden hız ölçmek. Ucuz cihazlarda modeller genelde böyle çalıştırılır. Çevrilen modelin doğruluğu da val'de yeniden kontrol edilmeli.
- **Zaman filtresi (demo ve uyarı sistemi aşamasında):** bir cisim için uyarı verilmeden önce, cismin art arda birkaç karede görünmesi şart koşulacak. Amaç, tek karelik yanlış alarmları azaltmak. Bu filtre modelin ölçümünü ve eşik kuralını değiştirmez; uyarı sisteminin üstüne eklenir ve etkisi ayrıca ölçülür.

### Düşük maliyet hedefi için deneyler
Ana hedef: radar ve özel kamera kullanan pahalı ticari FOD sistemlerine karşı düşük maliyetli bir alternatif.
- **Nano ve small karşılaştırması:** YOLO'nun nano ve small sürümleri aynı veriyle eğitilecek; doğruluk (mAP, precision, recall) ve hız yan yana karşılaştırılacak.
- **Yalnızca işlemciyle hız ölçümü:** Modelin ekran kartı olmadan saniyede kaç kare işlediği ölçülecek. Bu ölçüm ucuz cihazı temsil eder.
- **Donanım maliyet tablosu:** Kullanılacak donanımın (kamera, küçük bilgisayar vb.) maliyeti tablo halinde çıkarılacak.

Fiyat kuralı: ticari sistemlerin fiyatları tahminle yazılmaz. Kaynağı olmayan fiyat tabloya konmaz, yerine "bilinmiyor" yazılır.

## Kurallar
- Train, val ve test görüntüleri asla karışmaz. Test seti proje sonuna kadar eğitimde kullanılmaz.
- Her deneyin mAP, precision ve recall değerleri `results/` klasörüne kaydedilir.
