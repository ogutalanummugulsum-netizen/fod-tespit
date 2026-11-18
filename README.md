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
| `telefon/` | Telefon testi: sınıf listesi, çekim defteri şablonu; her set için etiketler ve çekim defteri. **Fotoğraflar burada değil, Drive'da** |
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

## Sızıntılı ayrım sonuçları (18 Kasım 2026)
Yukarıdaki plan ve yorum kurallarına göre yapıldı; hiçbir satır çıkarılmadı. Kod commit `edeae19`. exp004 hazır test'te bir kez (`results/test_log_sizintili.csv`), ortak kare karşılaştırması bir kez (`results/leak_compare_log.csv`) çalıştırıldı. Ham sonuçlar: `results/exp004_nano_sizintili/`, `results/leak_compare/`, `results/leaky_split_report.txt` ve `results/summary.csv`.

Kısaltmalar: P = precision, R = recall, FA = fazladan kutu (yanlış alarm), cisimde / zeminde = FA'nın ayrımı, Kaç. = kaçırılan cisim.

### Sızıntı raporu (doğrulandı)
- Listelerin sorunları, README'de önceden yazılan sayılarla aynı çıktı: trainval'de 4 tekrar eden numara (47 fazla satır), test'te 4 tekrar eden numara (19 fazla satır), iki listede birden olan 4 görüntü (014567, 015731, 016653, 016763), hiçbir listede olmayan 70 görüntü (hepsi bizim bölmede train'de).
- Kare sayıları: train 22.765, val 2.529, test 8.429. Ortak kare 845, 19 video parçası.
- **Hazır test'in bütün kareleri (%100), eğitimde de bulunan bir video parçasından geliyor; %89,7'sinin hemen önceki ya da sonraki karesi eğitimde.** Ortak 845 karede bu oran %89,3. Hazır val'de de aynı: %100 ve %89,7.

### Eğitim
exp004, exp002 ile aynı ayarlarla eğitildi (yolo11n, en fazla 50 epoch, patience 10, 320, batch 64, seed 0; T4, yaklaşık 108 dakika). **Erken durmadı:** 50 epoch'un hepsini çalıştı, en iyi epoch 50 (exp002: 26 epoch'ta durdu, en iyi epoch 16). Son epoch'larda sızıntılı val puanı yatay (mAP50-95 0,899-0,900). Sızıntılı val'de seçilen güven eşiği (F2 kuralı) **0,40** (exp002: 0,25).

Bu önceden yazılmamış bir farktır, sonuçla birlikte not edilir: sızıntılı val, eğitim karelerine çok benzediği için puan uzun süre yükselmeye devam etti ve erken durma devreye girmedi. Yani exp004 daha uzun eğitildi; bu da ortak karelerdeki farkı etkileyebilecek nedenlerden biridir.

**Sızıntılı val ve sızıntılı test neredeyse aynı** (mAP50-95 0,9009 ve 0,9045; R 0,9899 ve 0,9898). Dürüst bölmede ise val'den test'e düşüş görülmüştü (nano mAP50-95 0,828'den 0,781'e). Sızıntılı ayrımda val de test de eğitime benzediği için bu düşüş görünmüyor.

### 1. Ortak kare karşılaştırması: sızıntının doğrudan etkisi (asıl ölçüm)
845 kare, 845 kutu, 19 video parçası. Her model kendi val F2 eşiğiyle (exp002 0,25; exp004 0,40). mAP eşikten bağımsızdır.

| Model | Eşik | mAP50 | mAP50-95 | P | R | FA | cisimde | zeminde | Kaç. |
|---|---|---|---|---|---|---|---|---|---|
| exp002_nano (dürüst) | 0,25 | 0,9943 | 0,7893 | 0,9565 | 0,9882 | 38 | 23 | 15 | 10 |
| exp004_nano_sizintili (sızıntılı) | 0,40 | 0,9950 | **0,9371** | **0,9988** | **1,0000** | **1** | 1 | 0 | **0** |
| **Fark (sızıntılı - dürüst)** | | +0,0007 | **+0,1478** | +0,0423 | +0,0118 | -37 | -22 | -15 | -10 |

**Eşleştirilmiş tablo: aynı cisim için iki modelin sonucu** (her model kendi eşiğiyle)

| Kırılım | Grup | Kutu | İkisi de buldu | Yalnız exp002 | Yalnız exp004 | İkisi de kaçırdı |
|---|---|---|---|---|---|---|
| tümü | tümü | 845 | 835 | 0 | 10 | 0 |
| ışık | Kuru-aydınlık (Dry-Bright) | 105 | 102 | 0 | 3 | 0 |
| ışık | Kuru-karanlık (Dry-Dark) | 418 | 411 | 0 | 7 | 0 |
| ışık | Kuru-loş (Dry-Dim) | 121 | 121 | 0 | 0 | 0 |
| ışık | Islak-loş (Wet-Dim) | 201 | 201 | 0 | 0 | 0 |
| boyut | <16 px (az veri, 2 parça) | 39 | 31 | 0 | 8 | 0 |
| boyut | 16-32 px | 90 | 88 | 0 | 2 | 0 |
| boyut | 32-64 px | 139 | 139 | 0 | 0 | 0 |
| boyut | 64-96 px | 171 | 171 | 0 | 0 | 0 |
| boyut | ≥96 px | 406 | 406 | 0 | 0 | 0 |

exp002'nin kaçırdığı 10 cismin hepsini exp004 buldu; exp004'ün bulup exp002'nin kaçırdığı cisim yok. Bu 10 cismin 8'i 16 pikselden küçük.

**Kırılımlar** (üst satır exp002 dürüst, alt satır exp004 sızıntılı)

| Grup | Kare | Parça | Model | mAP50 | mAP50-95 | P | R | FA | cisimde | zeminde | Kaç. | Not |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Bizim bölmede eğitimde var | 776 | 17 | dürüst | 0,9942 | 0,7911 | 0,9575 | 0,9871 | 34 | 20 | 14 | 10 | |
| | | | sızıntılı | 0,9950 | 0,9376 | 0,9987 | 1,0000 | 1 | 1 | 0 | 0 | |
| Bizim bölmede eğitimde yok (Screw, Tape) | 69 | 2 | dürüst | 0,9950 | 0,7726 | 0,9452 | 1,0000 | 4 | 3 | 1 | 0 | az veri |
| | | | sızıntılı | 0,9950 | 0,9289 | 1,0000 | 1,0000 | 0 | 0 | 0 | 0 | az veri |
| Kuru-aydınlık | 105 | 3 | dürüst | 0,9899 | 0,4087 | 0,9623 | 0,9714 | 4 | 4 | 0 | 3 | |
| | | | sızıntılı | 0,9950 | 0,8971 | 0,9906 | 1,0000 | 1 | 1 | 0 | 0 | |
| Kuru-karanlık | 418 | 8 | dürüst | 0,9940 | 0,8163 | 0,9384 | 0,9833 | 27 | 13 | 14 | 7 | |
| | | | sızıntılı | 0,9950 | 0,9297 | 1,0000 | 1,0000 | 0 | 0 | 0 | 0 | |
| Kuru-loş | 121 | 3 | dürüst | 0,9950 | 0,8436 | 0,9528 | 1,0000 | 6 | 6 | 0 | 0 | |
| | | | sızıntılı | 0,9950 | 0,9609 | 1,0000 | 1,0000 | 0 | 0 | 0 | 0 | |
| Islak-loş | 201 | 5 | dürüst | 0,9948 | 0,8924 | 0,9950 | 1,0000 | 1 | 0 | 1 | 0 | |
| | | | sızıntılı | 0,9950 | 0,9528 | 1,0000 | 1,0000 | 0 | 0 | 0 | 0 | |
| <16 px | 39 | 2 | dürüst | 0,8685 | 0,2532 | 0,7750 | 0,7949 | 9 | 6 | 3 | 8 | az veri |
| | | | sızıntılı | 0,9950 | 0,5688 | 0,9750 | 1,0000 | 1 | 1 | 0 | 0 | az veri |
| 16-32 px | 90 | 7 | dürüst | 0,9939 | 0,4569 | 0,9888 | 0,9778 | 1 | 1 | 0 | 2 | |
| | | | sızıntılı | 0,9950 | 0,8016 | 1,0000 | 1,0000 | 0 | 0 | 0 | 0 | |
| 32-64 px | 139 | 13 | dürüst | 0,9950 | 0,7815 | 0,9653 | 1,0000 | 5 | 5 | 0 | 0 | |
| | | | sızıntılı | 0,9950 | 0,9440 | 1,0000 | 1,0000 | 0 | 0 | 0 | 0 | |
| 64-96 px | 171 | 14 | dürüst | 0,9950 | 0,7867 | 0,9828 | 1,0000 | 3 | 2 | 1 | 0 | |
| | | | sızıntılı | 0,9950 | 0,9528 | 1,0000 | 1,0000 | 0 | 0 | 0 | 0 | |
| ≥96 px | 406 | 16 | dürüst | 0,9950 | 0,8941 | 0,9531 | 1,0000 | 20 | 9 | 11 | 0 | |
| | | | sızıntılı | 0,9950 | 0,9771 | 1,0000 | 1,0000 | 0 | 0 | 0 | 0 | |

**Tutarlılık kontrolü: geçti.** exp002'nin ortak karelerdeki hataları (38 yanlış alarm, 10 kaçırma), bütün test'teki hatalarından (175, 73) az.

### 2. Raporlanan puan farkı
Her model kendi test setinde. İki test seti farklı karelerden oluşuyor; bu farkın tamamı sızıntı değildir.

| Model | Test seti | Eşik | mAP50 | mAP50-95 | P | R | FA | cisimde | zeminde | Kaç. |
|---|---|---|---|---|---|---|---|---|---|---|
| exp004 (sızıntılı) | hazır test: 8.429 kare, 8.613 kutu, 116 parça | 0,40 | 0,9948 | 0,9045 | 0,9963 | 0,9898 | 32 | 30 | 2 | 88 |
| exp002 (dürüst) | bizim test: 3.439 kare, 19 parça | 0,25 | 0,9932 | 0,7811 | 0,9506 | 0,9788 | 175 | 107 | 68 | 73 |
| **Ağırlıklı ortalama, kaba tahmin**: exp004 | | | 0,9925 | 0,9024 | 0,9964 | 0,9901 | - | - | - | - |
| **Ağırlıklı ortalama, kaba tahmin**: exp002 | | | 0,9771 | 0,6192 | 0,9468 | 0,9579 | - | - | - | - |

Ağırlıklı ortalamalar iki model için aynı ağırlıklarla (exp002'ninkiler: aydınlık %50,1, ıslak-loş %21,4, loş %15,5, karanlık %13,0) hesaplandı. exp004'ün kendi dosyasındaki ağırlıklar çok az farklı (aydınlık %50,2), çünkü sızıntılı veri klasöründe kullanılmayan 70 görüntü yok; o ağırlıklarla sonuç 0,9925 / 0,9023 / 0,9963 / 0,9901, yani aynı.

Hazır test'te ışık koşulu dağılımı bütün veriye benziyor (aydınlık 4.286 kare, %51). exp004'ün hazır test'teki en zayıf satırları da bizim testtekilerle aynı yerde: kuru-aydınlık (mAP50-95 0,857) ve 16 pikselden küçük kutular (R 0,928, mAP50-95 0,654).

### Sonuçların özeti (yorum kurallarına göre)
- **Sızıntı puanı şişiriyor.** Aynı 845 karede sızıntılı model dürüst modelden her ölçütte daha iyi: mAP50-95 0,789'dan 0,937'ye (+0,148), yanlış alarm 38'den 1'e, kaçırma 10'dan 0'a. Eşleştirilmiş tabloda tersi yönde tek bir cisim bile yok (835 / 0 / 10 / 0). Yorum kuralına göre bu fark "sızıntının puanı şişirmesi" olarak yorumlanır.
- **mAP50 neredeyse aynı** (0,994 ve 0,995). Sızıntının asıl etkisi, kutunun yerini tam bulmada (mAP50-95) ve yanlış alarmlarda görülüyor.
- **Fark en çok zor durumlarda büyük:** kuru-aydınlıkta mAP50-95 0,409'dan 0,897'ye, 16 pikselden küçük kutularda 0,253'ten 0,569'a (az veri). Dürüst testte en zayıf çıkan yerler, sızıntılı modelde neredeyse sorunsuz görünüyor.
- **Bir makalede raporlanacak puan ile dürüst puan arasındaki fark:** mAP50-95 0,905'e karşı 0,781; ağırlıklı ortalamada 0,902'ye karşı 0,619. Yanlış alarm oranı da çok farklı (hazır test'te 8.613 kutuda 32, bizim test'te 3.439 kutuda 175).
- **Sonuç ne çıkarsa çıksın nihai model, eşik ve test sonuçları değişmez** (plan gereği). exp004 başka hiçbir amaçla kullanılmayacak.

### Farkı etkileyebilecek başka nedenler
Sonuçlar görülmeden yazılanlar:
- exp004'ün eğitim verisi daha az: 22.765 kare, exp002'de 26.906. Bu sızıntılı modelin aleyhinedir, yine de daha iyi çıktı.
- exp004, bizim test'te "eğitimde yok" sayılan Screw ve Tape'i eğitimde gördü; bu da sızıntının bir parçası.
- exp002'nin yeniden çalıştırılmasında ilk test'e göre çok küçük sayısal farklar olabilir. Ortak karelerdeki ayrı bir sonuç ilk test'te olmadığı için bu doğrudan karşılaştırılamaz; tutarlılık kontrolü geçti.

Sonuçlar görüldükten sonra fark edilen:
- exp004 erken durmadı ve 50 epoch eğitildi; exp002'nin en iyi epoch'u 16 idi. Daha uzun eğitim farkın bir kısmını açıklayabilir. Bunu ayırmak için ayrı bir deney gerekir (ör. exp002'yi 50 epoch eğitmek); planlanmadı.
- İki modelin eşiği farklı (0,25 ve 0,40). Plan gereği her biri kendi val eşiğiyle ölçüldü; P, R ve hata sayıları eşiğe bağlıdır, mAP bağlı değildir.

## Telefon kamerasıyla gerçek dünya testi (plan ve ölçümler, 18 Kasım 2026, çekimden önce yazıldı)
Bu bölüm hiçbir fotoğraf çekilmeden ve model hiçbir telefon fotoğrafını görmeden yazıldı. Amaç: FOD-A'da eğitilen modelin gerçek asfalt üzerinde, telefonla çekilmiş fotoğraflarda ne kadar başarılı olduğunu ölçmek. Gerçek bir pist değil, trafiğe kapalı boş bir asfalt alan (otopark, okul bahçesi) kullanılır.

**Sabit olanlar:** model `exp002_nano` (nihai model), güven eşiği 0,25 (val'de F2 kuralıyla seçildi). Telefon fotoğraflarına bakılarak eşik ya da model değiştirilmez.

### İki ayrı çekim
| Set | Fotoğraf | Ne için | Model çalıştırılabilir mi |
|---|---|---|---|
| **Deneme (pilot)** | 30: 20 cisimli + 10 boş | Görüntü işleme yöntemini seçmek, çekim düzenini denemek | Evet, serbestçe |
| **Test** | 300: 200 cisimli + 100 boş | Asıl sonuç | Yalnızca bir kez, etiketler commit edildikten sonra |

Deneme seti test'ten farklı bir günde çekilir ve test'e hiçbir zaman girmez. Deneme setinde mümkün olduğunca her mesafeden ve her ışıktan fotoğraf olur.

### Görüntü işleme yöntemi (denemede seçilir)
Model 300×300 yakın plan karelerle eğitildi; telefon fotoğrafı yaklaşık 4000×3000 piksel. Fotoğraf olduğu gibi küçültülürse cisim birkaç piksele iner. Dört yöntem denenir:

| Yöntem | Ne yapılır |
|---|---|
| `320` | Fotoğrafın uzun kenarı 320'ye küçültülür, model bir kez çalışır |
| `640` | Uzun kenar 640'a küçültülür |
| `1280` | Uzun kenar 1280'e küçültülür |
| `parca` (tiling) | Uzun kenar 1920'den büyükse 1920'ye küçültülür. Fotoğraf 320×320 parçalara bölünür, komşu parçalar 80 piksel (%25) örtüşür; son parça kenara hizalanır. Model her parçada 320 boyutunda çalışır. Kutular fotoğrafın koordinatına taşınır ve birleştirilir: önce aynı cisme düşen tekrar kutular (örtüşme IoU ≥ 0,5) ayıklanır, sonra alanının en az %80'i daha emin bir kutunun içinde kalan parça kutular ayıklanır. |

- `parca` için 1920 seçildi, çünkü demo videosunun genişliği (Full HD) 1920 piksel. Bu ölçekte yakından çekilmiş bir cıvata yaklaşık 50-140 piksel olur; eğitimdeki cisimler de bu boyutlardaydı.
- Bu ayarlar (1920, 320, 80, 0,5, %80) denemede değiştirilmez; yalnızca dört yöntemden biri seçilir.
- **Seçim kuralı:** dört yöntem deneme setinin tamamında (30 fotoğraf, boşlar dahil), eşik 0,25 ile ölçülür. **F2'si en yüksek olan seçilir.** F2 üç ondalığa yuvarlanınca eşitse **daha hızlı olan** seçilir.
- **Hız:** her yöntem için fotoğraf başına süre (ortanca, ms) denemede yalnızca işlemciyle ölçülür. Süreye küçültme, parçalara bölme, model ve birleştirme dahil, diskten okuma hariç; ilk 3 fotoğraf ısınma sayılır. Dört yöntemin süresi de raporlanır. **`parca` seçilirse, tek seferlik yöntemlere göre ne kadar yavaş olduğu ayrıca not edilir.**
- Seçilen yöntem `results/telefon_pilot/secim.json` dosyasına yazılır. Test bu dosyadaki yöntemle yapılır ve değiştirilemez.

### Test setinin çekim düzeni
**Cisimli 200 fotoğraf:** 3 mesafe × 3 ışık = 9 grup, her grupta yaklaşık 22 fotoğraf.

| Mesafe | Nasıl |
|---|---|
| Yakın (~50 cm) | Çömelerek, telefon yere paralel, tam yukarıdan |
| Orta (~1 m) | Ayakta, telefon göğüs hizasında, aşağı bakarak |
| Uzak (~2-3 m) | Ayakta, eğik açıyla ileri bakarak; ufuk kadraja girmez |

| Işık | Ne zaman |
|---|---|
| Aydınlık | Güneşli gündüz |
| Loş | Bulutlu hava ya da gölge |
| Akşam | Gün batımından sonraki ilk 30 dakika, flaş kapalı |
| Islak (isteğe bağlı) | Yağmur yağarsa ek grup |

- **Cisimler:** vida, somun, pul, cıvata, taş. Her fotoğrafta 1-3 cisim; beş tür toplamda yaklaşık eşit sayıda (her biri yaklaşık 70 kutu). Modelin eğitiminde **vida (Screw) hiç yoktu**; vida ayrıca raporlanır.
- **Boş 100 fotoğraf** (hiç cisim yok), aynı ışık ve mesafelerde karışık: 50 temiz zemin, 50 "kafa karıştırıcı" zemin (boya çizgisi, çatlak, yağ lekesi, yaprak, rögar kapağı). FOD-A'da boş kare olmadığı için boş zemindeki yanlış alarm ilk kez burada ölçülür.
- **Çekim kuralları:**
  - Alan çekimden önce süpürülür.
  - Telefonun kendi kamerası kullanılır: 1x lens, yakınlaştırma yok, portre modu yok, filtre yok.
  - Biçim JPEG olur (iPhone'da *Ayarlar → Kamera → Biçimler → En Uyumlu*).
  - Fotoğraflar yatay çekilir.
- **Çekim defteri:** `telefon/cekim_defteri_sablon.csv` şablonuyla, her set için ayrı (Google E-Tablolar'da doldurulup CSV olarak indirilir). Sütunlar:

  | Sütun | Ne yazılır |
  |---|---|
  | `dosya_adi` | Fotoğrafın dosya adı (ör. `IMG_1234.jpg`) |
  | `tarih` | Çekim günü (ör. 2026-11-22) |
  | `isik` | `aydinlik`, `los`, `aksam` ya da `islak` |
  | `mesafe` | `yakin`, `orta` ya da `uzak` |
  | `zemin` | `temiz` ya da `karistirici` |
  | `karistirici_tur` | Zemin karıştırıcıysa ne olduğu (ör. `cizgi`, `catlak`, `leke`, `yaprak`, `rogar`) |
  | `vida`, `somun`, `pul`, `civata`, `tas` | Fotoğrafa bırakılan cisim sayısı (yoksa 0 ya da boş) |
  | `diger` | Bırakılmadığı halde görülen, 1 cm'den büyük gevşek cisim sayısı |
  | `not` | İsteğe bağlı |

### Etiketleme ve modelden bağımsızlık
- **Araç:** [Make Sense](https://www.makesense.ai): ücretsiz, tarayıcıda çalışıyor, kurulum ve hesap gerekmiyor, YOLO formatında dışa aktarıyor. Kendi açıklamasına göre fotoğraflar sunucuya yüklenmiyor.
- **Sınıflar:** `telefon/siniflar.txt` içindeki sırayla: vida, somun, pul, civata, tas, diger. Değerlendirmede hepsi tek "FOD" sınıfı sayılır; tür yalnızca kırılım içindir.
- **Kutular:** cismin kenarlarına sıkı çizilir. Bırakılan cisimlere ek olarak, görülen 1 cm'den büyük her gevşek cisim `diger` olarak etiketlenir.
- **Bağımsızlık kuralları:**
  1. Test fotoğraflarında model hiç çalıştırılmadan etiketleme yapılır.
  2. Script, her fotoğrafta her türün kutu sayısını çekim defteriyle karşılaştırır. Farklıysa durur ve fotoğrafı söyler; hata o aşamada düzeltilir.
  3. Etiket dosyaları (küçük txt dosyaları) ve çekim defteri `telefon/<set>/` altında **repoya commit edilir**. Değerlendirme etiketleri repodan okur ve bu dosyalar commit edilmemiş ya da commit'ten sonra değiştirilmişse çalışmaz.
  4. Test bir kez yapılır: `--final-test` gerekir, defterde `IS = "telefon_test"` ile birlikte `TEST_ONAY = "evet"` gerekir, kayıt `results/telefon_test_log.csv` içine yazılır ve ikinci kez çalışmaz.
  5. Sonuç görüldükten sonra etiket değiştirilmez. Etiket hatası bulunursa asıl sonuç olduğu gibi kalır; düzeltilmiş sayı ayrıca ve "sonradan" notuyla verilir.

### Raporlanacak ölçümler (test, bir kez)
Hepsi, iyi ya da kötü, raporlanır. "Geçti/kaldı" sınırı yok. Eşleşme kuralı FOD-A ile aynı: model kutusu gerçek kutuyla IoU ≥ 0,5 örtüşürse doğru; yanlış alarm gerçek bir kutuya 16 pikselden yakınsa "cisimde", değilse "zeminde" (pikseller modele giren ölçekte).

**Cisimli fotoğraflar:**
- Asıl ölçütler: R, P, F2, kaçırılan cisim, yanlış alarm (cisimde / zeminde).
- mAP50 ve mAP50-95 de verilir. mAP kendi kodumuzla hesaplanır, Ultralytics'in hesabıyla birebir aynı yöntem değildir. Elle çizilen kutular FOD-A kadar hassas olmadığı için mAP50-95 dikkatle okunur.
- Fotoğraf düzeyinde: bütün cisimleri bulunan fotoğraf oranı, en az bir cismi bulunan fotoğraf oranı.

**Boş fotoğraflar:** en az bir yanlış alarm olan fotoğraf sayısı ve oranı; fotoğraf başına yanlış alarm.

**Bütün fotoğraflar (cisimli + boş):** P, R, F2 (denemedeki seçim ölçütüyle aynı).

**Kırılımlar:**
- Mesafe ve ışık: cisimli fotoğraflarda bütün ölçütler, boş fotoğraflarda yanlış alarm ölçütleri.
- Cisim türü (vida ayrı: eğitimde yok) ve kutu boyutu (uzun kenar, modele giren ölçekte: <16, 16-32, 32-64, 64-96, ≥96 px): yalnızca R ve kaçırılan cisim. Bir yanlış alarmın türü ve boyutu olmadığı için P bu satırlarda verilmez.
- Boş zemin türü: temiz / karıştırıcı.
- 20'den az kutuya (boşlarda 20'den az fotoğrafa) dayanan satır "az veri" diye işaretlenir.

**Karşılaştırma:** FOD-A test sonucuyla (nano: R 0,979, P 0,951) yan yana. Düşüş bekleniyor; fark sayı olarak verilir.

**Hata görüntüleri:** yalnızca test'in asıl sayıları repoya commit edildikten sonra çıkarılabilir (`IS = "telefon_hata"`). Rapor ve sunum için örnek olarak kullanılır. Bunlara bakılarak yapılan her değişiklik (eşik, yöntem, yeniden eğitim) yeni sürüm sayılır ve **yeni çekilen fotoğraflarla** ölçülür; bu test setinde ölçülmüş sayılmaz.

**Hız** test fotoğraflarıyla ölçülmez (denemedeki yöntem karşılaştırması dışında).

### Gizlilik
- Kamera yere bakar; ufuk, bina, araç kadraja girmez. Alanda insan ve araç yokken çekilir. Okul bahçesi ya da otopark için önceden izin alınır.
- Telefonda konum etiketi kapatılır (Android: kamera ayarlarında "Konum etiketi"; iPhone: *Ayarlar → Gizlilik → Konum Servisleri → Kamera → Hiçbir Zaman*).
- Yüklemeden önce her fotoğrafa bakılır. Yüz, kişi, plaka, ev numarası ya da yazı varsa fotoğraf silinir ve yeniden çekilir.
- **Fotoğraflar repoya girmez** (repo herkese açık); yalnızca Drive'da durur (`fod_proje/telefon/<set>/foto/`). `.gitignore` `telefon/` altındaki görüntü dosyalarını engeller. Repoya yalnızca etiketler, çekim defteri ve sonuçlar girer.
- Kod: `scripts/phone_eval.py`; defterde `IS = "telefon_pilot"`, `"telefon_test"` (`TEST_ONAY = "evet"` ile) ya da `"telefon_hata"`.
- Script fotoğrafları yönüne göre çevirip okur. Hata görüntülerini yeniden çizerek kaydettiği için içlerinde fotoğrafın ek bilgisi (EXIF, konum dahil) bulunmaz.

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
6. **Hazır (sızıntılı) ayrımla karşılaştırma:** nano ve small bittikten sonra, hazır `trainval.txt` / `test.txt` ayrımıyla da bir eğitim yapılacak. Amaç, sızıntının puanı ne kadar şişirdiğini göstermek. Plan ve yorum kuralları: "Sızıntılı ayrım karşılaştırması" bölümü (defterde `IS = "sizinti"`). **Bitti** (18 Kasım 2026); bkz. "Sızıntılı ayrım sonuçları".
7. **Telefon kamerasıyla gerçek dünya testi ve demo** (nano ve small eğitimlerinden sonra). Projede özel kamera yok; test ve demo telefon kamerasıyla yapılacak. Test planı ve ölçümler: "Telefon kamerasıyla gerçek dünya testi" bölümü (çekimden önce yazıldı).
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
