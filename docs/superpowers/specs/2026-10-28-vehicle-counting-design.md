# Video Görüntülerinden Otomatik Araç Sayımı: Tasarım

Tarih: 2026-10-28
Durum: Onaylandı (2026-10-28)
İlgili belge: `docs/requirements.md` (fonksiyonel ve fonksiyonel olmayan gereksinimler)

## 1. Amaç ve neden

Sabit kameralı trafik videolarından geçen araçları otomatik sayan, yerelde çalışan bir web uygulaması.
Asıl kullanım: birinin gerçek trafik kamerası kaydını yükleyip işine yarayan, güvenilir sayımlar alması.
Ders projesi sunumu ve portfolyo yan etkidir; birincil hedef değildir.

Başarı ölçütü **sayım doğruluğudur**. Hız ve kurulum kolaylığı doğruluktan sonra gelir.

## 2. Hedefler, hedef olmayanlar, başarısızlık sayılanlar

### Hedefler
- Video yükle (1-5 dosya), sayım çizgisini önizleme üzerinde çiz, sayımı başlat.
- Araçları çizgiyi geçiş yönüne göre say: otomobil, otobüs, kamyon, motosiklet.
- Sonuç olarak: toplam ve yön sayıları, dakikaya göre dağılım, sınıfa göre dağılım, şüpheli geçişler.
- İşaretli MP4 ve ayrıntılı CSV indirilebilsin.
- Arayüz Türkçe ve İngilizce olsun.
- Doğruluk UA-DETRAC etiketleriyle ölçülsün (değerlendirme betiği MVP'de).

### Hedef olmayanlar (MVP dışı)
Canlı kamera akışı, hareketli kamera, plaka tanıma, bisiklet sayımı, kullanıcı hesabı ve yetkilendirme,
birden fazla sayım çizgisi, arayüzde doğruluk karşılaştırma ekranı, motosiklet için benchmark doğrulaması
(UA-DETRAC'ta motosiklet etiketi yok).

### Başarısızlık sayılanlar (teknik olarak çalışsa bile)
- Aynı araç iki kez sayılırsa.
- Arayüz karmaşıksa (çizgiyi ayarlamak ya da sonucu bulmak zorsa).
- Doğruluğu gösteren hiçbir ölçüm sunulamazsa.
- Başka bir bilgisayarda kurulamıyorsa.

## 3. Kısıtlar

- Yerel çalışma. Videolar üçüncü tarafa gönderilmez.
- Geliştirme makinesi: RTX 4060 (8 GB). Minimum: CPU üzerinde yolo11n.
- Kod sıfırdan yazılacak. Şeyda'nın deposu (`seydamolla/Automatic-Vehicle-Counting-from-Video-Footage`) yalnızca referanstır; kod kopyalanmaz.
- Commit ve push'u kullanıcı manuel yapar.
- Gereksinimler: `docs/requirements.md` (FR-1…FR-14, NFR-1…NFR-10).

## 4. Kararlar

### Kullanıcının kararları
- Ana ekran **tek sayfa**: solda büyük video önizlemesi ve sürüklenebilir sayım çizgisi, sağda ayarlar (model, güven eşiği) ve "Sayımı başlat" düğmesi.
- Sonuç ekranında şüpheli geçişler **neden ve zamanla listelenir**; kare numarası değil videodaki zaman gösterilir.
- Sayılan ve işaretli geçişler toplam sayıya **dahildir**; "olası kaçırılan araç" **dahil değildir**. Toplamın yanında belirsizlik sade
  dille gösterilir: "222 araç (3 fazla, 4 eksik olabilir)". Kullanıcı bu iki yaklaşımı (kaybolan aracı sayıma katmamak, belirsizliği
  ayrıca göstermek) örnekle anlatıldıktan sonra onayladı.
- Güvenilirlik rozeti **işaretli geçişlerin oranına** göre belirlenir. Eşik yüzdeleri UA-DETRAC ölçümüyle ayarlanır; şimdilik sayı konmaz.
- **Her video için ayrı sayım çizgisi** vardır (videolar farklı kameralardan olabilir). Sol listeden bir video seçilir, önizleme ve
  çizgisi o videoya ait olur.
- Zamana göre dağılım ve güvenilirlik göstergesi MVP'dedir.
- Arayüz dilleri: Türkçe ve İngilizce.

### Yazarın kararları (kullanıcı onaylamadı, itiraz edilebilir)
- Teknoloji: Python, FastAPI, YOLO11, ByteTrack, sade HTML/JS arayüz.
- Zamana göre dağılım varsayılan olarak dakikalık aralıklarla.
- Şüpheli geçişler varsayılan olarak kapalı bir bölümde gösterilir.
- Şüpheli geçiş türleri aşağıdaki tablodaki gibi (kullanıcı listeyi sorguladı, bu tablo revize hâlidir).

### Şüpheli geçiş türleri

| Neden | Kullanıcıya görünen metin |
|---|---|
| Çizgiye yaklaşırken kaybolan ve sayılmayan takip | Olası kaçırılan araç |
| Yakınında kısa süre önce kaybolan bir takip vardı | Olası çift sayım |
| Çizgi çevresinde kısa sürede çok yön değiştirdi | Kararsız geçiş |
| Takip çok kısa sürdü ya da ortalama tespit güveni düşük | Zayıf tespit |
| Takip boyunca sınıf değişti (ör. otomobil ↔ kamyon) | Sınıf belirsiz |
| Başka bir kutuyla yoğun örtüşme | Kalabalık geçiş |

Not: "Kimlik değişti" doğrudan gözlemlenemez; yalnızca "olası çift sayım" olarak dolaylı çıkarılır.

### İnceleme sonrası netleştirilenler (yazarın kararı, itiraz edilebilir)

- **Olası hata aralığı** istatistiksel bir sınır değildir; işaretli geçişlerin sayısıdır ve asimetriktir:
  "222 araç (a fazla, b eksik olabilir)". Burada **a (fazla)** "olası çift sayım", "kararsız geçiş", "zayıf tespit" ve "kalabalık geçiş" işaretlerinden gelir
  (bunlar sayıya dahildir ve fazla olabilir); **b (eksik)** "olası kaçırılan araç" sayısıdır (bunlar toplama **dahil değildir**,
  eksik olabilir). "Sınıf belirsiz" toplam sayıyı değil sınıf dağılımını etkiler, aralığa girmez.
- **Kaçırılan araç** toplama dahil edilmez. CSV'de bunlar ayrı bir "olası kaçırılan" bölümünde/etiketinde yer alır; geçiş karesi, sınıf ve yön
  alanları bilinmiyorsa boş bırakılır, yaklaşma yönü ve son görülen zaman yazılır.
- **Güvenilirlik rozeti** üç seviyelidir: yüksek, orta, düşük. Eşikler bir yapılandırma dosyasında tutulur; UA-DETRAC ölçümüne kadar
  yer tutucu değerlerle gelir ve bu açıkça belgelenir.
- **Yön adları:** sayım çizgisi yönlüdür; çizgi üzerinde bir ok gösterilir. İki yön "Yön 1" ve "Yön 2" olarak adlandırılır
  (okun gösterdiği tarafa geçiş Yön 1). Kullanıcı yönü tersine çevirebilir. CSV'de aynı adlar kullanılır.
- **İş modeli:** videolar sırayla işlenir, arayüzde video başına canlı ilerleme gösterilir. Sonuçlar diskte çıktı klasöründe kalır
  (sunucu yeniden başlasa da erişilebilir) ve kullanıcı silene kadar durur. Yüklenen kaynak videolar işlem bitince silinir.
- **Değerlendirme (UA-DETRAC):** gerçek geçişler, UA-DETRAC iz etiketlerinden (araç sınırlayıcı kutuları ve kimlikleri) sayım çizgisine göre
  türetilir. Her UA-DETRAC dizisi için sayım çizgisi, bir yapılandırma dosyasında önceden tanımlanır ve test videoları
  değerlendirilmeden önce dondurulur. Aynı çizgi hem sistem sayımında hem gerçek geçişlerin türetilmesinde kullanılır.
  İşaret kalitesi, işaretli geçişlerin türetilmiş gerçek geçişlerle karşılaştırılmasıyla ölçülür.

## 5. Arayüz

### Ana ekran (tek sayfa)
Solda seçili videonun önizlemesi ve sürüklenebilir çizgisi, yüklenen videoların listesi (her videonun çizgisi ayrıdır; çizgisi ayarlanmamış
video listede işaretlenir ve sayım başlatılmadan önce uyarı verilir). Sağda: video ekleme, model seçimi,
güven eşiği, "Sayımı başlat". Dil seçici (TR/EN) üstte.

### Sonuç ekranı (her video için bir kart)
- Yön sayıları ve toplam (olası hata aralığıyla).
- Güvenilirlik rozeti (işaretli geçiş oranına göre).
- Dakikaya göre sayım grafiği.
- Sınıf dağılımı.
- Şüpheli geçişler: kapalı bölüm; her satırda zaman, neden metni; tıklayınca işaretli videoda o ana atlama.
- Video indir, CSV indir.

## 6. Doğrulama

- Çizgi geçiş mantığı birim testleriyle doğrulanır (hedef kod kapsamı ≥ %90).
- UA-DETRAC değerlendirme betiği: video bazlı ve toplam sayım hatası, FPS.
- Şüpheli geçiş işaretlerinin kalitesi de ölçülür: işaretli geçişlerin ne kadarı gerçekten hatalıydı.
- Motosiklet için, elle sayılmış 2-3 kısa kendi videomuzla küçük bir doğrulama yapılır.

## 7. Builder'a bırakılanlar

Klasör yapısı ve modül sınırları, iş kuyruğu/arka plan işleme yaklaşımı, API şekilleri, olası hata aralığının
tam hesaplanma yöntemi, grafiğin çizim kütüphanesi (veya elle SVG), şüpheli geçiş eşik değerleri,
test düzeni ve geliştirme sırası.

## 8. Açık noktalar

- NFR-2 FPS hedefleri ilk ölçümden sonra kesinleşecek.
- Rozet eşikleri UA-DETRAC ölçümüyle belirlenecek.
- `docs/requirements.md` bu belgedeki kararlarla henüz güncellenmedi (zamana göre dağılım, güvenilirlik göstergesi, TR/EN, olası hata aralığı,
  video başına çizgi; FR-3 "ilk videonun önizlemesi" ifadesi artık geçerli değil).
