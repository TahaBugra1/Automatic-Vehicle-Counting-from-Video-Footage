# Gereksinimler / Requirements

Proje: Video Görüntülerinden Otomatik Araç Sayımı (Grup 31 - VisionFlow)

## Fonksiyonel Gereksinimler (FR)

| No | Gereksinim | Öncelik |
|---|---|---|
| FR-1 | Kullanıcı 1–5 trafik videosu yükleyebilmeli (izin verilen uzantılar: MP4 vb.). | Zorunlu |
| FR-2 | Sistem dosya türünü ve boyutunu (en çok 500 MB) denetlemeli, geçersiz dosyayı anlaşılır bir hata ile reddetmeli. | Zorunlu |
| FR-3 | İlk videonun önizleme karesi gösterilmeli; kullanıcı sanal sayım çizgisini sürükleyerek belirleyebilmeli. | Zorunlu |
| FR-4 | Çizgi koordinatları çözünürlükten bağımsız (0–1 aralığında) saklanmalı. | Zorunlu |
| FR-5 | Sistem her karede otomobil, otobüs, kamyon ve motosikleti tespit etmeli (YOLO11, COCO sınıfları). | Zorunlu |
| FR-6 | Tespitler ByteTrack ile kareler boyunca takip kimliğine bağlanmalı. | Zorunlu |
| FR-7 | Kutunun alt orta noktası çizgiyi geçtiğinde araç, takip kimliği başına yalnızca bir kez ve geçiş yönüne göre sayılmalı. | Zorunlu |
| FR-8 | Çizgi çevresindeki titreşim ölü bant ile bastırılmalı. | Zorunlu |
| FR-9 | Kullanıcı model boyutunu ve güven eşiğini seçebilmeli. | Önemli |
| FR-10 | Birden fazla video aynı işlemde sırayla işlenmeli; bir videodaki hata diğerlerini etkilememeli. | Zorunlu |
| FR-11 | Çıktı olarak tespit kutuları, çizgi ve sayaçlar çizilmiş, tarayıcıda oynatılabilir H.264 MP4 üretilmeli. | Zorunlu |
| FR-12 | CSV çıktısında geçiş karesi, zaman, takip kimliği, araç sınıfı, yön ve özet metrikler yer almalı. | Zorunlu |
| FR-13 | Her video için sonuç kartı gösterilmeli (iki yönlü sayaç, sınıf bazlı toplam, motosiklet dahil); video ve CSV indirilebilmeli. | Zorunlu |
| FR-14 | Sistem sayımı UA-DETRAC gerçek etiketleriyle karşılaştırıp video bazlı hata, toplam sayım hatası ve FPS raporlayabilmeli. | Önemli |

## Fonksiyonel Olmayan Gereksinimler (NFR)

| No | Kategori | Gereksinim |
|---|---|---|
| NFR-1 | Doğruluk | UA-DETRAC videolarında toplam sayım hatası ≤ %10 (otomobil, otobüs, kamyon için). İlk ölçümden sonra gözden geçirilir. Motosiklet: desteklenir, benchmark ile ölçülmez; elle sayılmış 2–3 kısa videoyla küçük doğrulama yapılır. |
| NFR-2 | Performans | Minimum donanım: CPU üzerinde yolo11n. Önerilen donanım: ≥ 4 GB VRAM'li NVIDIA GPU. Geliştirme makinesi: RTX 4060 (8 GB). FPS hedefi, ekip makinelerinde yapılacak ilk ölçümden sonra kesinleştirilecek (taslak: yolo11n/GPU ≥ 30 FPS, yolo11n/CPU ≥ 5 FPS). |
| NFR-3 | Tekrar üretilebilirlik | Aynı video, model ve ayarlarla aynı CSV sonucu üretilmeli. |
| NFR-4 | Kullanılabilirlik | Çizgi ayarı ve indirme tarayıcıdan eğitim gerektirmeden yapılabilmeli. Arayüz Türkçe/İngilizce dil seçeneği sunmalı; hatalar anlaşılır mesajlarla verilmeli. |
| NFR-5 | Uyumluluk | Çıktı video güncel tarayıcılarda oynatılabilmeli (H.264). Sistem Windows ve Linux'ta çalışmalı. |
| NFR-6 | Güvenilirlik | Geçersiz girdi (uzantı, boyut, çizgi koordinatı) servisi çökertmemeli; bozuk bir video toplu işlemi durdurmamalı. |
| NFR-7 | Güvenlik | Yüklenen dosyalar doğrulanmalı, dosya adları güvenli işlenmeli, geçici dosyalar iş bitince temizlenmeli. |
| NFR-8 | Bakım ve test | Çizgi geçiş mantığı otomatik testlerle doğrulanmalı (kod kapsamı ≥ %90). |
| NFR-9 | Taşınabilirlik | Kurulum ve kullanım adımları README'de belgelenmeli; bağımlılıklar sabit sürümlerle listelenmeli. |
| NFR-10 | Gizlilik | Videolar yerelde işlenmeli, üçüncü taraflara gönderilmemeli. |

## Kullanıcı Rolleri ve Yetkileri

- Tek rol vardır: **Kullanıcı**. Giriş yapmadan video yükler, sayım çizgisini ayarlar, işlemi başlatır, sonuç videosunu ve CSV'yi indirir.
- Yönetici rolü, kullanıcı hesabı ve yetkilendirme MVP'de yoktur. Sistem tek kullanıcılı, yerel çalışan bir araçtır.

## MVP Kapsamı

MVP'ye dahil:
- FR-1 … FR-13 (yükleme, çizgi ayarı, tespit, takip, sayım, video ve CSV çıktısı, sonuç kartı)
- FR-9 (model ve güven eşiği seçimi)
- FR-14, **değerlendirme betiği olarak**: UA-DETRAC etiketleriyle video bazlı ve toplam sayım hatası ile FPS ölçümü. Arayüzde karşılaştırma ekranı MVP'de yoktur.
- NFR-1 … NFR-10

Sonraki sürüme bırakılanlar:
- Canlı kamera akışı
- Arayüzde doğruluk karşılaştırma ekranı
- Birden fazla sayım çizgisi
- Bisiklet sayımı
- Kullanıcı hesabı ve yetkilendirme
- Motosiklet için benchmark doğrulaması (UA-DETRAC'ta motosiklet etiketi yok)

## Kapsam Dışı

- Canlı kamera akışı (ileride yapılacak işler listesinde)
- Hareketli kamera
- Plaka tanıma
- Bisiklet sayımı
- Kullanıcı hesabı ve oturum yönetimi

## Açık Noktalar

- NFR-2 FPS değerleri ilk ölçümden sonra güncellenecek.
- Diğer iki ekip üyesinin donanımı henüz doğrulanmadı.
