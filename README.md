# Automatic-Vehicle-Counting-from-Video-Footage
Bilgisayar Mühendisliği Tasarım Dersi Projesi

Video kayıtlarındaki araçları YOLO11 (tespit) ve ByteTrack (takip) ile otomatik sayan, yerel çalışan bir web uygulaması (Python paketi: `visionflow`). Her video için kullanıcı bir sayım çizgisi çizer; çizgiyi geçen araçlar yöne ve araç sınıfına (otomobil, otobüs, kamyon, motosiklet) göre sayılır. Çıktı: çizgili sonuç videosu, CSV ve şüpheli geçişlerin listesi.

## Gereksinimler

- Python 3.11 veya üstü. Birincil platform Windows.
- Ekran kartı isteğe bağlıdır. Asgari: CPU ile `yolo11n` (yavaş). Önerilen donanım: en az 4 GB VRAM'li NVIDIA ekran kartı; geliştirme makinesi RTX 4060 8 GB. **Ancak ekran kartı yalnızca CUDA destekli bir `torch` kurulduktan sonra kullanılır**; varsayılan kurulum CPU-only torch getirir ve her şey CPU'da çalışır (özellikle `yolo11m` için çok yavaş). Bkz. "GPU kullanımı".

## Hızlı başlangıç (Windows)

```powershell
.\start.ps1
```

Betik önce Python 3.11+ olup olmadığını denetler (`py -3`, sonra `python`), `.venv` yoksa oluşturur ve sunucuyu başlatır. Bağımlılıkları (`pip install --upgrade pip` ve `pip install -e .`) **yalnızca** `.venv` yeni oluşturulduysa veya `pyproject.toml` son başarılı kurulumdan beri değiştiyse kurar (karşılaştırma için `.venv\visionflow-install.sha256` dosyasında bir özet tutulur); sonraki başlatmalar internet olmadan da çalışır. Bu özet dosyası olmayan eski bir `.venv` ilk başlatmada bir kez daha kurulum yapar. Sonra tarayıcıda <http://127.0.0.1:8000> adresini açın. Betik çalıştırma izni vermezse: `powershell -ExecutionPolicy Bypass -File .\start.ps1`. İlk kurulum yüzlerce MB indirir (torch dahil).

## Elle kurulum

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m uvicorn visionflow.api:create_app --factory --host 127.0.0.1 --port 8000
```

Sunucuyu yalnızca `127.0.0.1` üzerinde çalıştırın; uygulama kimlik doğrulaması içermez ve Host/Origin denetimi yalnızca yerel adresleri (`127.0.0.1`, `localhost`, `[::1]`) kabul eder.

### Linux / macOS

Kod taşınabilirdir, ancak `start.ps1` ve bu belgedeki `\` içeren yollar Windows içindir. Eşdeğer komutlar (bu depoda yalnızca Windows'ta denendi):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m uvicorn visionflow.api:create_app --factory --host 127.0.0.1 --port 8000
```

### Bağımlılık sürümleri

`pyproject.toml` doğrudan bağımlılıkları test edilen sürümlere sabitler (NFR-9): `fastapi==0.162.0`, `uvicorn[standard]==0.54.0`, `python-multipart==0.0.32`, `ultralytics==8.4.191`, `opencv-python==5.0.0.93`, `imageio-ffmpeg==0.7.0`, `numpy==2.5.4`, `lap==0.5.13`, `PyYAML==6.0.3`; geliştirme için `pytest==9.2.0`, `pytest-cov==7.1.0`, `httpx==0.28.1`, `ruff==0.17.3`. `torch` doğrudan bağımlılık değildir (Ultralytics getirir; bkz. "GPU kullanımı"). Yükseltmek için: `pyproject.toml` içindeki sürümü değiştirin, `.venv\Scripts\python -m pip install -e ".[dev]"` çalıştırın (veya `start.ps1`, değişen dosyayı görüp yeniden kurar) ve testleri çalıştırın.

## GPU kullanımı

Varsayılan `pip install -e .` (`start.ps1` de bunu çalıştırır) **CPU-only torch** kurar; bu durumda NVIDIA ekran kartı olsa bile GPU kullanılmaz (geliştirme makinesinde de böyleydi: `torch 2.15.0+cpu`, `torch.cuda.is_available()` False, RTX 4060 Laptop GPU mevcut). GPU için, `pip install -e .` işleminden SONRA, sanal ortamın içinde CUDA destekli torch kurun:

1. <https://pytorch.org/get-started/locally/> sayfasında Windows / Pip / kendi CUDA sürümünüzü seçin.
2. Sayfanın ürettiği `pip install ...` komutunu `.venv` içinde çalıştırın (ör. `.venv\Scripts\python -m pip install ...`).
3. Kontrol edin:

```powershell
.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Beklenen: `True` ve `+cuXXX` ekli bir sürüm. `False` ise GPU kullanılmıyordur. Bu adımlar bu depoda uçtan uca denenmedi.

## Kullanım

1. **Video ekleyin:** 1-5 video (`.mp4`, `.mov`, `.avi`, `.mkv`; her biri en çok 500 MB). Sürükleyip bırakabilir veya tıklayıp seçebilirsiniz.
2. **Sayım çizgisini çizin:** Soldaki önizlemede (videonun ilk karesi) çizginin iki ucunu sürükleyin ya da odaklanıp ok tuşlarıyla taşıyın. Her videonun kendi çizgisi vardır. Ok, "Yön 1" tarafını gösterir; çizgiyi geçen araç oka doğru gidiyorsa Yön 1, tersiyse Yön 2 sayılır. **Yönü çevir** düğmesi okun yönünü değiştirir (Yön 1 ile Yön 2'nin yerini değiştirir).
3. **Ayarlar:** Model `yolo11n` / `yolo11s` / `yolo11m` (büyük model daha doğru ama daha yavaştır). Arayüz açılışta sunucuya CUDA olup olmadığını sorar (`GET /api/system`): GPU varsa `yolo11s`, yoksa `yolo11n` önceden seçilir ve model seçicinin altında "GPU yok: yolo11n önerilir" notu görünür; elle yaptığınız seçim korunur. (`POST /api/jobs` isteğinde model belirtilmezse sunucu varsayılanı `yolo11s`'tir.) Güven eşiği 0,05-0,95 arasıdır (varsayılan 0,25); düşük değer daha çok araç yakalar ama daha çok yanlış alarm verir. Eşik ByteTrack'in içinde uygulanır (bkz. "Bilinen sınırlamalar"). "Listeden çıkar" yüklenen videoyu sunucudan da siler.
4. **Sayımı başlatın.** Videolar sırayla işlenir (aynı anda bir video); ilerleme çubuğunun yanında aşama görünür: *model yükleniyor* (ilk çalıştırmada ağırlık indirmesi dahil), *kareler işleniyor*, *sonuç videosu hazırlanıyor*.
5. **Sonuçlar:** İş bitince toplam, yöne/sınıfa göre dağılım, dakikalık grafik, güvenilirlik rozeti, çizgili sonuç videosu (tarayıcıda oynar), MP4, CSV ve Özet CSV indirme düğmeleri ile şüpheli geçiş listesi görünür. Listedeki bir satıra tıklamak videoyu o ana atlatır. Sonuçlar veri klasöründe (`outputs/<iş kimliği>/`, bkz. "Gizlilik ve veri klasörü") saklanır; sayfa yenilense veya sunucu yeniden başlasa da arayüzde yeniden listelenir ve arayüzden silinebilir (hatalı/yarıda kalmış işler dahil). Silerken dosya başka bir programda açıksa iş silinmez ve "videoyu kapatıp tekrar deneyin" uyarısı görünür. Kaynak video işlemden sonra silinir.

Arayüz Türkçe ve İngilizcedir (sağ üstteki dil seçici).

## Sonuçları okuma

- **Toplam:** `N araç (a fazla, b eksik olabilir)`. `a`, toplamda sayılmış ama şüpheli işaretlenmiş (yanlış fazla sayılmış olabilecek) geçiş sayısıdır; `b`, çizgiye varmadan kaybolan, sayılmamış olası kaçırılan araç sayısıdır. Bu bir güven aralığı değil, sezgisel bir uyarıdır.
- **Olası kaçırılan araçlar toplama dahil DEĞİLDİR.** Yalnızca listede ve CSV'de görünür.
- **Güvenilirlik rozeti:** `(a + b) / (toplam + b)` oranına göre. Oran `< 0,05` ise Yüksek, `< 0,15` ise Orta, değilse Düşük; hiç araç sayılmadıysa Belirsiz. Eşikler `config/reliability.json` dosyasındadır ve **yer tutucudur** (`"placeholder": true`); UA-DETRAC ile ayarlanmamıştır.
- **Şüpheli geçiş türleri:**
  - *olası kaçırılan araç:* Sayılmamış bir takip, çizgi parçasının hizasında (çizgi uzunluğunun %5'i kadar pay ile; sayaçla aynı) çizgiye 60 px'ten yakın bir yerde kayboldu. Ayrıca: takip en az 10 kare sürmüş olmalı, çizgiye ilk görüldüğü yerden en az 20 px yaklaşmış olmalı, çizginin hâlâ başladığı tarafında olmalı ve videonun son 5 karesinde görünmüyor olmalı (video bittiği için "kaybolmuş" sayılmaz). Yaklaşma yönü (gittiği taraf) listede ve CSV'nin `yön` sütununda verilir. Toplamda yoktur.
  - *olası çift sayım:* Sayılan bir takip, 15 kare içinde ve 40 px yakında biten başka bir sayılmış takibin hemen ardından başladı (aynı araç iki kimlik almış olabilir).
  - *kararsız geçiş:* Takip 30 kare içinde çizginin iki yanı arasında 3 kez gidip geldi (çizgiye 2 px'ten yakın noktalar sayaçtaki gibi yok sayılır; çizgi üzerinde duran aracın titremesi işaretlenmez).
  - *zayıf tespit:* Sayılan takip 10 kareden kısa sürdü veya ortalama güveni 0,35'ten düşük.
  - *sınıf belirsiz:* Takibin kareleri arasında araç sınıfı değişti.
  - *kalabalık geçiş:* Sayım anında araç kutusu başka bir kutuyla %50'den fazla örtüşüyordu.

  Bu eşiklerin hepsi de **yer tutucudur** (`src/visionflow/suspicion.py` içindeki `FlagConfig`) ve ayarlanmamıştır. "Şüpheli" bir geçiş sayılmış olabilir; işaret yalnızca gözle kontrol içindir.
- **CSV** (`crossings.csv`, UTF-8 **BOM'lu**, virgülle ayrılmış) sütunları: `tür`, `video`, `kare`, `zaman_s`, `takip_kimliği`, `sınıf`, `yön`, `işaretler`, `güven`. `tür` değeri sayılan geçişler için `geçiş`, olası kaçırılan araçlar için `olası_kaçırılan`'dır (bu satırlarda kare, sınıf ve güven boştur; `yön` aracın yaklaştığı yöndür; `zaman_s` takibin son görüldüğü andır). Önce geçişler zamana göre, sonra kaçırılanlar sıralanır. `=`, `+`, `-`, `@` ile başlayan video adları e-tablo formülü sayılmasın diye başına `'` konur.
- **Özet CSV** (`summary.csv`, indirirken `<video>_summary.csv`; UTF-8 BOM'lu, `metrik,deger` satırları; FR-12): `video`, `Yön 1`, `Yön 2`, `toplam`, `fazla`, `eksik`, `guvenilirlik`, `sure_s`, `fps_isleme`, `model` ve sınıf başına sayılar (`car`, `bus`, `truck`, `motorcycle`).
- **Excel:** BOM sayesinde Türkçe karakterler doğru görünür. Türkçe bölge ayarlı Excel liste ayırıcısı olarak `;` beklediği için dosyaya çift tıklayınca tüm satır tek sütunda görünebilir; bu durumda **Veri > Metinden/CSV'den** ile içe aktarıp ayırıcı olarak virgülü seçin (veya Windows bölge ayarlarında liste ayırıcısını değiştirin).

## Gizlilik ve veri klasörü

Video işleme yerel makinede yapılır; uygulama videoları veya sonuçları hiçbir sunucuya yüklemez. Ağ erişimi şunlardır:

- İlk kullanımda Ultralytics model ağırlıklarını (`yolo11*.pt`) indirir (ilk kurulumda `pip` de paketleri indirir).
- **Ultralytics, kapatılmadıkça anonim kullanım istatistikleri (telemetri) gönderebilir** (Ultralytics'in varsayılanı açık `sync` ayarı; video içeriği değil, ör. Python/torch sürümü, CPU bilgisi ve kullanılan işlem türü gibi kullanım olayları). Uygulama bu kullanıcı genelindeki ayarı kendisi değiştirmez. Kapatmak için sanal ortamın içinde bir kez çalıştırın:

  ```powershell
  .venv\Scripts\yolo settings sync=False
  ```

  (Linux/macOS: `.venv/bin/yolo settings sync=False`.) Bu, Ultralytics'in kullanıcı ayar dosyasını (Windows'ta genellikle `%APPDATA%\Ultralytics\settings.json`) değiştirir ve bu kullanıcının tüm Ultralytics projelerini etkiler.

Yüklenen videolar ve sonuçlar **depo klasörünün dışında**, bir veri klasöründe tutulur; böylece depo OneDrive gibi senkronize bir klasördeyse bile videolar varsayılan olarak senkronize edilmez:

- Windows: `%LOCALAPPDATA%\visionflow\` (ör. `C:\Users\<kullanıcı>\AppData\Local\visionflow\`; yerel, dolaşmayan klasör);
- Linux/macOS: `$XDG_DATA_HOME/visionflow/`, tanımlı değilse `~/.local/share/visionflow/`.

İçinde `uploads/` (yüklenen, henüz işlenmemiş videolar) ve `outputs/` (sonuçlar) bulunur. Başka bir yer için sunucuyu başlatmadan önce `VISIONFLOW_DATA_DIR` ortam değişkenini ayarlayın (ör. `$env:VISIONFLOW_DATA_DIR = "D:\visionflow-veri"`); bu klasörü senkronize bir klasör seçerseniz videolar da senkronize edilir.

Temizlik: kaynak video işlemden sonra (başarı veya hata) silinir; "Listeden çıkar" yüklenen videoyu siler; sunucu her başlangıçta 24 saatten eski veya boş yükleme klasörlerini ve yarıda kalmış işlerin geçici (`*.tmp*`) dosyalarını siler. Sonuçlar siz silene kadar kalır.

## Değerlendirme betiği (UA-DETRAC)

`eval/evaluate.py`, sayım hattını UA-DETRAC dizilerinde YOLO11 modelleriyle karşılaştırır. Beklenen yapı:

- `--images`: her dizi için bir klasör (`MVI_20011/` gibi) ve içinde `img*.jpg` kareleri.
- `--annotations`: her dizi için `<DİZİ>.xml` (ör. `MVI_20011.xml`).
- `--lines` (varsayılan `eval/lines.json`): `{"frozen": false, "sequences": {"MVI_20011": [x1, y1, x2, y2]}}`; koordinatlar 0-1 arası normalize edilmiştir. Çizgiler yalnızca **train** dizilerinde seçilir; sonra `"frozen": true` yapılır. `--split test` yalnızca dosya dondurulmuşsa çalışır (UA-DETRAC train/test klasörleri ayrı gelir, bu resmi ayrımı korur).

```powershell
.venv\Scripts\python -m eval.evaluate --images C:\veri\DETRAC-train-data\Insight-MVT_Annotation_Train --annotations C:\veri\DETRAC-Train-Annotations-XML --lines eval\lines.json --out rapor.csv --models yolo11n.pt yolo11s.pt --split train --repeats 3
```

Diğer bayraklar: `--sequences MVI_20011 ...` (varsayılan: tüm XML'ler), `--confidence` (varsayılan 0,25). Çıktı sütunları: `model`, `sequence`, `gt_count`, `system_count`, `matched`, `total_error`, `abs_error`, `relative_error`, `precision`, `recall`, `f1`, `flagged`, `flagged_wrong`, `flag_precision`, `missed_flags`, `missed_hits`, `missed_flag_hit_rate`, `median_fps`, `mean_relative_error`, `warnings`, `error`. Her model için sonda `sequence=ALL` satırı vardır; başarılı dizilerin ham sayımlarını toplayarak hesaplanır (hatalı diziler dışarıda kalır). Metrik tanımları için `eval/evaluate.py` dosyasının başındaki açıklamaya bakın. Çıkış kodu: 0 başarı, 1 bazı çalıştırmalar başarısız, 2 argüman/dosya hatası. Yukarıdaki klasör yolları örnektir; kendi veri konumunuza göre değiştirin.

## Test ve lint

```powershell
.venv\Scripts\python -m pytest                                          # hızlı testler (gerçek model yok)
.venv\Scripts\python -m pytest --cov=visionflow --cov-fail-under=90     # kapsama (NFR-8)
.venv\Scripts\python -m ruff check src tests eval
```

Gerçek YOLO modeli gerektiren `slow` test, örnek bir trafik videosuyla elle çalıştırılır:

```powershell
$env:VISIONFLOW_SAMPLE_VIDEO = "C:\yol\ornek.mp4"
.venv\Scripts\python -m pytest -m slow
```

`VISIONFLOW_SAMPLE_VIDEO` tanımlı değilse bu test atlanır.

## Proje yapısı

```
src/visionflow/   api.py (FastAPI uçları), jobs.py (iş kuyruğu), pipeline.py, tracking.py (YOLO+ByteTrack),
                  counter.py, geometry.py, suspicion.py, summary.py, export.py (CSV + çizgili video), config.py,
                  paths.py (veri klasörü + yükleme temizliği)
web/              index.html, app.js, i18n.js, style.css (arayüz)
config/           reliability.json (güvenilirlik eşikleri, yer tutucu)
eval/             evaluate.py, ground_truth.py, lines.json (UA-DETRAC değerlendirmesi)
tests/            pytest testleri
docs/             requirements.md, tasarım belgesi ve planlar
start.ps1         Windows kurulum + başlatma betiği
```

API uçları: `GET /api/health`, `GET /api/system` (`{"cuda", "default_model"}`), `POST /api/videos`, `GET /api/videos/{id}/preview`, `DELETE /api/videos/{id}` (gönderilmiş videoda 409), `POST /api/jobs`, `GET /api/jobs`, `GET /api/jobs/{id}` (`status`, `progress`, `phase`, `error`), `GET /api/jobs/{id}/result`, `GET /api/jobs/{id}/video`, `GET /api/jobs/{id}/csv`, `GET /api/jobs/{id}/summary`, `DELETE /api/jobs/{id}` (dosya kullanımdaysa 409).

## Bilinen sınırlamalar

- **Gerçek YOLO + ByteTrack hattı bu depodaki otomatik çalıştırmalarda uçtan uca doğrulanmadı.** Otomatik testler sahte (fake) takip kaynağı kullanır; gerçek model için ağırlık dosyası ve örnek video gerekir. Elle kontrol: (1) `.\start.ps1` ile sunucuyu başlatın (ilk işte seçilen modelin ağırlığı, ör. `yolo11n.pt`, indirilir; bu sırada aşama "model yükleniyor" görünür); (2) trafik içeren kısa bir video yükleyin, çizgiyi yolu kesecek şekilde çizin, sayımı başlatın; (3) iş "Tamamlandı" olunca sonuç videosunun tarayıcıda oynadığını, çizginin ve kutuların göründüğünü ve toplamın gözle saydığınız sayıya yakın olduğunu kontrol edin; (4) CSV'de `geçiş` satırları (varsa `olası_kaçırılan` satırları) olduğunu doğrulayın. İsterseniz yukarıdaki `pytest -m slow` testini de çalıştırın.
- **Güven eşiği ByteTrack içinde uygulanır, bu değişiklik gerçek modelle DOĞRULANMADI.** Dedektör ByteTrack'in ikinci (düşük skorlu) eşleştirme aşamasını besleyebilmek için düşük bir tabanla (`min(eşik, 0,1)`) çalışır; kullanıcının eşiği her iş için üretilen geçici bir ByteTrack ayar dosyasında `track_high_thresh` ve `new_track_thresh` olarak, `track_low_thresh` 0,1 olarak verilir (Ultralytics'in kurulu `bytetrack.yaml` dosyasından kopyalanır, sistemin geçici klasörüne yazılır ve iş bitince silinir). Birim testleri yalnızca sahte bir YOLO ile yapıldı; etkisi gerçek bir çalıştırma ve UA-DETRAC değerlendirmesiyle ölçülmelidir.
- **Ctrl+C ile durdurma:** çalışan iş ve kuyruktaki işler "interrupted" hatasıyla biter ve **kaynak videoları silinir**; bu videoları yeniden yüklemeniz gerekir. Sunucu beklenmedik şekilde kapanırsa, yarıda kalan işlerin kaynak videoları bir sonraki başlangıçta (bulunabilirse) silinir.
- **Motosiklet** sınıfı desteklenir ancak benchmark ile ölçülmedi (UA-DETRAC'ta motosiklet etiketi yoktur).
- **Güvenilirlik eşikleri ve şüpheli geçiş eşikleri yer tutucudur**; UA-DETRAC ile ayarlanması gerekir.
- **UA-DETRAC XML ayrıştırıcısı gerçek veri olmadan yazıldı.** Gerçek veriyle şunları doğrulayın: (1) kare numaraları gerçekten 1 tabanlı mı (kod `num - 1` kullanır); (2) kutular piksel cinsinden mi (`left`, `top`, `width`, `height`); (3) yok sayılan bölge (ignored region) testi kutunun alt orta noktasına göre yapılır, bu seçim doğru mu.
- **GPU varsayılan olarak kullanılmaz:** varsayılan kurulum CPU-only torch getirir; GPU için CUDA torch'u elle kurmanız gerekir (bkz. "GPU kullanımı"). GPU yoksa arayüz `yolo11n`'i önerir. CUDA kurulumu bu depoda doğrulanmadı.
- **Windows dosya kilitleri:** Veri klasörü varsayılan olarak senkronize edilmeyen `%LOCALAPPDATA%` altındadır. `VISIONFLOW_DATA_DIR` ile senkronize bir klasör (OneDrive vb.) seçerseniz dosya kilitleri çıktı videosunun yazılmasını veya silinmesini engelleyebilir (kod birkaç kez yeniden dener; silme başarısız olursa iş listede kalır ve sonra yeniden silinebilir).
- **ASCII olmayan yollar:** Yüklenen dosya adları ASCII'ye indirgenir. Değerlendirme betiği kareleri `imdecode` ile okur. Uygulamanın kendi video okuması (`cv2.VideoCapture`) için ASCII olmayan veri klasörleri (ör. Windows kullanıcı adında ASCII olmayan harf varsa `%LOCALAPPDATA%` yolu) uçtan uca DOĞRULANMADI (bir inceleyici Windows'ta ASCII olmayan bir klasörde VideoCapture yazma/açma/probe işlemini yalnızca bir kez örnek olarak denedi). Sorun olursa `VISIONFLOW_DATA_DIR` ile ASCII bir yol seçin.
