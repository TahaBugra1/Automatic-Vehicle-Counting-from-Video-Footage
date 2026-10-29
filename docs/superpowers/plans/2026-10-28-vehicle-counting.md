# Araç Sayımı Uygulama Planı

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sabit kameralı trafik videolarından, video başına ayrı çizgiyle yön/sınıf/zaman bazlı araç sayımı yapan, şüpheli geçişleri ve belirsizliği gösteren yerel bir web uygulaması ve UA-DETRAC değerlendirme betiği.

**Architecture:** Saf Python çekirdek (geometri, sayaç, şüpheli geçiş kuralları, özet) test edilebilir ve model bağımsızdır. Tespit/takip bir `TrackSource` arayüzü arkasındadır (gerçek: YOLO11 + ByteTrack; testlerde sahte kaynak). Boru hattı sonuçları dosyalara yazar, tek işçili iş kuyruğu videoları sırayla işler, FastAPI ve sade HTML/JS arayüz bunların üstündedir.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, ultralytics (YOLO11, ByteTrack), opencv-python, imageio-ffmpeg, numpy, pytest, ruff. Arayüz: sade HTML/JS, bağımlılıksız.

**Spec:** `docs/superpowers/specs/2026-10-28-vehicle-counting-design.md` (gereksinimler: `docs/requirements.md`)

## Global Constraints

- Yerel çalışma; videolar üçüncü tarafa gönderilmez (NFR-10).
- Model: COCO önceden eğitilmiş YOLO11; arayüzde `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt` seçilebilir, **varsayılan `yolo11s.pt`** (Task 10 ölçümüne göre gözden geçirilir; ince ayar yapılmaz).
- Sınıflar: `car`, `bus`, `truck`, `motorcycle` (COCO sınıf numaraları 2, 5, 7, 3).
- Yüklenebilir: 1-5 video, uzantılar `.mp4 .mov .avi .mkv`, en çok 500 MB, geçersizse anlaşılır hata (FR-1, FR-2).
- Çizgi koordinatları 0-1 aralığında (normalize) saklanır; **her video için ayrı çizgi**; çizgi yönlüdür (FR-3, FR-4).
- Her takip kimliği en çok bir kez sayılır; ölü bant çizgi titreşimini bastırır (FR-7, FR-8).
- Toplam: "olası kaçırılan araç" toplama **dahil değildir**; gösterim: `222 araç (3 fazla, 4 eksik olabilir)` (FR-17).
- Yön adları `Yön 1` / `Yön 2`; CSV'de de aynı adlar.
- Zaman dağılımı varsayılan dakikalık; güvenilirlik rozeti üç seviye (yüksek/orta/düşük), eşikler `config/reliability.json` içinde yer tutucu değerlerle.
- Arayüz Türkçe ve İngilizce (NFR-4). Çıktı video H.264 MP4 (FR-11).
- Kaynak video işlem bitince silinir; sonuçlar diskte kullanıcı silene kadar kalır.
- Kod Şeyda'nın deposundan kopyalanmaz (yalnızca referans). Paket adı `visionflow`.
- **Commit ve push yalnızca kullanıcı tarafından, elle yapılır.** Hiçbir görev `git commit` ya da `git push` çalıştırmaz; her görevin son adımı önerilen commit mesajını yazmaktır.

## Review Focus

1. Hiç tespit içermeyen video: sayım 0, çökme yok, rozet "yüksek" gösterilmez (veri yok durumu "belirsiz" olarak ayrı işaretlenir).
2. 5 videodan biri bozuk/okunamaz: o video "hata" durumuna geçer, diğerleri tamamlanır.
3. Aynı noktada duran araç (sıkışık trafik) çizgi üzerinde dalgalanır: en çok bir kez sayılır ve "kararsız geçiş" olarak işaretlenir.
4. Bir dakikadan kısa video ve fps değeri 0/bilinmeyen video: zaman dağılımı tek kutu, bölme hatası yok.
5. Dosya adında yol karakterleri, boşluk, Türkçe harf: dosya sistemine güvenli adla yazılır, yol dışına çıkamaz.

---

## Dosya Yapısı

| Dosya | Sorumluluk |
|---|---|
| `pyproject.toml`, `.gitignore` | Bağımlılıklar (sabit sürüm aralıkları), pytest/ruff ayarı; `outputs/`, `.venv/`, `*.pt` yok sayılır |
| `src/visionflow/models.py` | Ortak veri sınıfları ve enum'lar |
| `src/visionflow/geometry.py` | Yönlü çizgi, taraf/izdüşüm hesapları |
| `src/visionflow/counter.py` | `LineCounter`: geçiş algılama |
| `src/visionflow/suspicion.py` | `TrackRecorder`: şüpheli geçiş kuralları |
| `src/visionflow/summary.py` | Zaman/sınıf dağılımı, belirsizlik, güvenilirlik |
| `src/visionflow/config.py` + `config/reliability.json` | Eşik/yapılandırma yükleme |
| `src/visionflow/tracking.py` | `TrackSource` arayüzü ve YOLO11+ByteTrack uyarlayıcısı |
| `src/visionflow/pipeline.py` | Bir videoyu uçtan uca işler |
| `src/visionflow/export.py` | CSV ve işaretli H.264 MP4 |
| `src/visionflow/jobs.py` | Sıralı iş kuyruğu, kalıcı sonuç, temizlik |
| `src/visionflow/api.py` | FastAPI uç noktaları |
| `web/index.html`, `web/app.js`, `web/i18n.js`, `web/style.css` | Tek sayfa arayüz |
| `eval/ground_truth.py`, `eval/evaluate.py`, `eval/lines.json` | UA-DETRAC değerlendirmesi |
| `tests/` | Her modül için testler |

---

### Task 1: Proje iskeleti

**Files:**
- Create: `pyproject.toml`, `src/visionflow/__init__.py`, `tests/test_smoke.py`, `config/reliability.json`
- Modify: `.gitignore` (mevcut `Hafta3_Proje_Raporu_.docx` satırını koru)

**Interfaces:**
- Produces: `visionflow.__version__: str`

- [ ] **Step 1: Failing test:** `tests/test_smoke.py::test_package_importable` — `import visionflow; assert visionflow.__version__ == "0.1.0"`.
- [ ] **Step 2: Çalıştır, FAIL gör:** `python -m pytest tests/test_smoke.py -v` → `ModuleNotFoundError`.
- [ ] **Step 3: Oluştur:** `pyproject.toml` (src düzeni, Python ≥3.11, bağımlılıklar `fastapi`, `uvicorn[standard]`, `python-multipart`, `ultralytics`, `opencv-python`, `imageio-ffmpeg`, `numpy`, `lap`; dev: `pytest`, `httpx`, `ruff`; pytest `testpaths=["tests"]`, ruff satır uzunluğu 100) ve paket. `config/reliability.json`: `{"high_below": 0.05, "medium_below": 0.15, "placeholder": true}`.
- [ ] **Step 4: Kur ve çalıştır:** `python -m pip install -e ".[dev]"` sonra `python -m pytest -v` → PASS.
- [ ] **Step 5: Önerilen commit mesajı yaz (çalıştırma):** `chore: scaffold visionflow package`

---

### Task 2: Geometri ve çizgi sayacı

**Files:**
- Create: `src/visionflow/models.py`, `src/visionflow/geometry.py`, `src/visionflow/counter.py`
- Test: `tests/test_counter.py`

**Interfaces:**
- Produces (`models.py`): `VehicleClass(StrEnum)` değerleri `car, bus, truck, motorcycle`; `Direction(StrEnum)` `ONE="Yön 1"`, `TWO="Yön 2"`; `Detection(frame:int, time_s:float, track_id:int, cls:VehicleClass, confidence:float, bbox:tuple[float,float,float,float])` (piksel, x1 y1 x2 y2, `frozen, slots`) ve `Detection.bottom_center -> tuple[float,float]`; `CrossingEvent(frame, time_s, track_id, cls, direction, confidence)`.
- Produces (`geometry.py`): `Line(start:tuple[float,float], end:tuple[float,float])` normalize; `Line.to_pixels(width:int, height:int) -> Line`; `Line.flipped() -> Line` (uçları değiştirir); `Line.signed_distance(p) -> float`; `Line.projection(p) -> float` (0..1 parça içinde). Uçlar eşitse `ValueError`.
- Produces (`counter.py`): `LineCounter(line: Line, deadband: float = 2.0, projection_margin: float = 0.05)`; `update(det: Detection) -> CrossingEvent | None`; `counts -> dict[Direction, int]`. **Yön 1 = negatiften pozitif tarafa geçiş**; pozitif taraf `signed_distance > 0`. Okun gösterdiği taraf pozitif taraftır.

- [ ] **Step 1: Failing testler** (`tests/test_counter.py`), yatay çizgi `(0,100)→(200,100)`, piksel uzayında:
  - `test_crossing_counts_once_per_track`: aynı kimlik çizgiyi 2 kez geçse de toplam 1.
  - `test_direction_one_when_moving_to_positive_side`, `test_direction_two_when_moving_to_negative_side`.
  - `test_deadband_suppresses_jitter`: çizgi ±2 px içinde gidip gelen kimlik 0 sayım.
  - `test_crossing_outside_segment_is_ignored`: izdüşüm marjı dışında geçiş 0.
  - `test_two_distinct_tracks_counted_separately`: kimlik 2 ve 4 → toplam 2.
  - `test_zero_length_line_raises_value_error`.
- [ ] **Step 2: Çalıştır, FAIL gör.**
- [ ] **Step 3: Uygula:** yukarıdaki imzalarla; sayaç kimlik başına son kararlı tarafı tutar, ölü banttaki noktaları yok sayar.
- [ ] **Step 4: Çalıştır, PASS gör:** `python -m pytest tests/test_counter.py -v`.
- [ ] **Step 5: Önerilen commit mesajı:** `feat: add directed line geometry and crossing counter`

---

### Task 3: Şüpheli geçiş kuralları

**Files:**
- Create: `src/visionflow/suspicion.py`
- Test: `tests/test_suspicion.py`

**Interfaces:**
- Consumes: `Detection`, `CrossingEvent`, `Line` (piksel), `Direction`.
- Produces: `FlagKind(StrEnum)`: `MISSED="olası kaçırılan araç"`, `DOUBLE_COUNT="olası çift sayım"`, `UNSTABLE="kararsız geçiş"`, `WEAK="zayıf tespit"`, `CLASS_UNCERTAIN="sınıf belirsiz"`, `CROWDED="kalabalık geçiş"`; `Flag(kind, track_id:int, frame:int, time_s:float, counted:bool, detail:str)`; `FlagConfig(short_track_frames=10, low_confidence=0.35, unstable_side_changes=3, unstable_window_frames=30, lost_near_line_px=60.0, double_count_px=40.0, double_count_frames=15, crowd_iou=0.5)`; `TrackRecorder(line: Line, config: FlagConfig = FlagConfig())` ile `observe(det: Detection) -> None` ve `finish(events: list[CrossingEvent]) -> list[Flag]`.
- Kural tanımları: `MISSED`: sayılmamış, ≥ `short_track_frames` uzunluğunda, son konumu çizgiye `lost_near_line_px` içinde ve çizgiye yaklaşırken biten takip. `DOUBLE_COUNT`: sayılan takip, kendisinden en çok `double_count_frames` önce biten başka bir takibin son konumunun `double_count_px` yakınında başlıyor. `UNSTABLE`: `unstable_window_frames` içinde ≥ `unstable_side_changes` taraf değişimi. `WEAK`: takip `short_track_frames`'ten kısa ya da ortalama güven < `low_confidence` (yalnızca sayılan takipler). `CLASS_UNCERTAIN`: takip boyunca ≥2 farklı sınıf. `CROWDED`: geçiş karesinde başka bir kutuyla IoU > `crowd_iou`.

- [ ] **Step 1: Failing testler**, her tür için en az biri: `test_missed_flag_when_track_ends_near_line_uncounted`, `test_double_count_flag_when_new_track_starts_where_another_ended`, `test_unstable_flag_on_flip_flopping_track`, `test_weak_flag_for_short_counted_track`, `test_class_uncertain_when_class_changes`, `test_crowded_flag_on_overlapping_boxes_at_crossing`, `test_clean_track_produces_no_flags`, `test_missed_flag_not_counted_in_total` (`flag.counted is False`).
- [ ] **Step 2: FAIL gör.**
- [ ] **Step 3: Uygula** `TrackRecorder` (kimlik başına geçmiş tutar, `finish` kuralları sırayla uygular).
- [ ] **Step 4: PASS gör:** `python -m pytest tests/test_suspicion.py -v`.
- [ ] **Step 5: Önerilen commit mesajı:** `feat: flag suspicious crossings`

---

### Task 4: Özet: dağılım, belirsizlik, güvenilirlik

**Files:**
- Create: `src/visionflow/summary.py`, `src/visionflow/config.py`
- Test: `tests/test_summary.py`

**Interfaces:**
- Consumes: `CrossingEvent`, `Flag`, `FlagKind`, `Direction`, `VehicleClass`.
- Produces (`config.py`): `ReliabilityThresholds(high_below: float, medium_below: float)`; `load_reliability(path: Path | None = None) -> ReliabilityThresholds` (varsayılan `config/reliability.json`).
- Produces (`summary.py`): `Reliability(StrEnum)` `HIGH, MEDIUM, LOW, UNKNOWN`; `TimelineBin(minute:int, one:int, two:int)`; `VideoSummary(total:dict[Direction,int], total_count:int, by_class:dict[VehicleClass,int], timeline:list[TimelineBin], too_many:int, too_few:int, reliability:Reliability, flags:list[Flag])`; `summarize(events: list[CrossingEvent], flags: list[Flag], duration_s: float, thresholds: ReliabilityThresholds) -> VideoSummary`; `format_total(summary: VideoSummary, lang: str) -> str` (tr: `"222 araç (3 fazla, 4 eksik olabilir)"`, en: `"222 vehicles (3 too many, 4 too few possible)"`).
- Kurallar: `too_many` = `MISSED` olmayan türlerden herhangi biriyle (`CLASS_UNCERTAIN` hariç) işaretlenmiş **benzersiz sayılan** geçişlerin sayısı; `too_few` = `MISSED` sayısı. Oran = `(too_many + too_few) / max(total_count + too_few, 1)`; olay yoksa güvenilirlik `UNKNOWN`. Zaman kutusu `int(time_s // 60)`; süre 0 ya da < 60 ise tek kutu.

- [ ] **Step 1: Failing testler:** `test_timeline_bins_by_minute`, `test_short_video_has_single_bin`, `test_zero_duration_does_not_divide_by_zero`, `test_by_class_counts`, `test_uncertainty_counts_missed_as_too_few`, `test_class_uncertain_does_not_affect_totals`, `test_reliability_levels_use_thresholds` (0.04→HIGH, 0.10→MEDIUM, 0.30→LOW), `test_no_events_reliability_unknown`, `test_format_total_tr` (`"222 araç (3 fazla, 4 eksik olabilir)"`), `test_format_total_en`.
- [ ] **Step 2: FAIL gör.**
- [ ] **Step 3: Uygula.**
- [ ] **Step 4: PASS gör:** `python -m pytest tests/test_summary.py -v`.
- [ ] **Step 5: Önerilen commit mesajı:** `feat: add summaries, uncertainty and reliability`

---

### Task 5: Tespit/takip kaynağı ve boru hattı

**Files:**
- Create: `src/visionflow/tracking.py`, `src/visionflow/pipeline.py`
- Test: `tests/test_pipeline.py` (sahte kaynak), `tests/test_tracking_real.py` (`@pytest.mark.slow`, model ister)

**Interfaces:**
- Consumes: Task 2-4 türleri.
- Produces (`tracking.py`): `FrameResult(frame:int, time_s:float, width:int, height:int, detections:list[Detection], image:numpy.ndarray)`; `Protocol TrackSource: frames(video_path: Path, model_name: str, confidence: float) -> Iterator[FrameResult]`; `UltralyticsTrackSource` (YOLO11, `tracker="bytetrack.yaml"`, yalnızca COCO sınıfları 2,3,5,7, fps 0/bilinmiyorsa 25 varsayar); `probe_video(path: Path) -> VideoInfo(width:int, height:int, fps:float, frame_count:int, duration_s:float)`; okunamayan videoda `VideoReadError`.
- Produces (`pipeline.py`): `ProcessOptions(model_name:str="yolo11s.pt", confidence:float=0.25)`; `VideoResult(summary: VideoSummary, events: list[CrossingEvent], fps: float, duration_s: float, processing_fps: float)`; `process_video(video_path: Path, line: Line, options: ProcessOptions, source: TrackSource, on_frame: Callable[[FrameResult, list[CrossingEvent]], None] | None = None, on_progress: Callable[[float], None] | None = None) -> VideoResult` (çizgi normalize gelir, piksele burada çevrilir).

- [ ] **Step 1: Failing testler** (sahte `TrackSource`): `test_pipeline_counts_scripted_crossings` (2 kimlik, biri her yönde → `total[Direction.ONE]==1`, `total[Direction.TWO]==1`), `test_pipeline_with_no_detections_returns_zero_and_unknown_reliability`, `test_line_scales_with_resolution` (aynı normalize çizgi 640×360 ve 1280×720'de aynı sayıyı verir), `test_progress_reaches_one`, `test_unreadable_video_raises_video_read_error`.
- [ ] **Step 2: FAIL gör.**
- [ ] **Step 3: Uygula**; `Detection.cls` COCO numarasından `VehicleClass`'a çevrilir.
- [ ] **Step 4: PASS gör:** `python -m pytest tests/test_pipeline.py -v`. Gerçek model testi ayrıca: `python -m pytest -m slow tests/test_tracking_real.py -v` (kısa örnek video ile en az bir araç tespiti; model ağırlığı ilk çalışmada iner).
- [ ] **Step 5: Önerilen commit mesajı:** `feat: add tracking source and processing pipeline`

---

### Task 6: Çıktılar: CSV ve işaretli MP4

**Files:**
- Create: `src/visionflow/export.py`
- Test: `tests/test_export.py`

**Interfaces:**
- Consumes: `VideoResult`, `Flag`, `FrameResult`, `Line`.
- Produces: `write_csv(result: VideoResult, video_name: str, path: Path) -> None` — sütunlar `tür, video, kare, zaman_s, takip_kimliği, sınıf, yön, işaretler, güven`; `tür` değeri `geçiş` ya da `olası_kaçırılan`; olası kaçırılanlarda kare/sınıf/yön bilinmiyorsa boş, zaman son görülen zaman. `AnnotatedWriter(path: Path, width:int, height:int, fps:float, line: Line)` bağlam yöneticisi, `write(frame: FrameResult, events: list[CrossingEvent]) -> None` (kutu, çizgi, ok, sayaçlar çizer), kapanışta H.264'e dönüştürür (imageio-ffmpeg).

- [ ] **Step 1: Failing testler:** `test_csv_has_header_and_crossing_rows`, `test_csv_marks_missed_rows_with_blank_direction`, `test_csv_uses_yon_labels`, `test_annotated_video_is_h264_and_readable` (ffprobe/OpenCV ile aç, `codec` H.264, kare sayısı sahte girdiyle eşit).
- [ ] **Step 2: FAIL gör.**
- [ ] **Step 3: Uygula.**
- [ ] **Step 4: PASS gör:** `python -m pytest tests/test_export.py -v`.
- [ ] **Step 5: Önerilen commit mesajı:** `feat: export CSV and annotated H.264 video`

---

### Task 7: İş kuyruğu ve kalıcı sonuçlar

**Files:**
- Create: `src/visionflow/jobs.py`
- Test: `tests/test_jobs.py`

**Interfaces:**
- Consumes: `process_video`, `write_csv`, `AnnotatedWriter`, `Line`.
- Produces: `JobStatus(StrEnum)` `QUEUED, RUNNING, DONE, ERROR`; `VideoJob(job_id:str, video_name:str, line:Line, status:JobStatus, progress:float, error:str|None)`; `JobRunner(output_root: Path, source: TrackSource)` ile `submit(video_path: Path, display_name: str, line: Line, options: ProcessOptions) -> str` (tek işçi iş parçacığı, sırayla), `get(job_id) -> VideoJob`, `result(job_id) -> dict` (diskten, sunucu yeniden başlasa da okunur), `delete(job_id) -> None`, `list_jobs() -> list[VideoJob]`. Sonuç `output_root/<job_id>/{result.json, annotated.mp4, crossings.csv}`; kaynak video işlem bitince (başarıyla ya da hatayla) silinir. Dosya adları `safe_stem(name: str) -> str` ile temizlenir (yalnızca `[A-Za-z0-9_-]`, en çok 80 karakter, boşsa `video`).

- [ ] **Step 1: Failing testler:** `test_jobs_run_sequentially_in_submit_order`, `test_bad_video_marks_error_and_next_job_still_completes`, `test_source_video_deleted_after_processing`, `test_result_survives_new_runner_instance`, `test_delete_removes_result_folder`, `test_safe_stem_blocks_path_traversal` (`"../../etc/x y.mp4"` yol dışına çıkmaz), `test_safe_stem_handles_turkish_and_empty_names`.
- [ ] **Step 2: FAIL gör.**
- [ ] **Step 3: Uygula.**
- [ ] **Step 4: PASS gör:** `python -m pytest tests/test_jobs.py -v`.
- [ ] **Step 5: Önerilen commit mesajı:** `feat: add sequential job runner with persisted results`

---

### Task 8: API

**Files:**
- Create: `src/visionflow/api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `JobRunner`, `probe_video`, `safe_stem`.
- Produces: `create_app(runner: JobRunner | None = None, upload_dir: Path | None = None) -> FastAPI` ile uç noktalar: `POST /api/videos` (1-5 dosya, doğrular, `[{video_id, name, width, height}]` döner), `GET /api/videos/{video_id}/preview` (ilk kare JPEG), `POST /api/jobs` (gövde: `{"videos":[{"video_id","line":[x1,y1,x2,y2]}], "model_name","confidence"}` → `[{job_id}]`), `GET /api/jobs` ve `GET /api/jobs/{job_id}` (durum/ilerleme), `GET /api/jobs/{job_id}/result`, `GET /api/jobs/{job_id}/video` ve `/csv` (indirme), `DELETE /api/jobs/{job_id}`, `GET /api/health`. Hata mesajları `Accept-Language`/`?lang=` ile tr/en.
- Doğrulama: uzantı, 500 MB, 1-5 dosya, çizgi 4 sayı ve her biri 0-1 aralığında ve uçlar farklı; ihlalde 400/413 ve anlaşılır ileti.

- [ ] **Step 1: Failing testler** (FastAPI `TestClient`, sahte `TrackSource`): `test_upload_rejects_bad_extension`, `test_upload_rejects_more_than_five_files`, `test_upload_rejects_oversize` (sınır test için küçültülebilir yapılandırma ile), `test_job_rejects_line_out_of_range`, `test_job_rejects_identical_endpoints`, `test_each_video_uses_its_own_line`, `test_result_contains_total_text_and_flags`, `test_download_csv_and_video`, `test_delete_job`.
- [ ] **Step 2: FAIL gör.**
- [ ] **Step 3: Uygula.**
- [ ] **Step 4: PASS gör:** `python -m pytest tests/test_api.py -v`.
- [ ] **Step 5: Önerilen commit mesajı:** `feat: add FastAPI upload, job and download endpoints`

---

### Task 9: Web arayüzü

**Files:**
- Create: `web/index.html`, `web/app.js`, `web/i18n.js`, `web/style.css`
- Modify: `src/visionflow/api.py` (statik dosya sunumu)
- Test: `tests/test_web.py` (statik dosya ve i18n tamlık testleri); elle kontrol listesi aşağıda

**Interfaces:**
- Consumes: Task 8 uç noktaları.
- Produces: Tek sayfa yerleşim (spec §5): solda seçili videonun önizlemesi ve sürüklenebilir yönlü çizgi (ok, "Yönü çevir" düğmesi), video listesi (çizgisi ayarlanmamış video işaretli, başlatma öncesi uyarı); sağda model, güven eşiği, "Sayımı başlat". Sonuç kartı: `format_total` metni, güvenilirlik rozeti, dakikalık grafik (elle SVG), sınıf dağılımı, kapalı "Şüpheli geçişler" bölümü (zaman ve neden; tıklayınca sonuç videosunda o saniyeye atlar), indirme düğmeleri, canlı ilerleme. `i18n.js`: `tr` ve `en` sözlükleri aynı anahtar kümesine sahip.

- [ ] **Step 1: Failing testler:** `test_index_served`, `test_i18n_tr_and_en_have_same_keys` (anahtarlar `i18n.js`'ten okunup karşılaştırılır), `test_static_assets_served`.
- [ ] **Step 2: FAIL gör.**
- [ ] **Step 3: Uygula** (sade JS, framework yok; çizgi koordinatları görüntü boyutuna göre normalize edilip gönderilir).
- [ ] **Step 4: PASS gör:** `python -m pytest tests/test_web.py -v`. Elle kontrol: sunucuyu başlat, iki video yükle, her birine farklı çizgi çiz, sayımı başlat; TR/EN geçişi; şüpheli satıra tıklayınca video atlar.
- [ ] **Step 5: Önerilen commit mesajı:** `feat: add single-page web UI with TR/EN`

---

### Task 10: UA-DETRAC değerlendirme betiği

**Files:**
- Create: `eval/ground_truth.py`, `eval/evaluate.py`, `eval/lines.json`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: `process_video`, `Line`, `CrossingEvent`, `Flag`.
- Produces: `derive_ground_truth_crossings(xml_path: Path, line: Line, width:int, height:int) -> list[GtCrossing(frame:int, track_id:int, direction:Direction)]` (UA-DETRAC XML iz etiketlerinden: her kimliğin alt orta noktası `LineCounter` ile aynı kuralla çizgiyi geçince bir geçiş); `evaluate_sequence(video_path, xml_path, line, source, options) -> SequenceMetrics(total_error:int, relative_error:float, precision:float, recall:float, f1:float, flag_precision:float, median_fps:float)`; komut satırı `python -m eval.evaluate --data DIR --lines eval/lines.json --out report.csv`. `lines.json` dizi adından çizgiye eşleme; test dizilerinden önce dondurulur ve değerlendirmede değişmez. Karşılaştırma toplam araç sayısı üzerinedir (UA-DETRAC sınıfları car/bus/van/others, sistem sınıflarıyla birebir eşleşmediği için sınıf bazlı doğruluk raporlanmaz). Geçiş eşleştirmesi: aynı yönde ve ±15 kare içinde. `flag_precision`: işaretli geçişlerden gerçek geçişle eşleşmeyenlerin oranı. UA-DETRAC kare dizisi (JPEG, 25 fps) olarak dağıtıldığı için `frames_to_video(frames_dir: Path, out_path: Path, fps: float = 25.0) -> Path` (ffmpeg ile MP4) betikte `evaluate_sequence` öncesinde çağrılır. Betik aynı eğitim dizilerinde `yolo11n/s/m` modellerini karşılaştırır (`--models yolo11n.pt yolo11s.pt yolo11m.pt`) ve model başına toplam sayım hatası ile ortanca FPS raporlar; varsayılan model bu ölçüme göre gözden geçirilir. Betik, resmi 60 eğitim / 40 test ayrımını korur; çizgileri yalnızca eğitim dizilerinde belirler.

- [ ] **Step 1: Failing testler** (küçük sentetik XML; `test_frames_to_video_makes_readable_mp4` ve `test_report_has_row_per_model` dahil): `test_ground_truth_counts_scripted_crossings`, `test_ground_truth_respects_direction`, `test_metrics_perfect_match`, `test_metrics_with_one_miss_and_one_extra`, `test_flag_precision_computation`, `test_lines_file_must_cover_all_sequences` (eksik dizi için anlaşılır hata).
- [ ] **Step 2: FAIL gör.**
- [ ] **Step 3: Uygula.** UA-DETRAC XML biçimi (`sequence/frame/target_list/target/box`) gerçek veriyle doğrulanır; biçim farklıysa ayrıştırıcı uyarlanır.
- [ ] **Step 4: PASS gör:** `python -m pytest tests/test_eval.py -v`.
- [ ] **Step 5: Önerilen commit mesajı:** `feat: add UA-DETRAC evaluation script`

---

### Task 11: Belgeler ve uçtan uca doğrulama

**Files:**
- Modify: `README.md`
- Create: `start.ps1` (Windows kurulum + sunucu başlatma)
- Test: elle uçtan uca (aşağıda)

- [ ] **Step 1: README:** kurulum, çalıştırma, test, değerlendirme, motosiklet için "desteklenir, benchmark ile ölçülmedi" notu, güvenilirlik eşiklerinin yer tutucu olduğu notu.
- [ ] **Step 2: `start.ps1`:** venv oluşturur, `pip install -e .`, sunucuyu `http://127.0.0.1:8000`'da başlatır.
- [ ] **Step 3: Uçtan uca (gerçek model):** iki kısa video ile arayüzden sayım; çıktı videosu tarayıcıda oynar; CSV'de `geçiş` ve gerekirse `olası_kaçırılan` satırları var.
- [ ] **Step 4: Tüm testler, kapsam ve lint:** `python -m pytest --cov=visionflow --cov-fail-under=90` ve `python -m ruff check src tests eval` → hatasız (NFR-8; Task 1'de dev bağımlılıklarına `pytest-cov` eklenir).
- [ ] **Step 5: Önerilen commit mesajı:** `docs: add setup guide and start script`
