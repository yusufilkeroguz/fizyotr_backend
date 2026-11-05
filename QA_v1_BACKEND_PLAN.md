# FizyoTR QA v1 — Backend (Cloud Run) İş Planı

Hedef: **Analiz backend'i** (`https://fizyotr-backend-...run.app`, `/analyze/<slug>`).
Kaynak: `FizyoTR_Yonerge_ve_Rapor_Duzenleme_Talepleri_v1.docx`. Bu dosya, QA v1'in
**dashboard'da çözülemeyen** (hesaplama/veri gerektiren) maddelerini içerir.
Dashboard tarafı Faz 0-3 ile tamamlandı; aşağıdakiler backend'e bağımlıdır.

## Sözleşme (mevcut)

Backend her testte aynı zarfı döner:
```json
{ "category": "<kategori>", "test": "<test-id>", "result": { ... teste özel ... } }
```
- Yeni **hesaplanan** alanlar `result` objesine eklenir.
- Dashboard `result`'taki snake_case alanları otomatik gösterir; teste özel
  formatlı görünüm ve **statik klinik referans tablolar (normatif/MDC)** zaten
  dashboard'da (`manifest.reportReference`). Backend yalnızca **sayısal/hesaplanan**
  değerleri dönmeli.

---

## 0. Çapraz kesen (TÜM endpoint'ler)

### 0.1 🟠 UTF-8 Türkçe (md 2.1)
- `category` ve dönülen tüm metin alanları **ASCII'ye düşürülmemeli**. Şu an
  `"Sicrama/Alt-ekstremite"` dönüyor; **`"Sıçrama/Alt Ekstremite"`** olmalı.
- Tüm string çıktıları UTF-8 olmalı (ı, ş, ğ, ç, ö, ü, İ, Ş, Ğ …).

### 0.2 🔴 `test` discriminator doğruluğu (md 2.3)
- Her endpoint `result.test`'i **kendi test id'siyle** dönmeli. (Geçmişte Single
  Leg Hop raporu "broad-jump" olarak üretiliyordu — tüm endpoint'lerde teyit et.)

### 0.3 Bacak-bazlı testler + LSI sözleşmesi (md 2.3, 3.9, 3.10, 3.11, 3.15)
LSI/asimetri için **iki bacağın** ölçümü gerekir. Karar verilmeli:
- **Öneri:** Her `/analyze/<slug>` çağrısı tek bacak ölçer ve `result.side`
  ("left"/"right") + ölçümü döner. **LSI = (Etkilenmiş / Sağlıklı) × 100** iki
  bacak sonucu birleştiğinde hesaplanır (raporlama anında; dashboard veya
  backend bir "paired" endpoint ile). Hangi tarafın "etkilenmiş" olduğu doktor
  girdisidir.
- İlgili testler: Single Leg Hop, Single Leg CMJ, Squat Jump (tek bacak), 505.

---

## 1. Hesaplama hataları (DÜZELT — KRİTİK)

### 1.1 SEBT kompozit skor (md 3.5)
Mevcut (yanlış): `(3 yön toplamı) / (3 × boy) × 100`.
**Doğru:**
```
Kompozit Skor = (ANT + PM + PL) / (3 × bacak_uzunluğu) × 100
```
- ANT=Anterior, PM=Posteromedial, PL=Posterolateral.
- **Bacak uzunluğu** = ASIS → medial malleol (cm). **Boy DEĞİL.**
- Geçersiz deneme koşulları (4 koşuldan biri → deneme iptal): destek ayağının
  yerinden oynaması, uzanan ayağın yere basıp destek alması, eli yere/dizine
  dayama, başlangıç pozisyonuna kontrollü dönememe. (Tam liste doc 3.5 B'de.)

---

## 2. Teste özel hesaplanan `result` alanları (EKLE)

### 2.1 🔴 Single Leg Hop — `/analyze/single-leg-hop` (md 2.3, 3.11)
Şu an tek bacak: `distance_m, flight_time_ms, takeoff_side, landing_side` döner.
- **EKLE:** `side` (ölçülen bacak), bacak-bazlı mesafe.
- **LSI** = (Etkilenmiş bacak mesafesi / Sağlıklı bacak mesafesi) × 100 (bkz. 0.3).
- Ölçülen parametre tanımı: kalkış çizgisinden **topuğun ilk temas noktasına**
  kadar yatay mesafe.

### 2.2 30 m Sprint — `/analyze/30m` (md 3.1)
- `split_0_10_s`, `split_10_30_s` (bölünmüş zamanlar).
- `peak_velocity_ms` + `peak_velocity_at_m` (hangi metrede).
- `acceleration_phase_decay` (hızlanma eğrisinin lineer kaldığı süre/mesafe).
- **Hız-zaman eğrisi verisi** (velocity-time curve) — grafik için zaman serisi
  (örn. `velocity_series: [{t, v}]`).

### 2.3 5-10 m Sprint — `/analyze/5-15` (slug korunur, md 3.3)
- `split_0_5_s`, `split_5_10_s`, `start_acceleration_ms2`, `speed_change_ratio_5_10`.
- (Not: fiziksel kurulum 0-10 m, 5 m split. `finish_x` artık 10 m.)

### 2.4 10MWT — `/analyze/10mwt` (md 3.2)
- `gait_speed_ms` (var) + `step_length`, `cadence`, `knee_flexion_swing_deg`,
  `ankle_dorsiflexion_heelstrike_deg` (pozdan açısal/gözlemsel).

### 2.5 10-20-30 m — `/analyze/10-20-30m` (md 3.14)
- `split_0_10_s`, `split_10_20_s`, `split_20_30_s`, `total_30m_s`.
- (Normatif/MDC tabloları dashboard'da; backend sadece süreleri döner.)

### 2.6 LESS — `/analyze/less` (md 3.4)
- Toplam skor **0-17** (yüksek = kötü iniş). Sınıflandırma: ≤5 mükemmel, 6-7
  ortalama, ≥8 yüksek spor riski.
- 4 kategori alt-skoru: (a) sagittal açılar/hareket kalitesi, (b) frontal denge,
  (c) ayak-diz-kalça segmental dizilim, (d) genel iniş kontrolü.
- Platform referansı **20 cm**.

### 2.7 CMJ — `/analyze/cmj` (md 3.6) — yalnız hesaplama
- `jump_height_cm` (h = g·t²/8 veya kalça dikey deplasmanı).
- `flight_time_ms`, `countermovement_depth_cm`, `rsi_mod` (jump height / time-to-takeoff).

### 2.8 Bosco 60 sn — `/analyze/bosco-60` (md 3.7) — yalnız hesaplama
- `total_jump_count`, `avg_flight_time_ms`, `avg_contact_time_ms`,
  `fatigue_index_pct` (ilk 30 sn vs son 30 sn düşüş %), `mean_power_wkg` (Bosco formülü).

### 2.9 Squat Jump — `/analyze/squat-jump` (md 3.9)
- `sj_height_cm`, `cmj_sj_diff_cm` (CMJ ile yapıldıysa),
  `ssc_usage_pct = (CMJ−SJ)/SJ × 100`, `power_estimate_wkg` (Bosco/uçuş süresi),
  `lsi_pct` (tek bacak), `invalid_trial_count`, gözlemsel `knee_valgus_deg`.
- Geçersiz deneme tespiti: karşı hareket, eller belden kalkma, tek ayak iniş,
  diz valgus/varus >10°.

### 2.10 Single Leg CMJ — `/analyze/single-leg-cmj` (md 3.10)
- `side`, `jump_height_cm`; LSI iki bacak ile (bkz. 0.3).

### 2.11 Modifiye 505 — `/analyze/5-0-5` (slug korunur, md 3.15)
- `mod505_time_s` (başlangıç kapısından çıkıp dönüp aynı kapıdan geçme süresi).
- **COD Deficit** = `mod505_time_s − 10m_düz_koşu_süresi` (saf dönüş becerisi).
- `left_right_asymmetry_pct` (sağ/sol dönüş süreleri farkı).
- Dinamik biyomekanik: dönüşte `knee_valgus_deg`, gövde lateral komponenti,
  frenlemede COM alçaltma kalitesi.

---

## 3. Öncelik sırası
1. **1.1 SEBT formülü** (KRİTİK, klinik yanıltıcı).
2. **0.1 UTF-8** + **0.2 test id** doğrulaması (tüm endpoint'ler).
3. **2.1 Single Leg Hop LSI** + **0.3 bacak-bazlı sözleşme** (505, SL-CMJ, SJ ile birlikte tasarlanmalı).
4. Sprint split/velocity (2.2, 2.3, 2.5), CMJ/Bosco/SJ (2.7-2.9), 10MWT açısal (2.4), LESS (2.6), 505 (2.11).

## 4. Dashboard tarafı (referans — zaten yapıldı)
- Statik klinik referans (normatif/MDC/sınıflandırma) tabloları `manifest.reportReference`
  içinde, hem sonuç sayfası hem PDF'te render ediliyor. Backend bu tabloları
  DÖNMEZ; yalnızca yukarıdaki **hesaplanan** değerleri döner.
- Yeni `result` alanları snake_case geldiğinde dashboard otomatik gösterir;
  özel formatlı görünüm gerekiyorsa ayrıca eklenir.
