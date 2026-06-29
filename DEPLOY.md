# Google Cloud Run'a Deploy

Uygulama **scale-to-zero** calisir: istek gelmedigi surece hicbir instance ayakta
durmaz (idle maliyet ~0), analiz istegi gelince Cloud Run otomatik instance baslatir.

## Tek seferlik hazirlik

```bash
gcloud auth login
gcloud config set project <PROJE_ID>
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

## Deploy (kaynaktan — Dockerfile otomatik kullanilir)

```bash
gcloud run deploy fizyotr-backend \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --min-instances 0 \
  --max-instances 4 \
  --cpu 1 \
  --memory 512Mi \
  --concurrency 8 \
  --timeout 300
```

Bittiginde verdigi `https://fizyotr-backend-....run.app` adresine:
- `GET /app`     → tarayici arayuzu (videodan pose CSV cikarir)
- `GET /health`  → saglik kontrolu
- `POST /analyze/<test_id>` → analiz

---

# SOAPIE agent — AYRI servis (`fizyotr-soapie`)

SOAPIE oneri agent'i `fizyotr-backend`'ten **bagimsiz** ikinci bir Cloud Run
servisidir (kendi WSGI/auth/Dockerfile/requirements'i `soapie/` altinda). Build
context = `soapie/`. Claude API gerektirir; anahtar imaja GOMULMEZ, Secret
Manager'da tutulur. `soapie/.env` deploy EDILMEZ (yalnizca lokal gelistirme).

## 1) Tek seferlik — ANTHROPIC_API_KEY secret'i

```bash
gcloud services enable secretmanager.googleapis.com

# Anahtari secret olarak olustur (soapie/.env'deki degerden okur)
grep '^ANTHROPIC_API_KEY' soapie/.env | cut -d= -f2- | tr -d '"' | \
  gcloud secrets create anthropic-api-key --data-file=- --replication-policy=automatic
#   (anahtar degisirse yeni surum: ... | gcloud secrets versions add anthropic-api-key --data-file=-)

# Cloud Run runtime SA'ya secret okuma izni ver
PROJ_NUM=$(gcloud projects describe $(gcloud config get-value project) --format='value(projectNumber)')
gcloud secrets add-iam-policy-binding anthropic-api-key \
  --member="serviceAccount:${PROJ_NUM}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

## 2) Deploy (ayri servis, `--source soapie/`)

```bash
gcloud run deploy fizyotr-soapie \
  --source soapie/ \
  --region europe-west1 \
  --allow-unauthenticated \
  --min-instances 0 --max-instances 4 \
  --cpu 1 --memory 1Gi --concurrency 8 --timeout 300 \
  --set-env-vars SUPABASE_URL=https://<proje>.supabase.co \
  --update-secrets ANTHROPIC_API_KEY=anthropic-api-key:latest
```

Endpoint'ler (kendi `https://fizyotr-soapie-....run.app` adresinde):
- `GET /health`            → servis + indeks durumu
- `GET /soapie`            → indeks meta (auth'suz)
- `POST /soapie/suggest`   → SOAP onerisi (auth'lu)

> **Not:** SOAPIE (Opus + thinking) yaniti ~45 sn surebilir → `--timeout 300`.
> RAG indeksi (`soapie/data/chunks.json`, ~1 MB) imaja dahil; kaynak PDF bu
> context'te yok. `SUPABASE_URL` auth icin gerekir (ana servisle ayni proje).
> Deploy sonrasi `GET /health` ile `ready:true` dogrula. Kod degisince ayni
> komutu tekrar calistir — yeni revizyon olusur.

## Ayarlarin anlami

| Bayrak | Aciklama |
|--------|----------|
| `--min-instances 0` | Idle'da sifira iner — **sadece analiz baslatilirken calisir** (istedigin davranis). |
| `--max-instances 4` | Maliyet/yuk tavani. Yuke gore artir. |
| `--memory 512Mi` | numpy analizleri icin yeterli; cok uzun video CSV'lerinde `1Gi` yap. |
| `--concurrency 8` | Bir instance'in es zamanli istek sayisi. Analiz CPU-yogun; dusuk tutmak daha akici. |
| `--timeout 300` | Istek zaman asimi (sn). Analizler kisa; gerekirse artir. |
| `--allow-unauthenticated` | API/arayuz herkese acik. Kapatmak istersen kaldir (IAM ile eris). |

> **Not — istek boyutu:** Cloud Run HTTP/1'de istek govdesi ~32 MiB ile sinirli.
> 60 sn'lik bir videonun pose CSV'si tipik olarak ~4 MB, sorun olmaz. Cok uzun
> kayitlarda limit asilirsa Cloud Run'da HTTP/2 (`--use-http2`) acilabilir.

## Soguk baslangic (cold start)

Scale-to-zero geregi ilk istek instance ayaga kalkarken ~birkac saniye gecikir.
Imaj sade (yalnizca flask+numpy) oldugu icin cold start hizlidir. Gecikme istemezsen
`--min-instances 1` yap (ama o zaman surekli 1 instance acik kalir, idle maliyet doger).

## Guncelleme

Kod degisince ayni `gcloud run deploy ... --source .` komutunu tekrar calistir;
yeni revizyon olusur, trafik otomatik ona gecer.

## CI/CD (opsiyonel)

`main`'e her push'ta otomatik deploy icin:

```bash
gcloud run deploy fizyotr-backend --source . --region europe-west1 \
  --allow-unauthenticated --min-instances 0
```
komutunu bir Cloud Build trigger veya GitHub Actions adimina koy.
