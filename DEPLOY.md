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
