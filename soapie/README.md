# SOAPIE Öneri Agent'ı — BAĞIMSIZ servis

Ana analiz servisinden (`app.py` / `fizyotr-backend`) **AYRI** bir uygulamadır:
kendi WSGI giriş noktası, kendi auth'u, kendi Dockerfile + requirements; ayrı bir
Cloud Run servisi (`fizyotr-soapie`) olarak deploy edilir. Bağımsız ölçekleme,
bellek ve timeout alır.

Terapist **S (Subjektif)** oyküyü girer; agent, muskuloskeletal muayene
kitabından **RAG** ile bağlam çekerek **O (Objektif muayene planı)** ve
**A (Değerlendirme / olası tanı)** önerir, **P (Plan)** için yalnızca
test/görüntüleme önerisi verir + tedavi egzersizleri için harici kaynak
gerektiğini belirtir. Çıktı her zaman **taslaktır** ve fizyoterapist onayı gerektirir.

## RAG çeşidi: Lexical BM25 + LLM cross-lingual sorgu çıkarımı

1. **Sorgu çıkarımı** — Claude, Türkçe subjektiften İngilizce klinik terimleri
   çıkarır (`QueryTerms`). Türkçe girdi ↔ İngilizce kaynak dil farkını köprüler.
2. **Retrieval** — `rank-bm25` ile `data/chunks.json` üzerinde lexical arama
   (embedding / vektör DB / harici API yok → hafif, Cloud Run dostu).
3. **Sentez** — Claude, **yalnızca getirilen sayfalara** dayanarak SOAP önerisi
   üretir (`SoapieSuggestion`), sayfa atıflı.

## Dosyalar

| Dosya | Görev |
|---|---|
| `wsgi.py` | Bağımsız Flask uygulaması (`create_app()` + `app`); gunicorn giriş noktası |
| `auth.py` | Kendi Supabase JWT auth'u (`require_auth`, AUTH_DISABLED destekli) |
| `routes.py` | `register(app, auth)` → `GET /soapie`, `POST /soapie/suggest` |
| `agent.py` | `suggest()` pipeline; `ANTHROPIC_API_KEY` env'den okunur |
| `retrieval.py` | BM25 indeks (lazy singleton) + `search()` |
| `schema.py` | Structured output Pydantic modelleri (`QueryTerms`, `SoapieSuggestion`) |
| `prompts.py` | Türkçe klinik system prompt'lar |
| `requirements.txt` | Bu servisin bağımlılıkları (flask, PyJWT, anthropic, rank-bm25; numpy yok) |
| `Dockerfile` | Bağımsız imaj (build context = `soapie/`) |
| `tools/build_index.py` | PDF → `data/chunks.json` (offline derleme) |
| `data/chunks.json` | Önceden derlenmiş chunk'lar (runtime'da pypdf gerekmez) |

## Endpoint'ler

| Rota | Auth | Açıklama |
|---|---|---|
| `GET /health` | hayır | Servis + indeks durumu |
| `GET /soapie` | hayır | Indeks meta + kullanım |
| `POST /soapie/suggest` | evet | `{"subjective":"...", "patient":{...}, "k":8}` → SOAP önerisi |

## Lokal çalıştırma

```bash
# İndeksi yeniden derle (PDF değişirse):
python -m soapie.tools.build_index

# Servisi çalıştır (repo kökünden; soapie bir pakettir):
ANTHROPIC_API_KEY=sk-... AUTH_DISABLED=1 python -m soapie.wsgi    # :8081
# veya gunicorn:
ANTHROPIC_API_KEY=sk-... AUTH_DISABLED=1 gunicorn -b :8081 soapie.wsgi:app

# İstek:
curl -X POST http://localhost:8081/soapie/suggest \
  -H "Content-Type: application/json" \
  -d '{"subjective":"Sag dizde 2 haftadir agri, merdiven inerken artiyor, sislik var.","patient":{"age":34}}'
```

## Cloud Run deploy (ayrı servis)

```bash
# ANTHROPIC_API_KEY Secret Manager'da (bkz. kök DEPLOY.md), sonra:
gcloud run deploy fizyotr-soapie \
  --source soapie/ \
  --region europe-west1 \
  --allow-unauthenticated \
  --min-instances 0 --max-instances 4 \
  --cpu 1 --memory 1Gi --concurrency 8 --timeout 300 \
  --set-env-vars SUPABASE_URL=https://<proje>.supabase.co \
  --update-secrets ANTHROPIC_API_KEY=anthropic-api-key:latest
```

- `--source soapie/`: build context = bu klasör; `soapie/Dockerfile` kullanılır.
- `data/chunks.json` imaja dahildir; kaynak PDF (`documents/`) bu context'te yok.
- Auth için `SUPABASE_URL` gerekir (ana servisle aynı Supabase projesi).
- `P` (tedavi/egzersiz) için ileride harici tedavi kaynağı ile ikinci RAG indeksi eklenebilir.
