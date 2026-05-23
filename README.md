# Sportif Performans Testleri — CSV girisli Flask API

Pose tahmini **tarayicida** (MediaPipe.js) yapilir; sunucu yalnizca pose
noktalarinin **CSV**'sini alir, ilgili analizciyi calistirir ve sonucu JSON
doner. Boylece sunucuda `cv2` / `mediapipe` / model dosyalari **gerekmez** —
sadece `flask` + `numpy`.

19 test desteklenir (FMS 7, sicrama/alt-ekstremite 8, denge 2, yuruyus 1, ROM 1).

## Klasor yapisi

```
zipzip-flask/
├── app.py                # Flask API (giris noktasi, CSV girisli)
├── requirements.txt      # flask, gunicorn, numpy
├── web/index.html        # tarayici arayuzu (videodan pose CSV cikarir)
└── src/
    ├── pose_common.py    # iter_pose / iter_pose_csv + geometri
    ├── fms/              # FMS 7 testi + common
    ├── prod/             # squat/broad/single-leg sicramalar
    └── tests/            # cmj, less, bosco, rsi, flamingo, frt, 10mwt, rom
```

## Kurulum & calistirma

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python app.py                       # http://localhost:8080
# production:
gunicorn -b :8080 app:app
```

Tarayicida arayuzu ac: **http://localhost:8080/app**
Test sec → boy (height_m) gir → video sec → "CSV cikar & gonder". Pose tarayicida
cikarilir, CSV sunucuya gider, sonuc ekranda gosterilir. (Arayuz MediaPipe'i
CDN'den ceker; internet gerekir.)

## API

| Endpoint | Aciklama |
|----------|----------|
| `GET /`        | servis bilgisi + test listesi |
| `GET /app`     | tarayici arayuzu (HTML) |
| `GET /health`  | saglik kontrolu |
| `POST /analyze/<test_id>` | `csv=<pose.csv>` + form parametreleri → sonuc JSON |

```bash
curl -F csv=@pose.csv -F height_m=1.80 localhost:8080/analyze/squat-jump
curl -F csv=@pose.csv -F joint=knee     localhost:8080/analyze/rom
```

Ortak parametreler: `height_m` (FMS'de zorunlu), teste ozel (`painful`, `board`,
`duration`, `joint`, `side`, `distance`, `start_x`, `finish_x`).

### CSV bicimi

Header zorunlu, kare basina bir satir:

```
frame,time_s,width,height,fps, x0,y0,z0,v0 ... x32,y32,z32,v32, wx0,wy0,wz0 ... wx32,wy32,wz32
```

- `x*,y*` normalize (0..1), `z*` derinlik, `v*` visibility (MediaPipe 33 nokta)
- `w*` world koordinat (metre)
- Pose bulunmayan karede landmark hucreleri bos birakilir

CSV'yi tarayici arayuzu uretir; elle de uretebilirsiniz (yukaridaki sutun duzeni).
