# Upload-Härtung — Docker-Infrastruktur

Die Quelleneinreichung (`POST /submissions`, `scripts/serve.py`) härtet Datei-Uploads
über drei Schichten. Zwei davon brauchen Docker-Container auf dem Host `athena`
(robert ist in der `docker`-Gruppe, kein sudo nötig). Logik: `scripts/upload_security.py`.

| Schicht | Was | Komponente |
|---|---|---|
| 1. Virenscan | Rohe Upload-Bytes per ClamAV (INSTREAM) prüfen, Funde ablehnen | resident `athena-clamd`-Container |
| 2. Sandbox-Parsing | PDFs **nicht** im Serverprozess parsen, sondern im wegwerfbaren Container ohne Netz/Caps | `athena-pdf-sandbox`-Image |
| 3. PDF→Text-only | Nur den Text extrahieren, das binäre Original verwerfen (nie gespeichert) | `extract.py` in der Sandbox |

## 1. ClamAV (resident clamd)

```bash
docker volume create clamav-db   # Virendefinitionen persistent + freshclam-Auto-Update
docker run -d --name athena-clamd --restart unless-stopped \
  -p 127.0.0.1:3310:3310 \
  -v clamav-db:/var/lib/clamav \
  -e CLAMAV_NO_MILTERD=true \
  clamav/clamav:latest
```

Erster Start lädt die Definitionen (~ein paar Minuten). Status: `docker ps`
zeigt `healthy`. Health-Check der App: `GET /health` → `{"clamd":"up"}`.

## 2. Sandbox-Parser-Image

```bash
docker build -t athena-pdf-sandbox:latest docker/pdf-sandbox/
```

Wird pro PDF kurzlebig gestartet mit: `--network none --read-only --user 65534
--cap-drop ALL --security-opt no-new-privileges --memory 512m --pids-limit 64`.
Die Datei kommt nur über stdin herein, der Text geht über stdout zurück.

## Konfiguration (Env, optional)

| Var | Default | Zweck |
|---|---|---|
| `ATHENA_CLAMD_HOST` / `ATHENA_CLAMD_PORT` | `127.0.0.1` / `3310` | clamd-Adresse |
| `ATHENA_PDF_SANDBOX_IMAGE` | `athena-pdf-sandbox:latest` | Sandbox-Image |
| `ATHENA_SANDBOX_TIMEOUT_S` | `60` | Zeitlimit Parsing |
| `ATHENA_REQUIRE_SCAN` | `1` | fail-closed: clamd nicht erreichbar → Upload abgelehnt (statt ungescannt durchzulassen) |

## Hinweis: Geltungsbereich

Gehärtet ist der **Datei-Upload**. URL-Einreichungen, die auf ein PDF zeigen,
werden weiterhin von `auto_review.py`/`ingest.py` direkt geladen und in-process
geparst — anderer Vektor, hier bewusst nicht abgedeckt.
