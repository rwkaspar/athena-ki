# Athena-Backup

Folgt dem kaspar-family-Backup-Muster (DealMonitor NAS-Receiver, rsync-Daemon
über Tailscale, Port 8873). Da Athena kein Postgres hat, werden Daten-Verzeichnisse
als `tar.gz` gesichert statt `pg_dump`.

## Was wird gesichert
- `submissions/`, `contact/`, `logs/` — **unwiederbringlich** → jede Tier-Stufe (auch hourly).
- `athena-db/` (ChromaDB, ~1,1 GB, rebuildbar) → erst ab `daily`.
- NICHT: `api.key` (regenerierbares Secret), git-getrackte Inhalte (Programm/Code/Config liegen auf GitHub).

## Mechanik
`scripts/backup_athena.sh <hourly|daily|weekly|monthly>`: tar+gzip → `gzip -t`-Verify →
lokale Retention → rsync-Push zum NAS. Getriggert von systemd-User-Timern:
`athena-backup-{hourly,daily,weekly,monthly}.timer` (Service-Template `athena-backup@.service`).

Zeitplan: hourly alle 4h · daily 04:00 · weekly Mo 03:00 · monthly 1. 02:00 (UTC).
Lokale Retention: hourly 24 · daily 3 · weekly 2 · monthly 2 (NAS hält die Tiefe).

## NAS aktivieren
`.backup.env` (gitignored) aus `.backup.env.example` anlegen:
```
NAS_RSYNC_TARGET=rsync://backup@<nas-tailscale-ip>:8873/backups/athena
RSYNC_PASSWORD=<gleiches-passwort-wie-auf-dem-nas>
```
NAS-Receiver: entweder den vorhandenen DealMonitor-Receiver mitnutzen (Unterordner
`athena/`) oder einen eigenen aufsetzen (siehe Outline „Backup-Setup für ein neues Projekt").

## Restore
```
tar -tzf athena_daily_<ts>.tar.gz          # Inhalt prüfen
# Dienste stoppen, dann entpacken:
systemctl --user stop athena-uvicorn athena-chroma
tar -xzf athena_daily_<ts>.tar.gz -C /home/robert/athena-ki
systemctl --user start athena-chroma athena-uvicorn
```
ChromaDB wird heiß getart — im Zweifel ist die DB aus den Quelldokumenten neu
ingestierbar; `submissions/contact/logs` sind die kritischen, nicht-rebuildbaren Daten.

## Smoke-Test NAS
```
RSYNC_PASSWORD=<pw> rsync -av /tmp/test.txt rsync://backup@<nas-ip>:8873/backups/athena/hourly/
```
