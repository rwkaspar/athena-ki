#!/usr/bin/env bash
# Athena-Backup → Synology-NAS (rsync-Daemon über Tailscale), getiered.
# Folgt dem kaspar-family-Backup-Muster (DealMonitor NAS-Receiver, Port 8873).
#
# Unterschied zu DealMonitor: Athena hat KEIN Postgres. Gesichert werden die
# Daten-Verzeichnisse als tar.gz:
#   - submissions/  contact/  logs/   (UNwiederbringlich → jede Tier-Stufe)
#   - athena-db/    (ChromaDB, ~1,1 GB, rebuildbar → erst ab 'daily')
#
# Aufruf:  backup_athena.sh <hourly|daily|weekly|monthly>
# Env (aus .backup.env):  NAS_RSYNC_TARGET, RSYNC_PASSWORD
set -euo pipefail

TIER="${1:?Tier fehlt: hourly|daily|weekly|monthly}"
case "$TIER" in hourly|daily|weekly|monthly) ;; *) echo "ungültiger Tier: $TIER" >&2; exit 2;; esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # athena-ki root
ENV_FILE="${ATHENA_BACKUP_ENV:-$HERE/.backup.env}"
STAGE="${ATHENA_BACKUP_DIR:-$HERE/backups/staging}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

# Env laden (NAS-Ziel + Passwort) — optional; ohne läuft nur das lokale Backup.
[ -f "$ENV_FILE" ] && { set -a; . "$ENV_FILE"; set +a; }

mkdir -p "$STAGE/$TIER"

# Was kommt rein? Kleindaten immer, große ChromaDB erst ab daily.
INCLUDE=(submissions contact logs)
[ "$TIER" != "hourly" ] && INCLUDE+=(athena-db)
PATHS=(); for p in "${INCLUDE[@]}"; do [ -e "$HERE/$p" ] && PATHS+=("$p"); done
[ ${#PATHS[@]} -eq 0 ] && { echo "[backup] nichts zu sichern"; exit 0; }

ARCHIVE="$STAGE/$TIER/athena_${TIER}_${TS}.tar.gz"
echo "[backup] $TIER → $(basename "$ARCHIVE")  (Pfade: ${PATHS[*]})"
# Hinweis: ChromaDB wird heiß getart (Server läuft). Writes sind selten; die DB
# ist aus den Quelldokumenten rebuildbar. submissions/contact/logs sind klein/atomar.
tar -czf "$ARCHIVE" -C "$HERE" "${PATHS[@]}"
gzip -t "$ARCHIVE"
echo "[backup] verify OK ($(du -h "$ARCHIVE" | cut -f1))"

# Lokale Retention (NAS hält die Tiefe; lokal nur Schnellzugriff).
case "$TIER" in hourly) KEEP=24;; daily) KEEP=3;; weekly) KEEP=2;; monthly) KEEP=2;; esac
ls -1t "$STAGE/$TIER"/athena_${TIER}_*.tar.gz 2>/dev/null | tail -n +$((KEEP+1)) | xargs -r rm -f

# Push zum NAS (rsync-Daemon-Protokoll, kein SSH).
if [ -n "${NAS_RSYNC_TARGET:-}" ] && [ -n "${RSYNC_PASSWORD:-}" ]; then
  RSYNC_PASSWORD="$RSYNC_PASSWORD" rsync -a --timeout=120 \
    "$ARCHIVE" "${NAS_RSYNC_TARGET%/}/$TIER/"
  echo "[backup] ✅ Synced to NAS ($TIER)"
else
  echo "[backup] ⚠️ NAS nicht konfiguriert (.backup.env) — nur lokal gesichert."
fi
