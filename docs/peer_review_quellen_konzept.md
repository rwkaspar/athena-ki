# Peer-Review für Quellen — Konzept

*Status: Entwurf zur Diskussion · Stand 2026-06-04 · für Outline*

## Ziel

Die Verifizierung von Quellen für Athenas Wissensbasis von einem **Einzelperson-CLI**
(`review_submissions.py`, nur Robert) auf ein **mehrstufiges, transparentes Peer-Review**
umstellen. Quellen werden erst dann verifiziert (Tier 1–3) und für Athenas Antworten
nutzbar, wenn das Review sie bestätigt. Bis dahin bleiben sie Tier 0 (unverifiziert,
standardmäßig aus Antworten ausgeschlossen).

**Doppelter Gewinn:** Wenn die Freigabe über die API (uvicorn) statt über separate
CLI-Prozesse läuft, schreibt nur *ein* Prozess in ChromaDB → das Concurrency-Problem,
das am 2026-06-04 die `bund_fresh`-Collection beschädigt hat, ist strukturell gelöst.

## Prinzipien (EVIDENZ-Wertekanon)

- **§7 Faktentransparenz**: Jede Verifizierung ist öffentlich nachvollziehbar — wer,
  wann, mit welcher Begründung, welcher Tier.
- **§5 Wissenschaftlichkeit**: Athenas Auto-Bewertung (Herausgeber-Vertrauen, Relevanz,
  Tier-Vorschlag) ist dokumentierter *Input*, ersetzt aber kein menschliches Urteil.
- **Kein Einzelgatekeeper**: Entscheidungen sollen von mehreren getragen werden, sobald
  es mehrere Reviewer gibt.

---

## Frage 1 — Reviewer-Kreis: Empfehlung *phasiert*

Du tendierst zur **offenen Community** (passt zum Konzept) — dem stimme ich als *Endziel*
zu, aber **nicht als Startpunkt**:

- Offene Community braucht **Reputations-, Moderations- und Missbrauchsschutz** (Sockenpuppen,
  koordinierte Manipulation, Astroturfing) — viel Bau, und bei einer politischen Partei ein
  echtes Angriffsziel.
- Aktuell gibt es **noch keine Reviewer-Basis** (du bist allein) — eine Community-Mechanik
  liefe ins Leere.

**Empfehlung: in drei Phasen, Datenmodell von Anfang an community-fähig.**

| Phase | Reviewer | Wann |
|---|---|---|
| **1 — Vertrauenskreis** | Gründungsteam + eingeladene Fachleute (Allowlist) | MVP, jetzt |
| **2 — Erweitert** | + verifizierte Unterstützer:innen mit Einladung | wenn Team wächst |
| **3 — Offene Community** | jede:r Registrierte, mit Reputation & Moderation | wenn Basis + Schutz steht |

Rollen (`reviewer`, `trusted_reviewer`, `moderator`) und Reputations-Felder schon in Phase 1
im Modell anlegen → Phase 3 ohne Umbau.

---

## Frage 2 — Konsens-Regel: *schwellenbasiert, skaliert mit dem Team*

Problem: „Fast-Track + 2 bei Streit" oder „immer 2 OKs" funktioniert nicht, solange du
**allein** bist. Lösung: die Regel an die **Zahl aktiver Reviewer** koppeln.

```
benötigte_OKs = min(KONSENS_ZIEL, max(1, anzahl_aktiver_reviewer - eigen))
```

- **Solo (jetzt):** 1 OK genügt — aber mit vollem Audit-Log, und **Athenas Auto-Bewertung
  zählt als dokumentierte „beratende Zweitstimme"** (so hat selbst die Solo-Entscheidung eine
  zweite, festgehaltene Perspektive).
- **Ab 3+ Reviewern:** Ziel `KONSENS_ZIEL = 2` greift automatisch → Fast-Track (Athena-approve
  + 1 Mensch) für klare Fälle, **2 unabhängige Menschen** bei Streit (irgendein Reject oder
  Athena „needs_human").
- **Reject mit Begründungspflicht**; eigene Einreichungen kann man nicht selbst freigeben.

So musst du heute nichts künstlich blockieren, und die Strenge wächst automatisch mit dem Team.

---

## Frage 3 — Auth-Flow: konkret

### Variante A — Magic-Link (empfohlen)

SMTP existiert bereits (Kontaktformular) → kein Passwort-Management nötig.

```
1. Reviewer öffnet  https://evidenz-partei.de/review
2. Gibt E-Mail ein  →  POST /review/login {email}
3. Server prüft E-Mail gegen Allowlist (Phase 1) bzw. Registrierung (Phase 3)
4. Server mailt Einmal-Link:  /review/verify?token=<signiert, 15 Min gültig>
5. Reviewer klickt → Server prüft Token → setzt Session-Cookie (HttpOnly, 30 Tage)
6. Reviewer sieht die Warteschlange, stimmt ab → Stimme ist an die Identität gebunden
```

- Token = signiert (HMAC mit Server-Secret) + Ablauf → kein DB-Eintrag pro Link nötig.
- Funktioniert für Vertrauenskreis **und** spätere Community (dann Selbst-Registrierung statt
  Allowlist).

### Variante B — Persönliche Token-URL (simpler, nur Phase 1)

```
Jede:r Reviewer:in bekommt eine geheime URL:  /review?key=<zufalls-token>
Server-Map  key → {reviewer_id, name}.  Kein Mailflow.
```

Schnellster Start, aber: Link = Zugang (teilbar/leakbar), schlecht für viele/Community.

**Empfehlung:** Magic-Link von Anfang an — minimal mehr Aufwand, aber tragfähig bis Community.
Server-Secret + Allowlist liegen in der `.env` (nie ins Repo).

---

## Architektur

### Datenmodell (Erweiterung der `submission/meta.json`)

```jsonc
{
  "...": "bestehende Felder (url, auto_review, scope, …)",
  "reviews": [
    { "reviewer_id": "…", "name": "…", "vote": "approve|reject",
      "tier": 2, "comment": "…", "at": "2026-…Z" }
  ],
  "review_status": "open | verified | rejected",
  "decided_at": "…", "decided_tier": 2
}
```

### API-Endpoints (alle in uvicorn = einziger ChromaDB-Writer)

| Methode | Pfad | Zweck |
|---|---|---|
| POST | `/review/login` | Magic-Link anfordern |
| GET | `/review/verify` | Token einlösen → Session |
| GET | `/review/queue` | offene Submissions + eigene/fremde Stimmen (auth) |
| POST | `/review/vote` | Stimme abgeben; bei erreichtem Konsens: Tier-Hochstufung bzw. Löschen — **serverintern** |
| GET | `/review/log` | öffentliches Audit-Protokoll |

### Web-UI `/review` (geschützt)

- Warteschlange mit Athenas Vorbewertung, Quellen-Vorschau (Titel/URL/Auszug/Tags).
- Buttons: **approve (Tier)** / **reject (Grund)** / **skip**.
- Fremde Stimmen erst **nach eigener Abstimmung** sichtbar (gegen Anchoring).
- Filter nach Athena-Empfehlung/Tier/Thema.

### Transparenz

- Öffentliches Review-Log (erweitert `/sources`-Protokoll): jede verifizierte Quelle zeigt
  Reviewer (Name oder Pseudonym), Zeitpunkt, Tier, Kurzbegründung.
- Einreicher-Daten (IP/Mail) bleiben privat (wie bisher, Whitelist-Anonymisierung).

### Missbrauchsschutz (ab Phase 3)

- Reputationsgewichtung: neue Reviewer „on probation", Stimmen zählen erst nach Bestätigung.
- Rate-Limits pro Reviewer; Moderator kann Stimmen/Accounts sperren.
- Anomalie-Erkennung (viele Approves in kurzer Zeit, Cluster gleicher Quellen).

---

## Migration der bestehenden 428 Submissions

- **210** Athena-„approve", **20** „reject", **32** „needs_human", **166** ohne Verdict.
- Übergangsweise: das neue CLI-Batch-Tooling (`--review-pending`, `--accept-athena`,
  `--reject-athena`) bereinigt den Rückstand; der Web-Flow übernimmt ab Phase 1 den laufenden
  Betrieb.

## Aufwandsschätzung (grob)

- **Phase 1 MVP** (Magic-Link, Allowlist, Queue-UI, vote→promote in API, Solo-/Schwellen-Konsens,
  Audit-Log): überschaubar, baut auf vorhandenem Stack (FastAPI, SMTP, Caddy, ChromaDB).
- **Phase 3** (Registrierung, Reputation, Moderation, Anomalie-Erkennung): eigenes Projekt.

## Offene Punkte

- Reviewer-Pseudonymität: Klarname vs. Pseudonym im öffentlichen Log?
- Sollen Einreicher den Status ihrer Quelle sehen (Benachrichtigung bei Entscheidung)?
- Tier-Vergabe im Konsens: was, wenn zwei Reviewer unterschiedliche Tiers vergeben? (Vorschlag:
  niedrigerer/vorsichtigerer Tier gewinnt, oder Moderator entscheidet.)
