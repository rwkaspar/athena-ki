# Minimaler, netzwerkloser Sandbox-Container nur zum PDF→Text-Parsen.
# Wird zur Laufzeit mit --network none --read-only --cap-drop ALL als nobody
# gestartet; das hier installiert nur den Parser + pypdf.
FROM python:3.12-slim

# pypdf-Version pinnen (reproduzierbar, gleicher Parser wie im Hauptsystem)
RUN pip install --no-cache-dir "pypdf==6.10.2"

# Parser-Skript einbacken
COPY extract.py /opt/extract.py

# nobody — kein Schreibzugriff, kein Root
USER 65534:65534

ENTRYPOINT ["python", "/opt/extract.py"]
