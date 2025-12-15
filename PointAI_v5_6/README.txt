POINTAI v5 – OUT-OF-CORE (Zarr tiles + LOD) + Viewer

INSTALL
  python -m venv .venv
  .venv\Scripts\activate
  pip install -r requirements.txt

RUN
  python main.py

OUT-OF-CORE
- Crea uno store su disco (cartella *.zarr) con tiles 3D + LOD.
- Carica solo una ROI (region of interest) invece di tutta la nuvola.
- Editing non distruttivo: scrive operazioni in ops.json (applicate al volo).

WORKFLOW
1) Tab "Out-of-core (tiles/LOD)" -> "Crea Store (Zarr) da file sorgente"
2) Apri lo store (auto dopo build)
3) Imposta ROI center + radius, scegli LOD
4) "Carica ROI"
5) (Opzionale) "Edit: rimuovi bbox (ROI)" -> salva in ops.json
6) "Export PLY (filtri applicati)"

LIMITI ATTUALI
- Build store usa un ingest a campione (max_ingest). Per full-res serve una pipeline tile-first per ciascun formato.
- E57: pye57 non espone chunk; ingest è campionato.

FIX v5_1: compatibilità Zarr v3 (create_dataset richiede shape/dtype).

FIX v5_2: corretto indentation di oc_store.write_tile.


FIX v5_4: Zarr v3 codec mismatch -> requirements pin zarr<3. If you already have zarr 3.x installed, run: pip install "zarr<3" --upgrade
