"""
main.py — Pipeline complet de correction de transcription :

"Prétraitements" (OCR PDF + nettoyage transcription brute)
puis "Import & Analyse" (STS → Hapax → Scènes [→ Tirets optionnel]).

Détection automatique du travail à faire :
- gold : un .pdf déclenche l'OCR (Mistral) pour générer le fichier de
  référence ; un .json est utilisé tel quel.
- transcript : un fichier sans 'repliques' (mots seuls) déclenche le
  prétraitement LLM (regroupement en phrases) ; un fichier déjà propre
  ('repliques' présent) est utilisé tel quel.

Usage :
    python main.py fichier_gold.json fichier_transcript.json --port XXXX --asid XXXX [--api-key XXXX] [--tirets] [--architecture daia|otto] [-o output_name.json]
    python main.py scenario.pdf transcript_brut.json --architecture otto
"""

import argparse
import getpass
import io
import json
import logging
import os
import sys
import time
import requests
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from hapax_analyzer import HapaxAnalyzer
from pretraitements import get_mistral_client, run_ocr_pipeline, run_preprocessing_llm
from scenes_analyzer import ScenesAnalyzer
from sts_analyzer import STSAnalyzer
from tiret_option import TiretOption
from transcript_loader import load_otto, export_daia, export_otto

# Les analyzers appellent Streamlit (st.spinner/st.expander/...) en interne ;
# hors d'un `streamlit run`, Streamlit se contente de logger un avertissement
# "missing ScriptRunContext" à chaque appel. On relève son niveau de log pour
# ne garder que nos propres messages de progression dans le terminal.
logging.getLogger("streamlit").setLevel(logging.ERROR)


def log(etape_nom: str, message: str) -> None:
    """Message de progression façon logs de requêtes API : horodatage,
    étape, message."""
    horodatage = datetime.now().strftime("%H:%M:%S")
    print(f"[I2S] [{horodatage}] {etape_nom:<10} {message}", flush=True)


# Suivi d'avancement auprès du logiciel externe (no-op si --port n'est pas
# fourni : PORT reste None et report_progress/report_error ne font rien).
WORKERNAME = "scenarioworker"
PORT = None
ASID = None

def api_post_json_data(port, api, json_string):
    print(json_string)
    url = f"http://127.0.0.1:{port}{api}"

    print(f"PORT = {port}")
    print(f"API = {api}")
    print(f"URL = {url}")

    try:
        response = requests.post(
            url,
            data=json_string,
            headers={
                "Content-Type": "application/json"
                }
            )

        response.raise_for_status()

        print(f"HTTP CODE = {response.status_code}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"POST failed: {e}")
        return False


def report_progress(pc: int, description: str, endofprocess: bool = False) -> None:
    """POST /workers/{WORKERNAME} — signale l'avancement (pourcentage +
    description)"""
    if PORT is None:
        return
    api_post_json_data(PORT, f"/workers/{WORKERNAME}", json.dumps({
        "asid": ASID,
        "pid": os.getpid(),
        "workername": WORKERNAME,
        "pc": pc,
        "endofprocess": endofprocess,
        "description": description,
    }))


def report_error(errorcode: int, errorlib: str) -> None:
    """POST /workers/error — signale un échec"""
    if PORT is None:
        return
    api_post_json_data(PORT, "/workers/error", json.dumps({
        "asid": ASID,
        "pid": os.getpid(),
        "workername": WORKERNAME,
        "errorcode": errorcode,
        "errorlib": errorlib,
    }))


@contextmanager
def etape(nom: str, requete: str, pc: int = None):
    """Encadre une étape du pipeline avec des messages façon requête API
    (→ départ, ← fin avec durée écoulée). Redirige aussi stderr vers
    /dev/null le temps de l'étape : les analyzers réutilisés depuis
    app.py appellent Streamlit en interne (st.spinner, st.dataframe...),
    qui hors contexte d'app y imprime des avertissements sans rapport
    avec l'exécution ('missing ScriptRunContext', dépréciation de
    use_container_width). Nos propres logs passent par stdout et
    restent visibles."""
    log(nom, f"→ {requete}")
    if pc is not None:
        report_progress(pc, requete)
    debut = time.time()
    stderr_original = sys.stderr
    with open(os.devnull, "w") as silence:
        sys.stderr = silence
        try:
            yield
        finally:
            sys.stderr = stderr_original
    duree = time.time() - debut
    log(nom, f"← 200 OK ({duree:.1f}s)")


def nom_sortie(file_name: str) -> str:
    """Nom de base pour les fichiers générés : extension d'origine retirée
    (ex. '.pdf'), espaces remplacés par des underscores. Repris tel quel
    de app.py."""
    return Path(file_name).stem.replace(" ", "_")


def _obtenir_fichier(source):
    """Retourne un objet fichier prêt à être passé à un analyzer :
    - si source est un Path, ouvre un nouveau handle depuis le disque
      (un nouveau handle à chaque appel, pour pouvoir réutiliser le même
      fichier plusieurs fois dans le pipeline — cf. app.py où chaque
      rerun Streamlit fournit un nouvel UploadedFile non consommé) ;
    - si source est déjà un dict en mémoire (sortie d'OCR ou de
      prétraitement), l'encapsule dans un BytesIO, comme le fait déjà
      app.py pour transcript_after_sts."""
    if isinstance(source, Path):
        return open(source, "rb")
    buffer = io.BytesIO(json.dumps(source, ensure_ascii=False, indent=2).encode("utf-8"))
    buffer.name = "corpus.json"
    return buffer


def _nom(source, defaut: str) -> str:
    """Nom affichable d'une source (Path ou dict déjà en mémoire)."""
    return source.name if isinstance(source, Path) else defaut


def _extraire_items(corpus: dict) -> list:
    """Normalise 'repliques' (list ou dict) en liste, comme dans app.py."""
    repliques = corpus.get("repliques", {}) if corpus else {}
    return repliques if isinstance(repliques, list) else list(repliques.values())


def _compter_corrections(items_avant: list, items_apres: list, speaker_field: str) -> int:
    """Nombre de répliques dont le champ speaker a changé entre deux états
    successifs du pipeline, comme dans app.py."""
    return sum(
        1 for avant, apres in zip(items_avant, items_apres)
        if str(avant.get(speaker_field, "")) != str(apres.get(speaker_field, ""))
    )


# ── Onglet "Prétraitements" ──────────────────────────────────────────

def preparer_gold(chemin_gold: Path, api_key: str = ""):
    """Retourne une source gold utilisable par le pipeline (Path déjà
    prête, ou dict généré en mémoire). Si chemin_gold est un PDF, lance
    l'OCR (Mistral) pour produire le fichier de référence, comme
    'Générer un fichier de référence depuis un PDF' dans app.py."""
    if chemin_gold.suffix.lower() != ".pdf":
        return chemin_gold  # déjà un .json prêt à l'emploi

    if not api_key:
        raise RuntimeError(
            "Une clé API Mistral est requise pour l'OCR d'un PDF."
            "(--api-key ou variable d'environnement MISTRAL_API_KEY)."
        )

    with etape("OCR", f"POST /ocr  fichier={chemin_gold.name}", pc=10):
        client = get_mistral_client(api_key)
        json_output, raw_text = run_ocr_pipeline(chemin_gold.read_bytes(), chemin_gold.name, client)

    if not raw_text.strip():
        raise RuntimeError(f"OCR : aucun texte extrait de {chemin_gold.name} (vérifie le PDF / la clé API).")

    n_scenes = len(json_output.get("scenes", []))
    n_repliques = len(json_output.get("repliques", []))
    if n_scenes == 0:
        log("OCR", f"⚠ aucune scène identifiée ({len(raw_text)} caractère(s) de texte brut récupérés)")
    else:
        log("OCR", f"{n_scenes} scène(s), {n_repliques} réplique(s) extraite(s)")

    return json_output


def preparer_transcript(chemin_transcript: Path, architecture: str = "daia"):
    """Retourne une source transcript utilisable par le pipeline (Path
    déjà prête, ou dict généré en mémoire). Si le fichier n'a pas encore
    de 'repliques' (transcription brute, mots seuls), lance le
    prétraitement LLM pour regrouper les mots en répliques, comme
    'Nettoyage du fichier de transcription' dans app.py.

    architecture : 'daia' (mots sous 'words'/'content'/'speaker_id',
    déjà le format natif attendu ci-dessous — aucune conversion) ou
    'otto' (mots sous 'transcription'/'text'/'speakerId' — converti ici
    vers le format natif 'words'/'content'/'speaker_id' avant
    prétraitement, via transcript_loader.load_otto)."""
    with open(chemin_transcript, "rb") as f:
        corpus = json.load(f)

    if corpus.get("repliques"):
        return chemin_transcript  # déjà prétraité

    if architecture == "otto":
        normalise = load_otto(chemin_transcript)
        corpus = {
            "speakers": normalise["speakers"],
            "words": [
                {
                    "content": mot["text"],
                    # otto ne renseigne pas toujours speaker_id au niveau mot
                    # (vide dans les fichiers otto observés) ; on force alors 0
                    # (un int) plutôt que None, pour que ScenesAnalyzer.is_unassigned()
                    # détecte ces répliques et les corrige via l'alignement au gold —
                    # le même mécanisme qui gère déjà les speaker_id daia non résolus.
                    "speaker_id": mot["speaker_id"] if mot["speaker_id"] else 0,
                    "start_time": mot["start_time"],
                    "end_time": mot["end_time"],
                    "confidence": mot["confidence"],
                    "newline": mot["newline"],
                }
                for mot in normalise["words"]
            ],
        }

    n_mots = len(corpus.get("words", []))
    with etape("PRETRAITEMENT", f"POST /pretraitement  mots={n_mots}", pc=20):
        corpus_clean = run_preprocessing_llm(corpus, architecture=architecture)

    n_repliques = len(corpus_clean.get("repliques", {}))
    log("PRETRAITEMENT", f"{n_repliques} réplique(s) formée(s) à partir de {n_mots} mot(s)")

    return corpus_clean


# ── Onglet "Import & Analyse" ────────────────────────────────────────

def run_pipeline(gold_source, transcript_source, appliquer_tirets: bool = False) -> dict:
    """Exécute le pipeline de correction (STS → Hapax → Scènes [→ Tirets]),
    comme l'onglet 'Import & Analyse' de app.py. gold_source/transcript_source
    sont chacun soit un Path (fichier .json déjà prêt), soit un dict déjà
    en mémoire (sortie de preparer_gold/preparer_transcript). Retourne le
    corpus final."""

    nom_gold = _nom(gold_source, "gold (généré par OCR)")
    nom_transcript = _nom(transcript_source, "transcript (prétraité)")

    if isinstance(transcript_source, Path):
        with open(transcript_source, "rb") as f:
            corpus_original = json.load(f)
    else:
        corpus_original = transcript_source
    n_repliques_depart = len(_extraire_items(corpus_original))
    log("TRANSCRIPT", f"chargé — {nom_transcript} ({n_repliques_depart} réplique(s))")

    corpus_gold_preview = (
        json.load(open(gold_source, "rb")) if isinstance(gold_source, Path) else gold_source
    )
    log("GOLD", f"chargé — {nom_gold} ({len(_extraire_items(corpus_gold_preview))} réplique(s))")

    # ── STS ──
    with etape("STS", f"POST /analyse/sts  gold={nom_gold} transcript={nom_transcript}", pc=40):
        corpus_final = STSAnalyzer.run_analyse_sts(
            _obtenir_fichier(gold_source), _obtenir_fichier(transcript_source), return_corpus=True
        )
    if not corpus_final:
        raise RuntimeError("L'analyse STS n'a retourné aucun corpus.")
    corpus_apres_sts = json.loads(json.dumps(corpus_final))
    log("STS", f"{len(_extraire_items(corpus_apres_sts))} réplique(s) en sortie")

    transcript_after_sts = io.BytesIO(json.dumps(corpus_final, ensure_ascii=False, indent=2).encode("utf-8"))
    transcript_after_sts.name = "transcript_after_sts-hapax.json"

    # ── Hapax ──
    with etape("HAPAX", "POST /analyse/hapax", pc=60):
        corpus_final = HapaxAnalyzer.run_analyse_hapax(
            _obtenir_fichier(gold_source), transcript_after_sts, return_corpus=True
        )
    if not corpus_final:
        raise RuntimeError("L'analyse Hapax n'a retourné aucun corpus.")
    corpus_apres_hapax = json.loads(json.dumps(corpus_final))
    log("HAPAX", f"{len(_extraire_items(corpus_apres_hapax))} réplique(s) en sortie")

    # ── Scènes (inclut désormais exclude_indexing_before_sentence) ──
    with etape("SCENES", "POST /analyse/scenes", pc=80):
        corpus_final = ScenesAnalyzer.scene_analyze(
            _obtenir_fichier(gold_source), corpus_in=corpus_final, return_corpus=True
        )
    if not corpus_final:
        raise RuntimeError("L'analyse Scènes n'a retourné aucun corpus.")
    corpus_apres_scenes = json.loads(json.dumps(corpus_final))
    log("SCENES", f"{len(_extraire_items(corpus_apres_scenes))} réplique(s) en sortie")

    # ── Tirets (optionnel) ──
    if appliquer_tirets:
        with etape("TIRETS", "POST /option/tirets", pc=90):
            corpus_final = TiretOption.run_option_tirets(
                _obtenir_fichier(transcript_source), corpus_in=corpus_final, return_corpus=True
            )
        log("TIRETS", "terminé")

    # ── Récapitulatif des corrections par approche ──
    items_originaux = _extraire_items(corpus_original)
    items_sts        = _extraire_items(corpus_apres_sts)
    items_hapax       = _extraire_items(corpus_apres_hapax)
    items_scenes      = _extraire_items(corpus_apres_scenes)
    speaker_field = "speaker_id" if items_originaux and "speaker_id" in items_originaux[0] else "speaker"

    corr_sts    = _compter_corrections(items_originaux, items_sts,    speaker_field)
    corr_hapax  = _compter_corrections(items_sts,       items_hapax,  speaker_field)
    corr_scenes = _compter_corrections(items_hapax,     items_scenes, speaker_field)
    total = corr_sts + corr_hapax + corr_scenes
    n_repliques = len(items_originaux)
    pct = round(100 * total / n_repliques, 1) if n_repliques else 0.0

    log("RECAP", f"STS={corr_sts} Hapax={corr_hapax} Scènes={corr_scenes} | total={total}/{n_repliques} ({pct}%)")

    # ── Complétion avec les champs originaux manquants ──
    for key, value in corpus_original.items():
        if key not in corpus_final:
            corpus_final[key] = value

    return corpus_final


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline complet (Prétraitements + Import & Analyse) : "
            "gold peut être un .pdf (OCR) ou un .json prêt ; transcript peut être brut "
            "(mots seuls) ou déjà prétraité (repliques)."
        )
    )
    parser.add_argument("gold", type=Path, help="Fichier gold : scénario .pdf (OCR) ou référence .json déjà prête")
    parser.add_argument("transcript", type=Path, help="Fichier transcription : brut (.json, mots seuls) ou déjà prétraité (repliques)")
    parser.add_argument("--api-key", default=os.environ.get("MISTRAL_API_KEY", ""),
                         help="Clé API Mistral (requise seulement si gold est un .pdf). "
                              "Par défaut : variable d'environnement MISTRAL_API_KEY.")
    parser.add_argument("--tirets", action="store_true", help="Applique l'option Tirets en fin de pipeline")
    parser.add_argument("--architecture", choices=["daia", "otto"], default="daia",
                         help="Architecture du fichier transcript brut (ignoré si 'repliques' déjà présent) : "
                              "'daia' (mots/content/speaker_id, défaut) ou 'otto' (transcription/text/speakerId).")
    parser.add_argument("--port", type=int, default=None,
                         help="Port local"
                              "Si omis, aucun rapport d'avancement n'est envoyé.")
    parser.add_argument("--asid", type=int, default=None,
                         help="Identifiant de l'asset (transcription)")
    parser.add_argument(
        "-o", "--sortie", type=Path, default=Path("output/transcription_finale.json"),
        help="Fichier JSON de sortie (défaut : transcription_finale.json) dans le dossier output.",
    )
    args = parser.parse_args()

    global PORT, ASID
    PORT, ASID = args.port, args.asid
    if PORT is not None and ASID is None:
        log("ERREUR", "--asid est requis quand --port est fourni.")
        sys.exit(1)

    if not args.gold.exists():
        log("ERREUR", f"fichier gold introuvable : {args.gold}")
        sys.exit(1)
    if not args.transcript.exists():
        log("ERREUR", f"fichier transcript introuvable : {args.transcript}")
        sys.exit(1)
    log("INPUTS", f"gold={args.gold.name} transcript={args.transcript.name}")

    api_key = args.api_key
    if args.gold.suffix.lower() == ".pdf" and not api_key:
        api_key = getpass.getpass("Clé API Mistral : ")
        if not api_key:
            log("ERREUR", "clé API Mistral requise pour l'OCR d'un PDF gold.")
            sys.exit(1)

    try:
        gold_source = preparer_gold(args.gold, api_key=api_key)
        transcript_source = preparer_transcript(args.transcript, architecture=args.architecture)
        corpus_final = run_pipeline(gold_source, transcript_source, appliquer_tirets=args.tirets)
        corpus_final = (
            export_otto(corpus_final, args.transcript)
            if args.architecture == "otto"
            else export_daia(corpus_final)
        )
    except Exception as e:
        log("ERREUR", f"échec du pipeline — {e}")
        report_error(999, f"Pipeline failed: {e}")
        sys.exit(1)

    args.sortie.parent.mkdir(parents=True, exist_ok=True)
    with etape("EXPORT", f"écriture de {args.sortie.name}", pc=95):
        args.sortie.write_text(
            json.dumps(corpus_final, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    taille_ko = args.sortie.stat().st_size / 1024
    log("EXPORT", f"{args.sortie.name} écrit ({taille_ko:.1f} Ko)")
    report_progress(100, "Everything is fine.", endofprocess=True)


if __name__ == "__main__":
    main()
