"""
Module de prétraitement pour l'app Streamlit (app.py) — regroupe deux
pipelines indépendants qui partagent le même client Mistral :

  A) OCR -> scènes/répliques -> JSON  (run_ocr_pipeline)
     PDF scénario -> texte OCR -> extraction scènes/locuteurs/répliques
     par un modèle génératif (extract_scenes_llm) -> nettoyage.

  B) Prétraitement transcript "words" -> répliques  (run_preprocessing_llm)
     Fichier JSON mot-par-mot (avec speaker_id) -> regroupement en
     répliques via preprocessing.words_concatenation() (script existant,
     sans appel au modèle génératif).

Étape 2 du pipeline A (extraction scènes/locuteurs/répliques) : réalisée
par extract_scenes_llm, qui REMPLACE le parsing par regex
(decouper_en_scenes/traiter_scene/Stanza). Ces fonctions regex sont
conservées plus bas, marquées "LEGACY — non utilisé par défaut".

Différences avec les scripts CLI d'origine (voir commentaires "# ADAPT") :
  - les fonctions opèrent en mémoire (bytes uploadés / dict Python), pas
    sur des chemins disque codés en dur ;
  - le client Mistral est mis en cache via st.cache_resource ;
  - pas de CLI/argparse pour le pipeline B (pretraitements2.py en avait
    un ; il est retiré ici puisque ce module est importé par app.py — si
    tu veux garder un usage CLI autonome pour B, dis-le-moi, je l'ajoute
    à côté).
"""

import json
import os
import re

import stanza
import streamlit as st
from mistralai.client import Mistral

from preprocessing import words_concatenation, words_concatenation_newline

MODELE_EXTRACTION = "mistral-medium-latest"

EXTRACTION_SYSTEM_PROMPT = """Tu es un outil d'extraction de données pour l'analyse de scénarios.

À partir du texte brut fourni (issu d'un OCR, peut contenir des artefacts de mise en page), identifie :
- les scènes (découpage tel qu'indiqué dans le texte : "SCÈNE", "INT."/"EXT.", numérotation, etc.) ;
- pour chaque réplique de dialogue à l'intérieur d'une scène : le nom du locuteur et le texte exact de sa réplique.
- les personnages et ajoute un champ "speakers" rassemblant tous les personnages du fichier.

Réponds UNIQUEMENT avec un objet JSON de cette forme exacte, sans aucun texte autour :
{
  "scenes": [
    {"scene_number": 1987/02, "scene_desc": "<intitulé de scène tel qu'il apparaît dans le texte sans l'index>"}
  ],
  "repliques": {
    1: {"scene_index": 1987/08, "speaker": "<NOM DU LOCUTEUR>", "line": "<texte de la réplique>"}
    2: {"scene_index": 1987/08, "speaker": "<NOM DU LOCUTEUR>", "line": "<texte de la réplique>"}
  }
  "speakers": [
    {"id": <indice numérique>, "name": <NOM DU PERSONNAGE>}
  ]
}

Règles :
- "scene_index" référence à l'index de la scène dans la liste "scenes", par exemple : "1976/01"
- Ignore les didascalies, numéros de page, les mentions "SUPPRIMÉE"/"ANNULÉE", génériques, et tout ce qui n'est ni un intitulé de scène ni une réplique.
- Ne traduis pas, ne résume pas, ne reformule pas le texte des répliques : reproduis-le tel quel.
- Si aucune scène n'est identifiable dans le texte, retourne {"scenes": [], "repliques": []}.
"""


def extract_scenes_llm(text: str, client: Mistral, model: str = MODELE_EXTRACTION) -> dict:
    """Extrait scènes/répliques/locuteurs via un modèle génératif Mistral.
    Remplace le parsing par regex (decouper_en_scenes/traiter_scene)."""
    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = response.choices[0].message.content
    try:
        json_output = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Réponse du modèle non conforme au JSON attendu : {e}") from e
    json_output.setdefault("scenes", [])
    json_output.setdefault("repliques", [])

    # Le prompt système décrit "repliques" comme un objet à clés numériques
    # ("1": {...}, "2": {...}), mais le reste du pipeline (nettoyer_repliques,
    # etc.) attend une liste. Si le modèle a suivi le format dict, on le
    # convertit ici en liste triée par clé pour rester cohérent avec le
    # format produit par txt_to_scenes_json().
    if isinstance(json_output["repliques"], dict):
        json_output["repliques"] = [
            v for k, v in sorted(json_output["repliques"].items(), key=lambda kv: int(kv[0]))
        ]

    return json_output

# ══════════════════════════════════════════════════════════════════
# Ressources mises en cache (chargées une seule fois par session)
# ══════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Chargement de Stanza...")
def get_stanza_pipeline():
    # LEGACY — utilisé uniquement par detect_locuteur_stanza() (parsing
    # regex), non appelé par le chemin par défaut (extraction LLM).
    stanza_dir = os.path.join(os.path.expanduser("~"), "stanza_resources", "fr")
    if not os.path.exists(stanza_dir):
        stanza.download("fr", processors="tokenize,ner")
    return stanza.Pipeline("fr", processors="tokenize,ner", tokenize_no_ssplit=True)


@st.cache_resource(show_spinner=False)
def get_mistral_client(api_key: str) -> Mistral:
    return Mistral(api_key=api_key)


# ══════════════════════════════════════════════════════════════════
# Pipeline A, étape 1 — OCR (issu de ocr_to_text.py / pretraitements.py)
# ══════════════════════════════════════════════════════════════════
# ADAPT : prend des bytes en mémoire (uploaded_file.getvalue()) au lieu d'un chemin PDF, et retourne le texte au lieu d'écrire un .txt.

def ocr_pdf_to_text(file_bytes: bytes, file_name: str, client: Mistral) -> str:
    """OCR un PDF (en mémoire) via Mistral et retourne le texte brut."""
    uploaded_file = client.files.upload(
        file={"file_name": file_name, "content": file_bytes},
        purpose="ocr",
    )
    try:
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id)
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "document_url", "document_url": signed_url.url},
        )
        pages_text = []
        for page in ocr_response.pages:
            text = page.markdown
            text = text.replace("-\n", "")
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
            if text:
                pages_text.append(text)
        return "\n\n".join(pages_text)
    finally:
        client.files.delete(file_id=uploaded_file.id)


# ══════════════════════════════════════════════════════════════════
# Pipeline A, étape 2 (LEGACY) — parsing par regex, remplacé par
# extract_scenes_llm() ci-dessus. Conservé pour référence / debug.
# ══════════════════════════════════════════════════════════════════

def detecter_locuteurs_multiples(line):
    """Détecte 2 locuteurs séparés par &. Retourne (loc1, loc2) ou None."""
    match = re.match(r'^([A-ZÉÈÀÇ]+)\s&\s([A-ZÉÈÀÇ]+)', line.strip())
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None


def fusionner_replique_multiligne(lines, index):
    """Concatène toutes les lignes d'une réplique en une seule."""
    replique = lines[index].strip()
    i = index + 1
    while i < len(lines):
        ligne_suivante = lines[i].strip()
        if (
            not ligne_suivante
            or re.match(r'^[A-Z0-9ÉÈÀÇ\s\(\)]+$', ligne_suivante)
            or re.search(r'\b(INT|EXT)\b', ligne_suivante)
        ):
            break
        replique += " " + ligne_suivante
        i += 1
    return replique, i


def detect_locuteur_stanza(line, current_speaker, nlp):
    """Utilise Stanza pour détecter un locuteur si la regex échoue."""
    # ADAPT : nlp passé en paramètre (au lieu d'une globale au niveau module)
    if current_speaker:
        return current_speaker, line
    doc = nlp(line)
    for sent in doc.sentences:
        for ent in sent.ents:
            if ent.type == "PER":
                locuteur = ent.text
                replique = line.replace(locuteur, "").strip(" :–—")
                return locuteur, replique
    return current_speaker, line


def normaliser_ligne(line):
    """Applique les normalisations communes à une ligne."""
    line = re.sub(r"\s*[-|–]\s*", " - ", line, count=1)
    line = re.sub(r"^(\d+\s?[A-C]?)\s+-?(INT|EXT)\b", r"\1 - \2", line)
    line = re.sub(r"^(\d+)$", "", line)
    line = re.sub(r"\([^)]*\)", "", line)
    line = re.sub(r"^(\d+)\s([A-C])", r"\1\2", line)
    line = re.sub(r"^EXT\s?/\s?INT\b.", "EXT/INT.", line)
    return line.strip()


def est_ligne_inutile(line):
    """Retourne True si la ligne doit être ignorée."""
    if re.match(r"^(\d+)\.?\s*-?\s*(SUP?PRIM[ÉE]E?|ANNUL[ÉE]E?)$", line, re.IGNORECASE):
        return True
    if re.match(r"^\s*JOUR\s+\d+\.?\s*$", line, re.IGNORECASE):
        return True
    if re.match(r"^(TEASER|FIN|GENERIQUE|GÉNÉRIQUE|FIN GENERIQUE|FIN GÉNÉRIQUE)\s*$", line):
        return True
    if line.startswith("«"):
        return True
    if re.match(r"SÉQUENCE EN ALTERNANCE AVEC LA SUIVANTE", line):
        return True
    return False


def decouper_en_scenes(content):
    """Retourne une liste de blocs, chacun commençant par un intitulé de scène."""
    scenes = []
    bloc_courant = []

    for line in content:
        line = normaliser_ligne(line)
        if not line or est_ligne_inutile(line):
            if bloc_courant:
                bloc_courant.append("")
            continue
        if re.search(r'\b(EXT|INT)\b', line):
            if bloc_courant:
                scenes.append(bloc_courant)
            bloc_courant = [line]
        else:
            bloc_courant.append(line)

    if bloc_courant:
        scenes.append(bloc_courant)

    return scenes


def traiter_scene(bloc, nlp):
    """Traite un bloc de scène et retourne (intitulé, répliques)."""
    intitule = bloc[0]
    repliques = []
    in_narrative = True
    current_speaker = None
    lines = bloc[1:]
    i = 0

    while i < len(lines):
        line = lines[i]
        i += 1

        if not line:
            current_speaker = None
            in_narrative = True
            continue

        if in_narrative:
            if re.match(r'^[A-ZÉÈÀÇ0-9][A-ZÉÈÀÇ0-9\s\(\)\-\./&]*$', line):
                in_narrative = False
            else:
                continue

        if re.match(r'^[A-ZÉÈÀÇ0-9][A-ZÉÈÀÇ0-9\s\(\)\-\./&]*$', line):
            double = detecter_locuteurs_multiples(line)
            if double:
                current_speaker = double
            else:
                current_speaker = line.strip()
            continue

        if current_speaker and line:
            line, i = fusionner_replique_multiligne(lines, i - 1)
            line = re.sub(r'\([^)]*\)', '', line).strip()

            current_speaker, line = detect_locuteur_stanza(line, current_speaker, nlp)

            if line.strip():
                phrases = re.split(r'(?<=[.!?])\s*', line.strip())
                speakers = list(current_speaker) if isinstance(current_speaker, tuple) else [current_speaker]
                for speaker in speakers:
                    for phrase in phrases:
                        if phrase and phrase != ".":
                            repliques.append({"speaker": speaker, "line": phrase.strip()})

            current_speaker = None
            in_narrative = True

    return intitule, repliques


def txt_to_scenes_json(text: str, nlp) -> dict:
    """Équivalent de simplify_script_to_json(), à partir d'une chaîne de
    texte en mémoire (issue de ocr_pdf_to_text()) plutôt que d'un fichier."""
    # ADAPT : text.splitlines() au lieu de open(...).readlines()
    content = text.splitlines()

    blocs = decouper_en_scenes(content)
    json_output = {"scenes": [], "repliques": []}

    for bloc in blocs:
        intitule, repliques = traiter_scene(bloc, nlp)

        if repliques:
            json_output["scenes"].append({
                "scene_number": len(json_output["scenes"]) + 1,
                "old_scene_number": intitule,
            })
            for r in repliques:
                json_output["repliques"].append({
                    "scene_index": len(json_output["scenes"]) - 1,
                    "speaker": r["speaker"],
                    "line": r["line"],
                })

    return json_output


# ══════════════════════════════════════════════════════════════════
# Pipeline A, étape 2bis — segmentation des répliques en phrases
# ══════════════════════════════════════════════════════════════════
# Découpe chaque réplique sur ponctuation forte ('.', '!', '?', '...'),
# en protégeant les abréviations courantes pour ne pas segmenter dessus
# (ex. "M. Cosmo" ne doit pas devenir deux phrases). À appliquer après
# l'extraction (extract_scenes_llm ou txt_to_scenes_json), avant
# nettoyer_repliques().

_ABBREVIATIONS_SEGMENTATION = [
    r"M", r"Mme", r"Mlle", r"Dr", r"Pr", r"St", r"Ste", r"L", r"J", r"Cie", r"etc",
]

# Abréviation suivie d'un espace + majuscule/chiffre : neutralisée avant le split.
_ABBREV_SEGMENTATION_RE = re.compile(
    r'\b(' + '|'.join(_ABBREVIATIONS_SEGMENTATION) + r')\.\s+(?=[A-ZÉÈÊËÀÂÙÛÎÏÔÇŒÆ0-9"\'])',
    re.IGNORECASE,
)

# Coupe après ponctuation forte ('.', '!', '?') suivie d'un espace + majuscule/chiffre/guillemet.
_STRONG_PUNCT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-ZÉÈÊËÀÂÙÛÎÏÔÇŒÆ0-9"\'«])')

_SEG_PLACEHOLDER = '\x00'


def segmenter_en_phrases(text: str) -> list[str]:
    """Découpe un texte (réplique) en phrases individuelles, sur
    ponctuation forte ('.', '!', '?', y compris les points de suspension
    '...'), en évitant de couper sur les abréviations courantes."""
    if not text or not text.strip():
        return []

    # 1. Neutraliser les points de suspension ("..." -> un seul caractère
    #    '…') pour qu'ils ne soient pas traités comme 3 points de phrase.
    protected = text.replace("...", "\u2026")

    # 2. Neutraliser les abréviations : "M. Cosmo" -> "M\x00Cosmo"
    protected = _ABBREV_SEGMENTATION_RE.sub(lambda m: m.group(1) + _SEG_PLACEHOLDER, protected)

    # 3. Découper sur ponctuation forte
    sentences = _STRONG_PUNCT_RE.split(protected)

    # 4. Restaurer abréviations et points de suspension dans chaque phrase
    phrases = []
    for s in sentences:
        s = s.replace(_SEG_PLACEHOLDER, ". ").replace("\u2026", "...").strip()
        if s:
            phrases.append(s)
    return phrases


def segmenter_repliques(json_output: dict) -> dict:
    """Applique segmenter_en_phrases() à chaque réplique de
    json_output["repliques"] : une réplique multi-phrases devient
    plusieurs répliques (mêmes scene_index/speaker, une par phrase)."""
    nouvelles_repliques = []
    for r in json_output.get("repliques", []):
        phrases = segmenter_en_phrases(r.get("line", ""))
        if not phrases:
            nouvelles_repliques.append(r)
            continue
        for phrase in phrases:
            nouvelle_replique = dict(r)
            nouvelle_replique["line"] = phrase
            nouvelles_repliques.append(nouvelle_replique)
    json_output["repliques"] = nouvelles_repliques
    return json_output


# ══════════════════════════════════════════════════════════════════
# Pipeline A, étape 2ter — retrait des didascalies entre parenthèses
# ══════════════════════════════════════════════════════════════════
# À appliquer après segmenter_repliques() (une didascalie en milieu de
# réplique ne doit pas fausser le découpage en phrases fait juste avant).

_DIDASCALIE_ETOILE_RE = re.compile(r'\*\([^)]*\)\*')  # *(blessée)*
_DIDASCALIE_PAREN_RE  = re.compile(r'\([^)]*\)')      # (penaud)


def retirer_didascalies(text: str) -> str:
    """Retire les didascalies entre parenthèses d'une réplique (ex.
    '(souriant)', '*(blessée)*'), puis normalise les espaces laissés
    par leur suppression."""
    if not text:
        return text
    text = _DIDASCALIE_ETOILE_RE.sub('', text)
    text = _DIDASCALIE_PAREN_RE.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def retirer_didascalies_repliques(json_output: dict) -> dict:
    """Applique retirer_didascalies() à chaque réplique de
    json_output["repliques"]. Une réplique qui ne contenait qu'une
    didascalie (ligne vide une fois la parenthèse retirée) est retirée
    de la liste."""
    nouvelles_repliques = []
    for r in json_output.get("repliques", []):
        ligne_nettoyee = retirer_didascalies(r.get("line", ""))
        if ligne_nettoyee:
            r["line"] = ligne_nettoyee
            nouvelles_repliques.append(r)
    json_output["repliques"] = nouvelles_repliques
    return json_output


# ══════════════════════════════════════════════════════════════════
# Pipeline A, étape 3 — nettoyage du texte des répliques
# (issu de preprocessing.py)
# ══════════════════════════════════════════════════════════════════

def normaliser_noms(text):
    replacements = [
        (r"\ble cosmétique\b", "L. Cosmétiques"),
        (r"\belle cosmétique\b", "L. Cosmétiques"),
        (r"\bl cosmétique\b", "L. Cosmétiques"),
        (r"\bl cosmétiques\b", "L. Cosmétiques"),
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return text


def normaliser_phrases(replique):
    if not isinstance(replique, str):
        replique = str(replique) if replique is not None else ""
    replique_clean = re.sub(r"\s+", " ", replique)
    return replique_clean


def nettoyer_repliques(json_output: dict) -> dict:
    """Applique normaliser_phrases()/normaliser_noms() à chaque réplique
    du JSON produit par txt_to_scenes_json() ou extract_scenes_llm()."""
    for r in json_output["repliques"]:
        line = normaliser_phrases(r["line"])
        line = normaliser_noms(line)
        r["line"] = line
    return json_output


# ══════════════════════════════════════════════════════════════════
# Pipeline A — orchestration (un fichier à la fois, appelée par app.py)
# ══════════════════════════════════════════════════════════════════

def run_ocr_pipeline(file_bytes: bytes, file_name: str, client: Mistral) -> tuple[dict, str]:
    """OCR -> extraction scènes/répliques (LLM) -> segmentation en
    phrases -> retrait des didascalies -> nettoyage, pour un seul
    fichier PDF en mémoire.

    Retourne (json_output, raw_text) : raw_text est le texte OCR brut,
    avant extraction, utile pour diagnostiquer un JSON vide (texte vide
    -> problème OCR ; texte non vide mais 0 scène -> le modèle n'a pas
    identifié de découpage en scènes dans ce texte)."""
    text = ocr_pdf_to_text(file_bytes, file_name, client)
    json_output = extract_scenes_llm(text, client)
    json_output = segmenter_repliques(json_output)
    json_output = retirer_didascalies_repliques(json_output)
    json_output = nettoyer_repliques(json_output)
    return json_output, text


# ══════════════════════════════════════════════════════════════════
# Pipeline B — prétraitement "words" -> répliques (issu de
# pretraitements2.py / preprocessing.py)
# ══════════════════════════════════════════════════════════════════
#
# Entrée/sortie identiques à preprocessing.py :
#   - entrée  : corpus_transcript["words"] = [{"content": ..., "speaker_id": ...}, ...]
#   - sortie  : corpus_transcript["repliques"] = {id: {"line": ..., "speaker_id": ...}}
#               + un champ "repl_id" ajouté à chaque mot de "words"

def load_json(input_file):
    """Charge un fichier JSON (inchangé par rapport à preprocessing.py)."""
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Erreur lors du chargement du fichier JSON: {e}")
        return {}


def _coerce_speaker_id(value):
    """Convertit en int si le speaker_id est purement numérique (ex. '3'),
    sinon garde la valeur telle quelle (ex. 'S19'). Les identifiants de
    locuteur ne sont pas toujours numériques selon la source du corpus."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def run_preprocessing_llm(corpus_transcript: dict, client: Mistral = None, architecture: str = "daia") -> dict:
    """Version en mémoire du bloc __main__ de preprocessing.py (pas de
    chemins disque codés en dur), utilisable depuis app.py ou un script.

    Appelle preprocessing.words_concatenation() (architecture "daia",
    découpage sur la ponctuation) ou words_concatenation_newline()
    (architecture "otto", découpage sur le champ "newline"). `client` est
    conservé dans la signature (non utilisé) pour ne pas modifier l'appel
    fait depuis app.py."""

    ##### CONVERSION "speakers" (int si numérique, sinon inchangé — ex. "S19")
    if "speakers" in corpus_transcript:
        for speaker in corpus_transcript["speakers"]:
            if "id" in speaker:
                speaker["id"] = _coerce_speaker_id(speaker["id"])

    ##### CONVERSION des speaker_id dans "words" (int si numérique, sinon inchangé)
    if "words" in corpus_transcript:
        for word in corpus_transcript["words"]:
            if "speaker_id" in word:
                word["speaker_id"] = _coerce_speaker_id(word["speaker_id"])

    ##### CONVERSION EN DICT (inchangé)
    words_dict = {
        i: dict(element)
        for i, element in enumerate(corpus_transcript.get("words", []))
    }

    ##### FORMATION des répliques (ponctuation pour daia, newline pour otto)
    concatener = words_concatenation_newline if architecture == "otto" else words_concatenation
    repliques = concatener(words_dict)

    ##### RÉINJECTION des words enrichis (le repl_id est déjà ajouté en
    ##### place par words_concatenation, comme dans preprocessing.py)
    corpus_transcript["words"] = list(words_dict.values())

    ##### IMPLEMENTATION du champ "repliques"
    corpus_transcript["repliques"] = repliques

    return corpus_transcript
