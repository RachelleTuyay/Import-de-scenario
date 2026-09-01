import io
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_echarts import st_echarts

from corpus import Corpus
from hapax_analyzer import HapaxAnalyzer
from pretraitements import get_mistral_client, run_ocr_pipeline, run_preprocessing_llm
from scenes_analyzer import ScenesAnalyzer
from sts_analyzer import STSAnalyzer
from tiret_option import TiretOption


def nom_sortie(file_name: str) -> str:
    """Nom de base pour les fichiers de sortie : extension d'origine
    retirée (ex. '.pdf'), espaces remplacés par des underscores.
    'GS EP 1954 CDVDEF.pdf' -> 'GS_EP_1954_CDVDEF'"""
    return Path(file_name).stem.replace(" ", "_")


def _label_json(cle: str) -> str:
    return f"{cle}.json"


def _extraire_items(corpus: dict) -> list:
    """Normalise 'repliques' (list ou dict) en liste, comme dans les analyzers."""
    repliques = corpus.get("repliques", {}) if corpus else {}
    return repliques if isinstance(repliques, list) else list(repliques.values())


def _compter_corrections(items_avant: list, items_apres: list, speaker_field: str) -> int:
    """Nombre de répliques dont le champ speaker a changé entre deux états
    successifs du pipeline (ex. avant/après l'étape STS)."""
    return sum(
        1 for avant, apres in zip(items_avant, items_apres)
        if str(avant.get(speaker_field, "")) != str(apres.get(speaker_field, ""))
    )


##### APP STREAMLIT (main)

apptitle = 'Import de scénario'
st.set_page_config(page_title=apptitle, page_icon=":clapper::", layout="wide")
st.title('Import de scénario 🎬')
st.markdown("---")

if "ocr_results" not in st.session_state:
    st.session_state["ocr_results"] = {}  # {nom_fichier: json_output}
if "ocr_raw_text" not in st.session_state:
    st.session_state["ocr_raw_text"] = {}  # {nom_fichier: texte OCR brut}

tab_preprocessing, tab_analyse, tab_doc = st.tabs(["🧹 Prétraitements", "📥 Import & Analyse", "📂 Documentation"])

##### ══════════════════════════════════════════════════════════════
##### Onglet 1 — Prétraitements (OCR + prétraitement transcription)
##### ══════════════════════════════════════════════════════════════

with tab_preprocessing:

    api_key = st.text_input(
        "Clé API Mistral",
        value=os.environ.get("MISTRAL_API_KEY", ""),
        type="password",
        key="mistral_api_key",
    )
    st.markdown("---")

    ##### Génération d'un fichier de référence
    st.subheader("Générer un fichier de référence depuis un PDF")

    uploaded_pdfs = st.file_uploader(
        "📄 Scénario(s) PDF à convertir", type=["pdf"], accept_multiple_files=True, key="pdf_ocr"
    )
    lancer_ocr = st.button("▶ Lancer l'OCR", key="lancer_ocr")

    if lancer_ocr:
        if not api_key:
            st.warning("Veuillez renseigner la clé API Mistral.")
        elif not uploaded_pdfs:
            st.warning("Veuillez charger au moins un fichier PDF.")
        else:
            client = get_mistral_client(api_key)
            # ── un fichier à la fois : un échec n'interrompt pas les suivants ──
            for pdf_file in uploaded_pdfs:
                cle = nom_sortie(pdf_file.name)
                with st.spinner(f"Traitement en cours..."):
                    try:
                        json_output, raw_text = run_ocr_pipeline(pdf_file.getvalue(), pdf_file.name, client)
                        st.session_state["ocr_results"][cle] = json_output
                        st.session_state["ocr_raw_text"][cle] = raw_text
                        n_scenes = len(json_output["scenes"])
                        n_repliques = len(json_output["repliques"])
                        if not raw_text.strip():
                            st.error(f"{pdf_file.name} : l'OCR n'a extrait aucun texte (vérifie le PDF / la clé API).")
                        elif n_scenes == 0:
                            st.warning(
                                f"{pdf_file.name} : texte OCR récupéré ({len(raw_text)} caractères) "
                                "mais le modèle n'a identifié aucune scène — vérifie le texte brut "
                                "ci-dessous (mise en page trop dégradée par l'OCR ?)."
                            )
                        else:
                            st.success(f"{pdf_file.name} : {n_scenes} scène(s), {n_repliques} réplique(s).")
                    except Exception as e:
                        st.error(f"Échec sur {pdf_file.name} : {e}")

    # ── Fichiers générés par l'OCR uniquement (exclut les fichiers "_clean"
    #    du prétraitement, qui partagent le même dict ocr_results) ──
    fichiers_ocr_generes = {
        k: v for k, v in st.session_state["ocr_results"].items() if not k.endswith("_clean")
    }
    if st.session_state["ocr_results"]:
        col_titre_ocr, col_clear_ocr = st.columns([4, 1])
        with col_titre_ocr:
            st.markdown("##### Fichier de référence généré")
        with col_clear_ocr:
            if st.button("🗑️ Effacer les fichiers importés", key="clear_ocr_results"):
                st.session_state["ocr_results"] = {}
                st.session_state["ocr_raw_text"] = {}
                st.rerun()
        for nom_fichier, json_output in fichiers_ocr_generes.items():
            transcript_bytes = json.dumps(json_output, ensure_ascii=False, indent=2).encode("utf-8")
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label=f"📥 Télécharger {nom_fichier}.json",
                    data=transcript_bytes,
                    file_name=f"{nom_fichier}.json",
                    mime="application/json",
                    key=f"dl_{nom_fichier}",
                )
            with col_dl2:
                raw_text = st.session_state["ocr_raw_text"].get(nom_fichier, "")
                st.download_button(
                    label=f"📥 Télécharger {nom_fichier}.txt (OCR brut)",
                    data=raw_text.encode("utf-8"),
                    file_name=f"{nom_fichier}.txt",
                    mime="text/plain",
                    key=f"dl_txt_{nom_fichier}",
                )
            with st.expander(f"Aperçu du texte OCR brut — {nom_fichier}"):
                st.text(raw_text[:3000] if raw_text else "(vide)")

    st.markdown("---")

    ##### Prétraitement de la transcription

    st.subheader("Nettoyage du fichier de transcription")
    st.markdown(
        "Regroupe les mots d'un fichier de transcription brut en phrases."
    )

    uploaded_words_json = st.file_uploader(
        "📄 Fichier transcription brut", type=["json"], key="words_json"
    )
    lancer_preprocessing = st.button("▶ Lancer le prétraitement", key="lancer_preprocessing")

    if lancer_preprocessing:
        if uploaded_words_json is None:
            st.warning("Veuillez charger un fichier de transcription.")
        else:
            uploaded_words_json.seek(0)
            corpus_transcript = json.load(uploaded_words_json)
            n_mots = len(corpus_transcript.get("words", []))
            with st.spinner(f"Prétraitement de {n_mots} mot(s) en cours..."):
                try:
                    corpus_clean = run_preprocessing_llm(corpus_transcript)
                    cle_prep = f"{nom_sortie(uploaded_words_json.name)}_clean"
                    st.session_state["ocr_results"][cle_prep] = corpus_clean
                    n_repliques = len(corpus_clean.get("repliques", {}))
                    st.success(f"{uploaded_words_json.name} : {n_repliques} réplique(s) formée(s).")
                except Exception as e:
                    st.error(f"Échec du prétraitement : {e}")

    # ── Fichiers prétraités disponibles (partagés avec l'onglet Import & Analyse) ──
    resultats_pretraitement = {
        k: v for k, v in st.session_state["ocr_results"].items() if k.endswith("_clean")
    }
    if resultats_pretraitement:
        col_titre_prep, col_clear_prep = st.columns([4, 1])
        with col_titre_prep:
            st.markdown("##### Fichier Transcription prétraité")
        with col_clear_prep:
            if st.button("🗑️ Effacer les fichiers prétraités", key="clear_preprocessing_results"):
                for cle in list(resultats_pretraitement.keys()):
                    st.session_state["ocr_results"].pop(cle, None)
                st.rerun()
        for nom_fichier, corpus_clean in resultats_pretraitement.items():
            clean_bytes = json.dumps(corpus_clean, ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button(
                label=f"📥 Télécharger {nom_fichier}.json",
                data=clean_bytes,
                file_name=f"{nom_fichier}.json",
                mime="application/json",
                key=f"dl_prep_{nom_fichier}",
            )
        st.caption(
            "Ces fichiers générés sont aussi sélectionnables pour l'onglet **Import & Analyse**."
        )

##### ══════════════════════════════════════════════════════════════
##### Onglet 2 — Import & Analyse (gold/transcript, pipeline STS/Hapax/Scènes)
##### ══════════════════════════════════════════════════════════════


with tab_analyse:

    ##### Import fichiers JSON
    st.subheader("Import des fichiers JSON")

    # Fichiers générés : on sépare les scénarios OCR (gold) des transcriptions prétraitées (_clean), pour que chaque case coche la bonne famille de fichiers.
    fichiers_gold_disponibles = [
        k for k in st.session_state["ocr_results"].keys() if not k.endswith("_clean")
    ]
    fichiers_transcript_disponibles = [
        k for k in st.session_state["ocr_results"].keys() if k.endswith("_clean")
    ]

    col_up1, col_up2 = st.columns(2)

    with col_up1:
        utiliser_gold_genere = False
        gold_choisi = None
        if fichiers_gold_disponibles:
            utiliser_gold_genere = st.checkbox("Utiliser le fichier généré pour le gold", key="use_gen_gold")
            if utiliser_gold_genere:
                gold_choisi = st.selectbox(
                    "Fichier gold à utiliser",
                    fichiers_gold_disponibles,
                    format_func=_label_json,
                    key="select_gold",
                )
        if utiliser_gold_genere and gold_choisi:
            json_output = st.session_state["ocr_results"][gold_choisi]
            uploaded_gold = io.BytesIO(json.dumps(json_output, ensure_ascii=False).encode("utf-8"))
            uploaded_gold.name = f"{gold_choisi}.json"
            st.info(f"Fichier gold utilisé : **{gold_choisi}.json**")
        else:
            uploaded_gold = st.file_uploader("📄 Fichier gold (référence)", type=["json"], key="gold")

    with col_up2:
        utiliser_transcript_genere = False
        transcript_choisi = None
        if fichiers_transcript_disponibles:
            utiliser_transcript_genere = st.checkbox("Utiliser le fichier généré pour la transcription", key="use_gen_transcript")
            if utiliser_transcript_genere:
                transcript_choisi = st.selectbox(
                    "Fichier transcript à utiliser",
                    fichiers_transcript_disponibles,
                    format_func=_label_json,
                    key="select_transcript",
                )
        if utiliser_transcript_genere and transcript_choisi:
            json_output = st.session_state["ocr_results"][transcript_choisi]
            uploaded_transcript = io.BytesIO(json.dumps(json_output, ensure_ascii=False).encode("utf-8"))
            uploaded_transcript.name = f"{transcript_choisi}.json"
            st.info(f"Fichier Transcription utilisé : **{transcript_choisi}.json**")
        else:
            uploaded_transcript = st.file_uploader("📄 Fichier Transcription (à corriger)", type=["json"], key="transcript")

    if uploaded_transcript:
        uploaded_transcript.seek(0)
        corpus_preview = json.load(uploaded_transcript)
        n = len(corpus_preview.get("repliques", []))
        st.success(f"Transcript chargé : {n} réplique(s).")

    if uploaded_gold:
        uploaded_gold.seek(0)
        corpus_preview = json.load(uploaded_gold)
        n = len(corpus_preview.get("repliques", []))
        st.success(f"Gold chargé : {n} réplique(s).")

    st.markdown("""
        **Cliquez sur le bouton `▶ Lancer l'analyse`**
    """)

    st.markdown("""
    ----
    ### Options possibles post-traitement

    **Tirets** : un tiret est ajouté à chaque changement de speaker.

    ----
    """)

    ### Sidebar : Options

    st.sidebar.markdown("## Options de post-traitement :")
    opt_tirets      = st.sidebar.checkbox("Ajout des tirets", key="opt_tirets")

    ### Boutons sidebar

    st.sidebar.markdown("---")

    # ── Bouton Lancer ──
    lancer = st.sidebar.button("▶ Lancer l'analyse", type="primary", use_container_width=True)

    if lancer:
        # Vérification des fichiers nécessaires (gold requis pour STS, Hapax, Scènes, et NER)
        if uploaded_transcript is None:
            st.warning("Veuillez charger le fichier transcript.")
        elif uploaded_gold is None:
            st.warning("Veuillez charger le fichier gold (nécessaire pour cette analyse).")
        else:
            # ── Analyses ── (on accumule le corpus corrigé : STS → Hapax → Scènes)
            st.markdown("### Résultats de l'analyse")

            corpus_final = None  # contiendra le corpus cumulativement corrigé

            # Snapshots du corpus après chaque étape, pour calculer la répartition
            # des corrections par approche dans le récapitulatif de fin.
            corpus_apres_sts = None
            corpus_apres_hapax = None
            corpus_apres_scenes = None

            with st.expander("Analyse STS", expanded=False):
                corpus_final = STSAnalyzer.run_analyse_sts(uploaded_gold, uploaded_transcript, return_corpus=True)
            if corpus_final:
                corpus_apres_sts = json.loads(json.dumps(corpus_final))

            if corpus_final:
                transcript_after_sts = io.BytesIO(
                    json.dumps(corpus_final, ensure_ascii=False, indent=2).encode("utf-8")
                )
                transcript_after_sts.name = "transcript_after_sts-hapax.json"

                with st.expander("Analyse Hapax", expanded=False):
                    corpus_final = HapaxAnalyzer.run_analyse_hapax(uploaded_gold, transcript_after_sts, return_corpus=True)
                if corpus_final:
                    corpus_apres_hapax = json.loads(json.dumps(corpus_final))

                # ── Analyse Scènes : suit systématiquement Hapax, sur les répliques encore non assignées après STS + Hapax ──
                if corpus_final:
                    with st.expander("Analyse Scènes", expanded=False):
                        corpus_final = ScenesAnalyzer.scene_analyze(
                            uploaded_gold,
                            corpus_in=corpus_final,
                            return_corpus=True,
                        )
                    if corpus_final:
                        corpus_apres_scenes = json.loads(json.dumps(corpus_final))

                # ── Récapitulatif : un seul tableau global + un tableau par approche ──
                if corpus_final:
                    uploaded_transcript.seek(0)
                    corpus_original = json.load(uploaded_transcript)

                    items_originaux = _extraire_items(corpus_original)
                    items_sts       = _extraire_items(corpus_apres_sts) if corpus_apres_sts else items_originaux
                    items_hapax     = _extraire_items(corpus_apres_hapax) if corpus_apres_hapax else items_sts
                    items_scenes    = _extraire_items(corpus_apres_scenes) if corpus_apres_scenes else items_hapax

                    speaker_field = "speaker_id" if items_originaux and "speaker_id" in items_originaux[0] else "speaker"
                    n_repliques = len(items_originaux)

                    corr_sts    = _compter_corrections(items_originaux, items_sts,   speaker_field)
                    corr_hapax  = _compter_corrections(items_sts,       items_hapax, speaker_field)
                    corr_scenes = _compter_corrections(items_hapax,     items_scenes, speaker_field)
                    total_corrections = corr_sts + corr_hapax + corr_scenes
                    pct_corrections = round(100 * total_corrections / n_repliques, 1) if n_repliques else 0.0
                    pct_affiche = f"{pct_corrections:.1f}".replace(".", ",")

                    non_corr_apres_sts    = n_repliques - corr_sts
                    non_corr_apres_hapax  = non_corr_apres_sts - corr_hapax
                    non_corr_apres_scenes = non_corr_apres_hapax - corr_scenes

                    st.markdown("---")

                    # ── Tableau 1 : répartition des corrections par approche ──
                    st.markdown("#### Corrections par approche")
                    df_par_approche = pd.DataFrame([
                        {"Approche": "STS",    "Corrections": corr_sts,          "Non corrigée": non_corr_apres_sts},
                        {"Approche": "Hapax",  "Corrections": corr_hapax,        "Non corrigée": non_corr_apres_hapax},
                        {"Approche": "Scènes", "Corrections": corr_scenes,       "Non corrigée": non_corr_apres_scenes},
                        {"Approche": "Total",  "Corrections": total_corrections, "Non corrigée": non_corr_apres_scenes},
                    ])
                    st.dataframe(df_par_approche, use_container_width=True, hide_index=True)
                    st.caption(f"Taux de correction sur l'ensemble du fichier : {pct_affiche} %")

                    # ── Camemberts interactifs (streamlit-echarts) : récapitulatif global ──
                    st.markdown("#### Récapitulatif")

                    COULEUR_CORRIGE = "#4C9F70"
                    COULEUR_NON_CORRIGE = "#E86A5C"

                    # Détecte le thème actif (clair/foncé) pour adapter la couleur de police
                    # des titres/texte central ; en cas d'échec de détection (ancienne version
                    # de Streamlit, thème système, etc.), on retombe sur un thème clair et un
                    # contour de texte (textBorder) garantit malgré tout la lisibilité.
                    try:
                        theme_actif = st.context.theme.type
                    except Exception:
                        theme_actif = None

                    if theme_actif == "dark":
                        COULEUR_TEXTE = "#F2F2F2"
                        COULEUR_TEXTE_SECONDAIRE = "#CCCCCC"
                        COULEUR_CONTOUR_TEXTE = "#000000"
                    else:
                        COULEUR_TEXTE = "#262730"
                        COULEUR_TEXTE_SECONDAIRE = "#5C5C5C"
                        COULEUR_CONTOUR_TEXTE = "#FFFFFF"

                    def _option_donut(valeurs, noms, titre, texte_central, sous_texte_central, formatter_tooltip, couleurs=None):
                        return {
                            "color": couleurs if couleurs else [COULEUR_CORRIGE, COULEUR_NON_CORRIGE],
                            "tooltip": {"trigger": "item", "formatter": formatter_tooltip},
                            "legend": {
                                "bottom": 0,
                                "left": "center",
                                "textStyle": {"fontSize": 12, "color": COULEUR_TEXTE},
                            },
                            "title": [
                                {
                                    "text": titre,
                                    "left": "center",
                                    "top": 0,
                                    "textStyle": {
                                        "fontSize": 15,
                                        "fontWeight": "bold",
                                        "color": COULEUR_TEXTE,
                                        "textBorderColor": COULEUR_CONTOUR_TEXTE,
                                        "textBorderWidth": 2,
                                    },
                                },
                                {
                                    "text": texte_central,
                                    "subtext": sous_texte_central,
                                    "left": "center",
                                    "top": "43%",
                                    "textStyle": {
                                        "fontSize": 22,
                                        "fontWeight": "bold",
                                        "color": COULEUR_TEXTE,
                                        "textBorderColor": COULEUR_CONTOUR_TEXTE,
                                        "textBorderWidth": 2,
                                    },
                                    "subtextStyle": {
                                        "fontSize": 12,
                                        "color": COULEUR_TEXTE_SECONDAIRE,
                                        "textBorderColor": COULEUR_CONTOUR_TEXTE,
                                        "textBorderWidth": 1.5,
                                    },
                                },
                            ],
                            "series": [{
                                "name": titre,
                                "type": "pie",
                                "radius": ["50%", "70%"],
                                "center": ["50%", "56%"],
                                "avoidLabelOverlap": False,
                                "itemStyle": {
                                    "borderRadius": 6,
                                    "borderColor": "#ffffff",
                                    "borderWidth": 3,
                                },
                                "label": {
                                    "show": False,
                                },
                                "emphasis": {
                                    "label": {"show": False},
                                    "itemStyle": {"shadowBlur": 12, "shadowColor": "rgba(0, 0, 0, 0.25)"},
                                },
                                "data": [
                                    {"value": valeurs[0], "name": noms[0]},
                                    {"value": valeurs[1], "name": noms[1]},
                                ],
                            }],
                        }

                    # ── speaker_id corrigés (%) et scene_index indexés (%) ──
                    n_scene_index_renseignes = sum(
                        1 for r in items_scenes if r.get("scene_index") is not None
                    )
                    pct_scene_index = round(100 * n_scene_index_renseignes / n_repliques, 1) if n_repliques else 0.0
                    pct_scene_index_affiche = f"{pct_scene_index:.1f}".replace(".", ",")

                    col_pie2, col_pie3 = st.columns(2)

                    with col_pie2:
                        option_speaker_id = _option_donut(
                            valeurs=[round(pct_corrections, 1), round(100 - pct_corrections, 1)],
                            noms=["Corrigé", "Non corrigé"],
                            titre="Correction des speaker_id",
                            texte_central=f"{pct_affiche} %",
                            sous_texte_central="speaker_id corrigés",
                            formatter_tooltip="{b} : {c} %",
                        )
                        st_echarts(options=option_speaker_id, height="380px", key="pie_speaker_id")

                    with col_pie3:
                        option_scene_index = _option_donut(
                            valeurs=[pct_scene_index, round(100 - pct_scene_index, 1)],
                            noms=["Indexé", "Non indexé"],
                            titre="Indexation des scene_index",
                            texte_central=f"{pct_scene_index_affiche} %",
                            sous_texte_central="scene_index indexés",
                            formatter_tooltip="{b} : {c} %",
                        )
                        st_echarts(options=option_scene_index, height="380px", key="pie_scene_index")
            else:
                st.warning("L'analyse STS n'a pas retourné de corpus, la suite du pipeline est ignorée.")

            # ── Options (appliquées sur le corpus déjà corrigé) ──
            # NER : en stand-by (case désactivée dans la sidebar, bloc conservé mais inatteignable).
            if any([opt_tirets]):
                st.markdown("### Résultats des options")

                if opt_tirets:
                    with st.expander("Tirets", expanded=True):
                        corpus_final = TiretOption.run_option_tirets(
                            uploaded_transcript,
                            corpus_in=corpus_final,
                            return_corpus=True,
                        )

            # ── Téléchargement du fichier final cumulé ──
            if corpus_final:
                st.markdown("---")
                st.markdown("### 📦 Fichier final")
                st.markdown("Cliquez sur le bouton `Télécharger la version corrigée`")

                # Vérification que tous les champs originaux sont présents
                if uploaded_transcript:
                    uploaded_transcript.seek(0)
                    corpus_original = json.load(uploaded_transcript)
                    for key, value in corpus_original.items():
                        if key not in corpus_final:
                            corpus_final[key] = value

                output_final = json.dumps(corpus_final, ensure_ascii=False, indent=2)
                st.download_button(
                    label="📥 Télécharger la version corrigée",
                    data=output_final,
                    file_name="transcription_finale.json",
                    mime="application/json",
                    type="primary",
                )
            else:
                st.info("Aucun résultat à télécharger.")


##### INFOS/Documentation
with tab_doc :
    st.markdown("""
        ### Description
        Ce projet consiste à développer un plugin d'import de scénario intégré au logiciel de sous-titrage/audiodescription OTTO.

        Son objectif est d'assister les sous-titreurs en corrigeant les transcriptions automatiques à partir de scénarios validés, notamment en améliorant l'identification des locuteurs et l'association des scènes aux répliques.

        Ce plugin **corrige environ 60% des locuteurs** et **indexe 80% les scènes au répliques** afin d'accélérer la production de relevés de dialogues, de sous-titres SME (sourds et malentendants), ainsi que de faciliter l'audiodescription et l'adaptation multilingue.


        Il s'appuie uniquement sur des scénarios validé de *Un si grand soleil* (USGS) et leurs transcriptions automatiques afin d'étudier dans quelle mesure une ressource textuelle fiable peut améliorer la qualité de correction des transcriptions générées automatiquement.

        -----

        ### Documentation
        - L'onglet `Prétraitements` :

        Cette partie permet de transformer 2 inputs (un scénario et une transcription brute) en 2 outputs nettoyés et comparables pour l'analyse.

        Dans un premier temps, elle transforme un PDF en JSON afin que ce fichier devient un fichier de référence. Dans un second temps, l'outil va nettoyer le fichier de transcription.

        Une fois que les fichiers nettoyés, ils sont prêts à passer dans l'algorithme de correction.


        - L'onglet `Import & Analyses` :

        Cette partie est concacrée à la correction automatique.

        1) Inputs : 2 fichiers nettoyés ayant la même mise en forme et format (ici en json).

        2) Cliquez sur le bouton "Lancer l'analyse" pour lancer l'algorithme.

        L'algorithme de correction se base sur l'alignement sémantique entre phrases.

        -----

        ### Autres :

        * [Voir le code](lien github)

    """)


