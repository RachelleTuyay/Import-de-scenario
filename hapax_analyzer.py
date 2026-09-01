import json
from collections import Counter
from dataclasses import dataclass

import pandas as pd
import streamlit as st
import torch
from sentence_transformers import util

from config import MODEL_SAVE_DIR
from corpus import Corpus
from model_loader import ModelLoader
from speaker_enricher import SpeakerEnricher
from sts_analyzer import STSAnalyzer


@dataclass
class HapaxAnalyzer:
    @classmethod
    def build_dict(cls, repliques) -> dict:
        all_words = []
        items = repliques if isinstance(repliques, list) else repliques.values()
        for r in items:
            text = r.get("line", "")
            if text:
                all_words.extend(STSAnalyzer.tokenize(text))
        return dict(Counter(all_words))

    @classmethod
    def similarity_scores_hapax_batch(cls, model, phrases_a: list, phrases_b: list) -> list:
        """
        Calcule les scores de similarité pour des paires (a[i], b[i]) en batch.
        OPTIMISATION : remplace l'ancien similarity_score_hapax() appelé ligne par ligne
        via df.apply(), qui ré-encodait chaque phrase individuellement (2 x N appels
        model.encode() pour N paires). Ici : 2 appels encode() au total, puis comparaison
        paire-à-paire (rapide, en mémoire).
        """
        emb_a = model.encode(phrases_a, convert_to_tensor=True)
        emb_b = model.encode(phrases_b, convert_to_tensor=True)
        # Similarité "diagonale" : on ne veut que sim(a[i], b[i]), pas la matrice complète
        scores = util.pairwise_cos_sim(emb_a, emb_b) if hasattr(util, "pairwise_cos_sim") \
            else torch.nn.functional.cosine_similarity(emb_a, emb_b)
        return [float(s) for s in scores]

    @classmethod
    def run_analyse_hapax(cls, file_gold, file_transcript, return_corpus=False):
        """Lance l'analyse Hapax complète et affiche les résultats dans Streamlit."""
        corpus_gold       = Corpus.load_json_file(file_gold)
        corpus_transcript = Corpus.load_json_file(file_transcript)

        if not corpus_gold or not corpus_transcript:
            st.error("Impossible de charger les fichiers JSON.")
            return

        gold_repliques       = corpus_gold.get("repliques", [])
        transcript_repliques = corpus_transcript.get("repliques", {})

        gold_speakers_map = {
            v["name"]: v["id"]
            for v in SpeakerEnricher.enrich_speakers(corpus_gold["speakers"])
        }
        # Normalisation en liste
        transcript_list = (
            transcript_repliques
            if isinstance(transcript_repliques, list)
            else list(transcript_repliques.values())
        )

        # ── Construction des hapax ──
        freq_gold       = cls.build_dict(gold_repliques)
        freq_transcript = cls.build_dict(transcript_list)

        hapax_gold       = {w for w, c in freq_gold.items()       if c == 1}
        hapax_transcript = {w for w, c in freq_transcript.items() if c == 1}
        common_words     = sorted(hapax_gold & hapax_transcript)
        common_words_set = set(common_words)

        st.info(f"Hapax gold : {len(hapax_gold)} | Hapax transcript : {len(hapax_transcript)} | Mots communs : {len(common_words)}")

        # ── Construction des tables ──
        # OPTIMISATION : tokenize() est maintenant appelé UNE FOIS par réplique (mis en cache
        # dans gold_tokens_cache / transcript_tokens_cache), au lieu d'être ré-exécuté pour
        # chaque mot commun x chaque réplique (complexité O(mots x répliques) -> O(répliques)).
        gold_tokens_cache       = [(r, set(STSAnalyzer.tokenize(r.get("line", "")))) for r in gold_repliques]
        transcript_tokens_cache = [(r, set(STSAnalyzer.tokenize(r.get("line", ""))))  for r in transcript_list]

        table_gold, table_transcript = [], []

        for r, tokens in gold_tokens_cache:
            for word in common_words_set & tokens:
                table_gold.append({
                    "word"        : word,
                    "sent_gold"   : r.get("line", ""),
                    "speaker_gold": gold_speakers_map.get(
                        r.get("speaker", ""),
                        r.get("speaker", "")
                    ),
                })

        for r, tokens in transcript_tokens_cache:
            for word in common_words_set & tokens:
                table_transcript.append({
                    "word"           : word,
                    "sent_transcript": r.get("line", ""),
                })

        df_gold       = pd.DataFrame(table_gold)
        df_transcript = pd.DataFrame(table_transcript)

        if df_gold.empty or df_transcript.empty:
            st.warning("Aucun hapax commun trouvé.")
            return

        df_merged = pd.merge(df_gold, df_transcript, on="word", how="inner")
        df_merged = df_merged[["word", "sent_gold", "speaker_gold", "sent_transcript"]]

        # ── Scores de similarité ──
        # OPTIMISATION : batch encoding (2 appels model.encode() au total) au lieu d'un
        # appel .apply() ligne par ligne qui ré-encodait chaque phrase individuellement.
        model = ModelLoader.load_model(MODEL_SAVE_DIR)
        with st.spinner("Calcul des scores de similarité…"):
            df_merged["similarity"] = cls.similarity_scores_hapax_batch(
                model,
                df_merged["sent_transcript"].tolist(),
                df_merged["sent_gold"].tolist(),
            )

        df_merged = df_merged[df_merged["similarity"] >= 0.4]
        df_merged = df_merged.drop_duplicates(subset=["sent_transcript"], keep="first")

        st.info(f"Paires retenues (similarité ≥ 0.4) : {len(df_merged)}")

        # ── Application des corrections ──
        sent_list    = df_merged["sent_transcript"].tolist()
        speaker_list = df_merged["speaker_gold"].tolist()
        gold_sent_list = df_merged["sent_gold"].tolist()

        # Lookup gold line → scene_index
        gold_scene_lookup = {r.get("line", ""): r.get("scene_index") for r in gold_repliques}

        transcript_corrige = json.loads(json.dumps(corpus_transcript))
        items = transcript_corrige.get("repliques", {})
        modifications = []

        # Détecte le bon champ speaker (speaker_id ou speaker)
        sample = next(iter(items if isinstance(items, list) else items.values()), {})
        speaker_field = "speaker_id" if "speaker_id" in sample else "speaker"

        iterate_over = items if isinstance(items, list) else items.values()
        for r in iterate_over:
            line = r.get("line", "")
            if line in sent_list:
                idx              = sent_list.index(line)
                nouveau_speaker  = speaker_list[idx]
                ancien_speaker   = r.get(speaker_field, r.get("speaker", r.get("speaker_id", "")))
                if str(ancien_speaker) != str(nouveau_speaker):
                    # Récupère le scene_index depuis la ligne gold correspondante
                    gold_line        = gold_sent_list[idx]
                    gold_scene_index = gold_scene_lookup.get(gold_line)
                    modifications.append({
                        "ancien_speaker" : ancien_speaker,
                        "nouveau_speaker": nouveau_speaker,
                        "line"           : line,
                        "scene_index"    : gold_scene_index,
                    })
                    r[speaker_field] = nouveau_speaker
                    if gold_scene_index is not None:
                        r["scene_index"] = gold_scene_index

        # Preservation de tous les champs du corpus original sauf repliques
        for key, value in corpus_transcript.items():
            if key != "repliques" and key not in transcript_corrige:
                transcript_corrige[key] = value
        # Renommage scene_number → scene_index dans scenes
        if "scenes" in transcript_corrige:
            for scene in (transcript_corrige["scenes"] if isinstance(transcript_corrige["scenes"], list)
                        else transcript_corrige["scenes"].values()):
                if "scene_number" in scene:
                    scene["scene_index"] = scene.pop("scene_number")
        # speakers : toujours ecrase par celui du gold (reference authoritative) + enrichissement
        if "speakers" in corpus_gold:
            transcript_corrige["speakers"] = SpeakerEnricher.enrich_speakers(corpus_gold["speakers"])

        # ── Affichage ──
        st.metric("Corrections effectuées", len(modifications))

        with st.expander("📋 Hapax communs & similarités"):
            st.dataframe(df_merged.reset_index(drop=True), use_container_width=True)

        with st.expander(f"🔧 Corrections de speakers ({len(modifications)})"):
            if modifications:
                st.dataframe(pd.DataFrame(modifications), use_container_width=True)
            else:
                st.info("Aucune correction effectuée.")

        if not return_corpus:
            Corpus.to_download_json(transcript_corrige, "transcription_corrigee_hapax.json")

        # Attribution du scene_index et du repl_id aux mots
        transcript_corrige = SpeakerEnricher.assign_scene_index_to_words(transcript_corrige)

        if return_corpus:
            return transcript_corrige
