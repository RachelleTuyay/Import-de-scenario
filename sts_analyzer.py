import json
import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd
import streamlit as st
import torch
from sentence_transformers import util

from config import MODEL_SAVE_DIR, SCORE_THRESHOLD
from corpus import Corpus
from model_loader import ModelLoader
from speaker_enricher import SpeakerEnricher


@dataclass
class STSAnalyzer:

    @classmethod
    def tokenize(cls, text: str):
        text = text.lower()
        return re.findall(r'\b[a-zàâäéèêëîïôùûüçœæ]+\b', text)

    @classmethod
    def normaliser_phrase(cls, sent: str) -> str:
        sent = re.sub(r'  ', ' ', sent)
        sent = re.sub(r'\s+', ' ', sent)
        sent = re.sub(r'([.!?,;:])([^\s])', r'\1 \2', sent)
        return sent

    @classmethod
    def moy_phrases_par_scene(cls, repliques: list) -> float:
        items = repliques if isinstance(repliques, list) else repliques.values()
        scenes = Counter(rep["scene_index"] for rep in items)
        return sum(scenes.values()) / len(scenes) if scenes else 0.0

    @classmethod
    def run_alignment(cls, model, gold_embeddings, gold_lines, gold_speakers, gold_repliques, transcript_keys, transcript_lines, transcript_items, n_gold, n_transcript, window_size, corpus_transcript):
        results, corrections = [], []
        # Deep-copy du corpus complet (comme comparaison_sent_to_sent), pas seulement les répliques
        transcript_corrige = json.loads(json.dumps(corpus_transcript))

        # ── OPTIMISATION : encodage batch de tout le transcript en un seul appel ──
        # (avant : model.encode() était appelé une fois par ligne dans la boucle, ce qui
        #  multipliait les appels au modèle par le nombre de répliques du transcript)
        transcript_embeddings = model.encode(transcript_lines, convert_to_tensor=True)

        for i, t_line in enumerate(transcript_lines):
            key    = transcript_keys[i]
            center = int(i * n_gold / n_transcript)
            start  = max(0, center - window_size)
            end    = min(n_gold, center + window_size + 1)

            t_emb             = transcript_embeddings[i]
            window_embeddings = gold_embeddings[start:end]
            scores            = util.cos_sim(t_emb, window_embeddings)[0]

            best_local_idx   = torch.argmax(scores).item()
            best_score       = scores[best_local_idx].item()
            best_gold_idx    = start + best_local_idx
            best_gold_line   = gold_lines[best_gold_idx]
            best_speaker     = gold_speakers[best_gold_idx]

            # clé réelle dans transcript_items / transcript_corrige["repliques"],
            # qui peut être 0-indexée (liste normalisée) ou 1-indexée (dict d'origine, ex. fichier _clean)
            rep = transcript_items[key]
            if "speaker_id" in rep:
                speaker_field    = "speaker_id"
                original_speaker = rep["speaker_id"]
            else:
                speaker_field    = "speaker"
                original_speaker = rep.get("speaker", "")

            if best_score >= SCORE_THRESHOLD:
                if str(original_speaker) != str(best_speaker):
                    transcript_corrige["repliques"][key][speaker_field] = best_speaker
                    # Ajout du scene_index du gold uniquement sur les répliques corrigées
                    gold_scene_index = gold_repliques[best_gold_idx].get("scene_index")
                    if gold_scene_index is not None:
                        transcript_corrige["repliques"][key]["scene_index"] = gold_scene_index
                    corrections.append({
                        "phrase"         : t_line,
                        "ancien_speaker" : original_speaker,
                        "nouveau_speaker": best_speaker,
                        "gold_phrase"    : best_gold_line,
                        "score"          : best_score,
                        "scene_index"    : gold_scene_index,
                    })
                if t_line.strip().lower() == best_gold_line.strip().lower():
                    pass  # conservé pour ne pas modifier la structure de la boucle

            results.append({
                "index"            : i + 1,
                "transcript"       : t_line,
                "gold"             : best_gold_line,
                "score"            : round(best_score, 3),
                "speaker_gold"     : best_speaker,
                "speaker_original" : original_speaker,
                "aligne"           : best_score >= SCORE_THRESHOLD,
            })

        total_aligned   = sum(1 for r in results if r["aligne"])
        unaligned_count = len(results) - total_aligned

        return results, corrections, transcript_corrige, {
            "total_aligned"  : total_aligned,
            "unaligned_count": unaligned_count,
            "corrections"    : len(corrections),
        }

    @classmethod
    def run_analyse_sts(cls, file_gold, file_transcript, return_corpus=False):
        """Lance l'analyse STS complète et affiche les résultats dans Streamlit."""
        corpus_gold       = Corpus.load_json_file(file_gold)
        corpus_transcript = Corpus.load_json_file(file_transcript)

        if not corpus_gold or not corpus_transcript:
            st.error("Impossible de charger les fichiers JSON.")
            return

        gold_repliques   = corpus_gold.get("repliques", [])
        gold_lines       = [r["line"] for r in gold_repliques]
        gold_speakers_map = {
            v["name"]: v["id"]
            for v in SpeakerEnricher.enrich_speakers(corpus_gold["speakers"])
        }

        gold_speakers = [
            gold_speakers_map.get(r["speaker"], r["speaker"])
            for r in gold_repliques
        ]

        transcript_items = corpus_transcript.get("repliques", {})
        # Supporte liste ou dict
        if isinstance(transcript_items, list):
            transcript_keys  = [str(i) for i in range(len(transcript_items))]
            transcript_lines = [r["line"] for r in transcript_items]
            transcript_items = {str(i): transcript_items[i] for i in range(len(transcript_items))}
        else:
            transcript_keys  = list(transcript_items.keys())
            transcript_lines = [transcript_items[k]["line"] for k in transcript_keys]

        moyenne     = cls.moy_phrases_par_scene(gold_repliques)
        window_size = int(round(moyenne))

        n_gold       = len(gold_lines)
        n_transcript = len(transcript_lines)

        st.info(f"Gold : {n_gold} répliques | Transcript : {n_transcript} répliques | Fenêtre : ±{window_size}")

        model = ModelLoader.load_model(MODEL_SAVE_DIR)

        with st.spinner("Encodage du corpus gold…"):
            gold_embeddings = model.encode(gold_lines, convert_to_tensor=True)

        with st.spinner("Alignement en cours…"):
            results, corrections, transcript_corrige, metrics = cls.run_alignment(
                model, gold_embeddings, gold_lines, gold_speakers, gold_repliques,
                transcript_keys, transcript_lines, transcript_items,
                n_gold, n_transcript, window_size, corpus_transcript,
            )

        # Ajout des champs du gold manquants (scenes) — comme comparaison_sent_to_sent
        if "scenes" not in transcript_corrige and "scenes" in corpus_gold:
            transcript_corrige["scenes"] = corpus_gold["scenes"]
        # Renommage scene_number → scene_index dans scenes
        if "scenes" in transcript_corrige:
            for scene in (transcript_corrige["scenes"] if isinstance(transcript_corrige["scenes"], list)
                        else transcript_corrige["scenes"].values()):
                if "scene_number" in scene:
                    scene["scene_index"] = scene.pop("scene_number")
        # speakers : toujours ecrase par celui du gold (reference authoritative) + enrichissement
        if "speakers" in corpus_gold:
            transcript_corrige["speakers"] = SpeakerEnricher.enrich_speakers(corpus_gold["speakers"])

        # ── Affichage des métriques ──
        st.markdown("#### Quelques chiffres...")
        col1, col2, col3 = st.columns(3)
        col1.metric("Alignées",     metrics["total_aligned"])
        col2.metric("Non alignées", metrics["unaligned_count"])
        col3.metric("Corrections",  metrics["corrections"])

        # ── Tableau des alignements ──
        with st.expander("📋 Détail des alignements"):
            df_results = pd.DataFrame(results)
            df_results["aligne"] = df_results["aligne"].map({True: "✓", False: "✗"})
            st.dataframe(df_results, use_container_width=True)

        # ── Corrections de speakers ──
        with st.expander(f"🔧 Corrections de speakers ({len(corrections)})"):
            if corrections:
                st.dataframe(pd.DataFrame(corrections), use_container_width=True)
            else:
                st.info("Aucune correction effectuée.")

        # ── Téléchargement ──
        if not return_corpus:
            Corpus.to_download_json(transcript_corrige, "transcription_corrigee_sts.json")

        # Attribution du scene_index et du repl_id aux mots
        transcript_corrige = SpeakerEnricher.assign_scene_index_to_words(transcript_corrige)

        if return_corpus:
            return transcript_corrige
