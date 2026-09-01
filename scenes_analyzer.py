import json
import re
from dataclasses import dataclass

import pandas as pd
import streamlit as st
from minineedle import core, needle
import torch
from sentence_transformers import util

from config import MODEL_SAVE_DIR
from corpus import Corpus
from model_loader import ModelLoader
from speaker_enricher import SpeakerEnricher

SCORE_THRESHOLD = 0.5

# Phrase à partir de laquelle l'indexation par scène commence : toute
# réplique avant elle (incluse) n'est pas indexée (scene_index remis à None).
SENTENCE_INDEXING_START = "I'll never strike somebody someone else and myself now looking to mirror local."

@dataclass
class ScenesAnalyzer:

    @classmethod
    def is_unassigned(cls, speaker_value) -> bool:
        """Une réplique est considérée 'non assignée' si son speaker_id est de type int
        (c-à-d pas encore normalisé en 'S1', 'S2'... par SpeakerEnricher)."""
        return isinstance(speaker_value, int)

    @classmethod
    def segmentation_by_scene(cls, repliques) -> dict:
        """Fonction qui segmente le corpus en une liste de scène.
        Retourne un dict {scene_index: [répliques...]} (clé None incluse si présente)."""
        items = repliques if isinstance(repliques, list) else list(repliques.values())
        scenes = {}
        for r in items:
            scenes.setdefault(r.get("scene_index"), []).append(r)
        return scenes

    @classmethod
    def infer_scene_index_voisinage(cls, items: list, idx: int):
        """Récupère le scene_index de la réplique précédente et suivante (les plus proches
        ayant un scene_index non nul), et infère le scene_index de la réplique courante
        uniquement si les deux coïncident."""
        scene_avant = None
        for j in range(idx - 1, -1, -1):
            si = items[j].get("scene_index")
            if si is not None:
                scene_avant = si
                break

        scene_apres = None
        for j in range(idx + 1, len(items)):
            si = items[j].get("scene_index")
            if si is not None:
                scene_apres = si
                break

        # Un seul voisin trouvé (ou aucun) → on ne fait rien
        if scene_avant is None or scene_apres is None:
            return None
        # Scènes différentes → on ne fait rien
        if scene_avant != scene_apres:
            return None
        # Scènes identiques → c'est la bonne scène
        return scene_avant

    @classmethod
    def is_scene_index_artifact(cls, items: list, idx: int) -> bool:
        """Détecte si la réplique à l'index idx est un artéfact de scene_index :
        une réplique isolée (seule occurrence à cette position) dont le
        scene_index diffère de celui de ses deux voisines immédiates, alors
        que ces deux voisines partagent le même scene_index entre elles.
        Ex. : scn1, scn1, scn1, scn1, scn3, scn1, scn1, scn1 → la réplique
        "scn3" est un artéfact (voisin avant = voisin après = scn1 ≠ scn3)."""
        current = items[idx].get("scene_index")
        if current is None:
            return False

        voisin_avant = items[idx - 1].get("scene_index") if idx - 1 >= 0 else None
        voisin_apres = items[idx + 1].get("scene_index") if idx + 1 < len(items) else None

        if voisin_avant is None or voisin_apres is None:
            return False

        return voisin_avant == voisin_apres and voisin_avant != current

    @classmethod
    def remove_scene_index_artifacts(cls, corpus_transcript: dict) -> tuple:
        """Dernière vérification : repère et SUPPRIME (au lieu de corriger) les
        répliques dont le scene_index est un artéfact isolé, entouré des deux
        côtés par le même scene_index (cf. is_scene_index_artifact). La
        réplique est retirée du corpus ; le scene_index des autres répliques
        n'est jamais modifié — seule la suppression a lieu, pas de
        réindexation.
        Retourne (corpus_transcript, artefacts_supprimés)."""
        repliques = corpus_transcript.get("repliques", {})
        items = repliques if isinstance(repliques, list) else list(repliques.values())

        est_artefact = [cls.is_scene_index_artifact(items, idx) for idx in range(len(items))]

        artefacts_supprimes = [
            {
                "line"       : items[idx].get("line", ""),
                "scene_index": items[idx].get("scene_index"),
                "position"   : idx,
            }
            for idx, artefact in enumerate(est_artefact) if artefact
        ]

        if isinstance(repliques, list):
            corpus_transcript["repliques"] = [
                r for idx, r in enumerate(items) if not est_artefact[idx]
            ]
        else:
            keys = list(repliques.keys())
            corpus_transcript["repliques"] = {
                k: repliques[k] for i, k in enumerate(keys) if not est_artefact[i]
            }

        return corpus_transcript, artefacts_supprimes

    @classmethod
    def find_sentence_index(cls, items: list, target_sentence: str, score_threshold: float = SCORE_THRESHOLD):
        """Recherche dans items (repliques) la position de la réplique
        correspondant à target_sentence : d'abord une correspondance exacte
        (texte normalisé via remove_punct, insensible à la ponctuation/casse),
        et à défaut la réplique sémantiquement la plus proche (au-dessus de
        score_threshold), pour couvrir une "phrase équivalente".
        Retourne l'index trouvé, ou None si aucune correspondance."""
        if not items:
            return None

        target_norm = cls.remove_punct(target_sentence)
        for idx, r in enumerate(items):
            if target_norm and cls.remove_punct(r.get("line", "")) == target_norm:
                return idx

        model             = ModelLoader.load_model(MODEL_SAVE_DIR)
        lines             = [r.get("line", "") for r in items]
        target_embedding  = model.encode([target_sentence], convert_to_tensor=True)
        lines_embeddings  = model.encode(lines, convert_to_tensor=True)
        scores            = util.cos_sim(target_embedding, lines_embeddings)[0]
        best_idx          = torch.argmax(scores).item()
        best_score        = scores[best_idx].item()

        return best_idx if best_score >= score_threshold else None

    @classmethod
    def exclude_indexing_before_sentence(
        cls, corpus_transcript: dict, target_sentence: str = SENTENCE_INDEXING_START
    ) -> tuple:
        """N'indexe pas les répliques situées avant target_sentence, celle-ci
        incluse : la clé scene_index est purement supprimée de ces répliques
        ET des mots (words) correspondants (via repl_id) — pas remise à None,
        la clé n'existe plus. La recherche de target_sentence accepte une
        phrase équivalente (similarité sémantique) si aucun texte identique
        (normalisé) n'est trouvé.
        Ne touche à rien après cette borne, et ne supprime aucune réplique
        ni aucun mot (contrairement à remove_scene_index_artifacts) — seule
        la clé scene_index est retirée, aux deux niveaux.
        Retourne (corpus_transcript, index_trouvé, répliques_désindexées,
        mots_désindexés)."""
        repliques = corpus_transcript.get("repliques", {})
        items = repliques if isinstance(repliques, list) else list(repliques.values())
        # rid des repliques : position (list) ou clé du dict, normalisée en
        # int quand possible — même logique que SpeakerEnricher.assign_scene_index_to_words,
        # pour que la correspondance avec le repl_id des mots soit fiable.
        rids = list(range(len(items))) if isinstance(repliques, list) else list(repliques.keys())

        idx_cible = cls.find_sentence_index(items, target_sentence)

        repliques_desindexees = []
        repl_ids_exclus = set()
        if idx_cible is not None:
            for i in range(idx_cible + 1):
                try:
                    repl_ids_exclus.add(int(rids[i]))
                except (ValueError, TypeError):
                    repl_ids_exclus.add(rids[i])

                if "scene_index" in items[i]:
                    repliques_desindexees.append({
                        "line"       : items[i].get("line", ""),
                        "scene_index": items[i]["scene_index"],
                        "position"   : i,
                    })
                    del items[i]["scene_index"]

        if isinstance(repliques, list):
            corpus_transcript["repliques"] = items
        else:
            keys = list(corpus_transcript["repliques"].keys())
            corpus_transcript["repliques"] = {keys[i]: items[i] for i in range(len(keys))}

        # ── Propagation aux mots (words) portant le même repl_id ──
        words = corpus_transcript.get("words", {})
        words_items = words if isinstance(words, list) else list(words.values())

        mots_desindexes = []
        for w in words_items:
            if w.get("repl_id") in repl_ids_exclus and "scene_index" in w:
                mots_desindexes.append({
                    "content"    : w.get("content", ""),
                    "repl_id"    : w.get("repl_id"),
                    "scene_index": w["scene_index"],
                })
                del w["scene_index"]

        return corpus_transcript, idx_cible, repliques_desindexees, mots_desindexes

    @classmethod
    def run_scene_index_inference(cls, corpus_transcript: dict) -> tuple:
        """Étape 1 : pour chaque réplique non assignée sans scene_index, tente de
        l'inférer à partir des scene_index des répliques voisines."""
        repliques = corpus_transcript.get("repliques", {})
        items = repliques if isinstance(repliques, list) else list(repliques.values())

        inferences = []
        for idx, r in enumerate(items):
            speaker_value = r.get("speaker_id", r.get("speaker"))
            if cls.is_unassigned(speaker_value) and r.get("scene_index") is None:
                inferred = cls.infer_scene_index_voisinage(items, idx)
                if inferred is not None:
                    r["scene_index"] = inferred
                    inferences.append({
                        "line"              : r.get("line", ""),
                        "speaker_id"        : speaker_value,
                        "scene_index_inféré": inferred,
                    })

        if isinstance(repliques, list):
            corpus_transcript["repliques"] = items
        else:
            keys = list(corpus_transcript["repliques"].keys())
            corpus_transcript["repliques"] = {keys[i]: items[i] for i in range(len(keys))}

        return corpus_transcript, inferences

    @classmethod
    def _prochain_id_speaker(cls, registry: dict) -> str:
        """Prochain id 'Sx' disponible : le numéro le plus haut déjà utilisé
        dans le registre + 1 — pas simplement len(registry) + 1, qui produirait
        des collisions si les id existants ont des trous ou ne suivent pas
        ce format (ex. speakers désynchronisé de son ordre d'origine)."""
        max_n = 0
        for id_ in registry.values():
            if isinstance(id_, str) and id_.startswith("S") and id_[1:].isdigit():
                max_n = max(max_n, int(id_[1:]))
        return f"S{max_n + 1}"

    @classmethod
    def fix_literal_speaker_names(cls, corpus_transcript: dict) -> tuple:
        """Nettoyage : remplace, dans 'repliques' (et les 'words' correspondants
        via repl_id), tout speaker_id encore écrit en clair (ex. "CLAIRE" au
        lieu de "S4") par son id — résidu de fichiers générés avant la
        correction de run_correction_speaker/run_boundary_matching. Réutilise
        l'id existant du registre {name: id} (basé sur corpus_transcript
        ["speakers"]) si ce nom est déjà connu, sinon en enregistre un
        nouveau et complète 'speakers'. Ne touche ni aux speaker_id déjà au
        format id connu, ni aux speaker_id encore bruts (int, non assignés).
        Retourne (corpus_transcript, corrections)."""
        speakers_transcript = corpus_transcript.get("speakers", [])
        registry = {
            s["name"]: s["id"] for s in speakers_transcript
            if isinstance(s, dict) and "name" in s and "id" in s
        }
        ids_connus     = set(registry.values())
        noms_existants = set(registry.keys())

        repliques = corpus_transcript.get("repliques", {})
        items = repliques if isinstance(repliques, list) else list(repliques.values())
        keys  = list(range(len(items))) if isinstance(repliques, list) else list(repliques.keys())

        corrections = []
        nouveaux_id_par_repl_id = {}  # repl_id (normalisé int si possible) → nouveau speaker_id

        for pos, r in enumerate(items):
            speaker_id = r.get("speaker_id", r.get("speaker"))
            # Nom en clair : une chaîne qui n'est pas déjà un id reconnu du registre
            if isinstance(speaker_id, str) and speaker_id not in ids_connus:
                if speaker_id not in registry:
                    registry[speaker_id] = cls._prochain_id_speaker(registry)
                    ids_connus.add(registry[speaker_id])
                nouveau_id = registry[speaker_id]

                corrections.append({
                    "line"              : r.get("line", ""),
                    "ancien_speaker_id" : speaker_id,
                    "nouveau_speaker_id": nouveau_id,
                })
                r["speaker_id"] = nouveau_id

                try:
                    repl_id_norm = int(keys[pos])
                except (ValueError, TypeError):
                    repl_id_norm = keys[pos]
                nouveaux_id_par_repl_id[repl_id_norm] = nouveau_id

        if isinstance(repliques, list):
            corpus_transcript["repliques"] = items
        else:
            corpus_transcript["repliques"] = {keys[i]: items[i] for i in range(len(keys))}

        # Propage aux mots correspondants (même repl_id)
        words = corpus_transcript.get("words", {})
        words_items = words if isinstance(words, list) else list(words.values())
        for w in words_items:
            if w.get("repl_id") in nouveaux_id_par_repl_id:
                w["speaker_id"] = nouveaux_id_par_repl_id[w["repl_id"]]

        # Complète la liste des locuteurs avec tout nom nouvellement enregistré
        for nom, id_ in registry.items():
            if nom not in noms_existants:
                speakers_transcript.append({"name": nom, "id": id_})
        corpus_transcript["speakers"] = speakers_transcript

        return corpus_transcript, corrections

    @classmethod
    def run_correction_speaker(cls, corpus_transcript: dict, corpus_gold: dict, registry: dict = None) -> tuple:
        """Étape 2 : pour chaque réplique non assignée dont le scene_index est connu,
        recherche la réplique gold la plus proche sémantiquement dans la même scène
        (comme STSAnalyzer) et corrige le speaker_id.

        Les id de locuteurs proviennent d'un registre {name: id} — passé par
        l'appelant (registry) pour rester partagé et cohérent avec les autres
        étapes de scene_analyze (ex. run_boundary_matching), ou, à défaut,
        initialisé ici depuis corpus_transcript["speakers"] pour un appel
        autonome. Un nom déjà connu réutilise son id existant, un nom encore
        inconnu (ex. "CLAIRE") en reçoit un nouveau — jamais écrit tel quel
        comme speaker_id. Si registry n'est pas fourni, les nouveaux noms sont
        aussi ajoutés à corpus_transcript["speakers"] en sortie (sinon, c'est
        à l'appelant de le faire une fois le registre partagé stabilisé)."""
        gold_repliques = corpus_gold.get("repliques", [])
        gold_by_scene  = cls.segmentation_by_scene(gold_repliques)

        speakers_transcript = corpus_transcript.get("speakers", [])
        registre_autonome = registry is None
        if registre_autonome:
            registry = {
                s["name"]: s["id"] for s in speakers_transcript
                if isinstance(s, dict) and "name" in s and "id" in s
            }

        repliques = corpus_transcript.get("repliques", {})
        items = repliques if isinstance(repliques, list) else list(repliques.values())

        # Répliques éligibles : non assignées + scene_index connu et présent dans le gold
        candidates_idx = [
            i for i, r in enumerate(items)
            if cls.is_unassigned(r.get("speaker_id", r.get("speaker")))
            and r.get("scene_index") is not None
            and r.get("scene_index") in gold_by_scene
        ]

        if not candidates_idx:
            return corpus_transcript, [], []

        model = ModelLoader.load_model(MODEL_SAVE_DIR)

        # ── OPTIMISATION : encodage batch de toutes les répliques candidates en un seul appel ──
        t_lines      = [items[i].get("line", "") for i in candidates_idx]
        t_embeddings = model.encode(t_lines, convert_to_tensor=True)

        # ── Cache des embeddings gold par scene_index : évite le ré-encodage si plusieurs
        #    répliques candidates partagent la même scène ──
        gold_embeddings_cache = {}
        corrections = []
        matches     = []  # toutes les correspondances candidates (acceptées ou non), pour affichage par scène

        for local_i, idx in enumerate(candidates_idx):
            r               = items[idx]
            scene_index     = r.get("scene_index")
            gold_candidates = gold_by_scene[scene_index]

            if scene_index not in gold_embeddings_cache:
                gold_lines = [g.get("line", "") for g in gold_candidates]
                gold_embeddings_cache[scene_index] = model.encode(gold_lines, convert_to_tensor=True)
            gold_embeddings = gold_embeddings_cache[scene_index]

            scores     = util.cos_sim(t_embeddings[local_i], gold_embeddings)[0]
            best_idx   = torch.argmax(scores).item()
            best_score = scores[best_idx].item()

            best_gold       = gold_candidates[best_idx]
            ancien_speaker  = r.get("speaker_id", r.get("speaker"))
            nom_gold        = best_gold.get("speaker", "")
            # Réutilise l'id existant du registre, ou en enregistre un nouveau
            # si ce nom n'a encore jamais été vu (au lieu d'écrire le nom brut).
            if nom_gold and nom_gold not in registry:
                registry[nom_gold] = cls._prochain_id_speaker(registry)
            nouveau_speaker = registry.get(nom_gold, nom_gold)
            accepte         = best_score >= SCORE_THRESHOLD

            matches.append({
                "scene_index"    : scene_index,
                "line"           : r.get("line", ""),
                "gold_phrase"    : best_gold.get("line", ""),
                "ancien_speaker" : ancien_speaker,
                "nouveau_speaker": nouveau_speaker,
                "score"          : round(best_score, 3),
                "accepté"        : "✅" if accepte else "❌",
            })

            if accepte:
                r["speaker_id"] = nouveau_speaker
                corrections.append({
                    "line"           : r.get("line", ""),
                    "scene_index"    : scene_index,
                    "ancien_speaker" : ancien_speaker,
                    "nouveau_speaker": nouveau_speaker,
                    "gold_phrase"    : best_gold.get("line", ""),
                    "score"          : round(best_score, 3),
                })

        if isinstance(repliques, list):
            corpus_transcript["repliques"] = items
        else:
            keys = list(corpus_transcript["repliques"].keys())
            corpus_transcript["repliques"] = {keys[i]: items[i] for i in range(len(keys))}

        # Complète la liste des locuteurs avec tout nom nouvellement enregistré
        # (seulement en appel autonome — sinon l'appelant synchronise une fois
        # le registre partagé stabilisé, après toutes les étapes concernées)
        if registre_autonome:
            noms_existants = {s["name"] for s in speakers_transcript if isinstance(s, dict) and "name" in s}
            for nom, id_ in registry.items():
                if nom not in noms_existants:
                    speakers_transcript.append({"name": nom, "id": id_})
            corpus_transcript["speakers"] = speakers_transcript

        return corpus_transcript, corrections, matches

    @classmethod
    def find_scene_span(cls, items: list, scene_index) -> tuple:
        """Retourne (min_idx, max_idx) : la plage d'indices (dans items)
        déjà connue pour un scene_index donné dans le transcript, ou
        (None, None) si aucune réplique n'a encore ce scene_index. Ça
        correspond au "découpage par scène" (début/fin) déjà repéré."""
        indices = [i for i, r in enumerate(items) if r.get("scene_index") == scene_index]
        if not indices:
            return None, None
        return min(indices), max(indices)

    @classmethod
    def run_boundary_matching(cls, corpus_transcript: dict, corpus_gold: dict, registry: dict = None) -> tuple:
        """Étape 3, complémentaire à run_scene_index_inference/run_correction_speaker :
        pour chaque scène déjà repérée dans le transcript (au moins une
        réplique avec ce scene_index), compare le texte normalisé (sans
        ponctuation/casse, via remove_punct) de la PREMIÈRE et de la
        DERNIÈRE réplique de la MÊME scène côté gold ("la scène 1 du
        gold" vs. "la potentielle scène 1 du transcript") aux répliques
        non assignées immédiatement adjacentes à cette scène — pour en
        retrouver le début et la fin exacts et compléter speaker_id/
        scene_index sur ces répliques de bordure.

        Comme run_correction_speaker, les id de locuteurs proviennent d'un
        registre {name: id} partagé (registry) — ou initialisé ici depuis
        corpus_transcript["speakers"] si appelé seul — pour qu'un nom encore
        inconnu (ex. "CLAIRE") reçoive un id au lieu d'être écrit tel quel."""
        gold_repliques = corpus_gold.get("repliques", [])

        speakers_transcript = corpus_transcript.get("speakers", [])
        registre_autonome = registry is None
        if registre_autonome:
            registry = {
                s["name"]: s["id"] for s in speakers_transcript
                if isinstance(s, dict) and "name" in s and "id" in s
            }

        def _resoudre_speaker(nom_gold: str):
            if nom_gold and nom_gold not in registry:
                registry[nom_gold] = cls._prochain_id_speaker(registry)
            return registry.get(nom_gold, nom_gold)

        gold_by_scene = cls.segmentation_by_scene(gold_repliques)

        repliques = corpus_transcript.get("repliques", {})
        items = repliques if isinstance(repliques, list) else list(repliques.values())

        corrections = []

        for scene_index, gold_items in gold_by_scene.items():
            if scene_index is None or not gold_items:
                continue

            # ── Découpage par scène : bornes déjà connues dans le transcript ──
            min_idx, max_idx = cls.find_scene_span(items, scene_index)
            if min_idx is None:
                continue  # scène pas encore repérée dans le transcript : rien à étendre

            premiere_gold = gold_items[0]
            derniere_gold = gold_items[-1]
            premiere_norm = cls.remove_punct(premiere_gold.get("line", ""))
            derniere_norm = cls.remove_punct(derniere_gold.get("line", ""))
            speaker_premier = _resoudre_speaker(premiere_gold.get("speaker", ""))
            speaker_dernier = _resoudre_speaker(derniere_gold.get("speaker", ""))

            # ── Remonte avant min_idx : recherche du début de la scène ──
            j = min_idx - 1
            while j >= 0 and cls.is_unassigned(items[j].get("speaker_id", items[j].get("speaker"))):
                texte_norm = cls.remove_punct(items[j].get("line", ""))
                if texte_norm and texte_norm == premiere_norm:
                    items[j]["scene_index"] = scene_index
                    items[j]["speaker_id"] = speaker_premier
                    corrections.append({
                        "line"       : items[j].get("line", ""),
                        "position"   : "début",
                        "scene_index": scene_index,
                        "speaker_id" : speaker_premier,
                    })
                    break  # une seule réplique de bordure attendue
                j -= 1

            # ── Avance après max_idx : recherche de la fin de la scène ──
            j = max_idx + 1
            while j < len(items) and cls.is_unassigned(items[j].get("speaker_id", items[j].get("speaker"))):
                texte_norm = cls.remove_punct(items[j].get("line", ""))
                if texte_norm and texte_norm == derniere_norm:
                    items[j]["scene_index"] = scene_index
                    items[j]["speaker_id"] = speaker_dernier
                    corrections.append({
                        "line"       : items[j].get("line", ""),
                        "position"   : "fin",
                        "scene_index": scene_index,
                        "speaker_id" : speaker_dernier,
                    })
                    break
                j += 1

        if isinstance(repliques, list):
            corpus_transcript["repliques"] = items
        else:
            keys = list(corpus_transcript["repliques"].keys())
            corpus_transcript["repliques"] = {keys[i]: items[i] for i in range(len(keys))}

        if registre_autonome:
            noms_existants = {s["name"] for s in speakers_transcript if isinstance(s, dict) and "name" in s}
            for nom, id_ in registry.items():
                if nom not in noms_existants:
                    speakers_transcript.append({"name": nom, "id": id_})
            corpus_transcript["speakers"] = speakers_transcript

        return corpus_transcript, corrections

    @classmethod
    def scene_analyze(cls, file_gold, corpus_in=None, return_corpus=False):
        """Fonction qui analyse le dialogue d'une scène + corrige le speaker
        pour les phrases non assignées (approche complémentaire à STS/Hapax) :
        1) inférence du scene_index par voisinage, 2) correction du speaker_id
        par similarité sémantique restreinte à la scène inférée, 3) repérage
        des bornes (début/fin) de chaque scène déjà repérée, par comparaison
        texte (sans ponctuation/casse) avec la même scène côté gold."""
        corpus_gold = Corpus.load_json_file(file_gold)

        if corpus_in is None:
            st.error("Aucun corpus en entrée pour l'analyse par scènes.")
            return None
        if not corpus_gold:
            st.error("Impossible de charger le fichier gold.")
            return corpus_in if return_corpus else None

        corpus_transcript = json.loads(json.dumps(corpus_in))

        with st.spinner("Inférence des scene_index par voisinage…"):
            corpus_transcript, inferences = cls.run_scene_index_inference(corpus_transcript)

        # Registre {name: id} partagé entre les étapes 2 et 3, pour qu'un nom
        # nouvellement rencontré (ex. "CLAIRE") reçoive le même id partout
        # dans scene_analyze plutôt que d'être recalculé (ou oublié) à
        # chaque étape.
        registry = {
            s["name"]: s["id"] for s in corpus_transcript.get("speakers", [])
            if isinstance(s, dict) and "name" in s and "id" in s
        }

        with st.spinner("Correction des speaker_id par similarité sémantique…"):
            corpus_transcript, corrections, matches = cls.run_correction_speaker(
                corpus_transcript, corpus_gold, registry=registry
            )

        with st.spinner("Repérage des bornes de scène (début/fin)…"):
            corpus_transcript, corrections_bornes = cls.run_boundary_matching(
                corpus_transcript, corpus_gold, registry=registry
            )

        # Synchronise la liste des locuteurs une fois le registre stabilisé
        # par les deux étapes ci-dessus.
        speakers_transcript = corpus_transcript.get("speakers", [])
        noms_existants = {s["name"] for s in speakers_transcript if isinstance(s, dict) and "name" in s}
        for nom, id_ in registry.items():
            if nom not in noms_existants:
                speakers_transcript.append({"name": nom, "id": id_})
        corpus_transcript["speakers"] = speakers_transcript

        st.info(
            f"Scene_index inférés : {len(inferences)} | "
            f"Speakers corrigés (sémantique) : {len(corrections)} | "
            f"Répliques de bordure retrouvées : {len(corrections_bornes)}"
        )

        with st.expander(f"🧭 Scene_index inférés par voisinage ({len(inferences)})"):
            if inferences:
                st.dataframe(pd.DataFrame(inferences), use_container_width=True)
            else:
                st.info("Aucun scene_index inféré.")

        with st.expander(f"🎬 Répliques de début/fin de scène retrouvées ({len(corrections_bornes)})"):
            if corrections_bornes:
                st.dataframe(pd.DataFrame(corrections_bornes), use_container_width=True)
            else:
                st.info("Aucune réplique de bordure retrouvée.")

        st.markdown("#### 🔧 Correspondances de phrases par scène")
        if matches:
            matches_by_scene = {}
            for m in matches:
                matches_by_scene.setdefault(m["scene_index"], []).append(m)

            for scene_index in sorted(matches_by_scene.keys(), key=lambda x: (x is None, x)):
                scene_matches = matches_by_scene[scene_index]
                n_acceptees   = sum(1 for m in scene_matches if m["accepté"] == "✅")
                with st.expander(f"Scène {scene_index} — {n_acceptees}/{len(scene_matches)} correspondance(s) acceptée(s)"):
                    st.dataframe(pd.DataFrame(scene_matches), use_container_width=True)
        else:
            st.info("Aucune correspondance à afficher.")

        if not return_corpus:
            Corpus.to_download_json(corpus_transcript, "transcription_corrigee_scenes.json")

        # Filet de sécurité : au cas où un speaker_id serait resté écrit en
        # clair (nom au lieu d'un id "Sx") à l'issue des étapes ci-dessus.
        with st.spinner("Vérification des speaker_id en clair…"):
            corpus_transcript, corrections_noms = cls.fix_literal_speaker_names(corpus_transcript)

        if corrections_noms:
            with st.expander(f"🏷️ Speaker_id en clair corrigés ({len(corrections_noms)})"):
                st.dataframe(pd.DataFrame(corrections_noms), use_container_width=True)

        # Cohérence avec STSAnalyzer/HapaxAnalyzer : propage scene_index/speaker_id aux mots
        corpus_transcript = SpeakerEnricher.assign_scene_index_to_words(corpus_transcript)

        # Dernière vérification : désindexe tout ce qui précède (et inclut) la phrase de coupure
        with st.spinner("Désindexation avant la phrase de coupure…"):
            corpus_transcript, idx_cible, repliques_desindexees, mots_desindexes = cls.exclude_indexing_before_sentence(
                corpus_transcript
            )

        with st.expander(
            f"✂️ Désindexation avant la phrase de coupure "
            f"({len(repliques_desindexees)} réplique(s), {len(mots_desindexes)} mot(s))"
        ):
            if idx_cible is not None:
                st.dataframe(pd.DataFrame(repliques_desindexees), use_container_width=True)
            else:
                st.info("Phrase de coupure introuvable — aucune désindexation appliquée.")

        if return_corpus:
            return corpus_transcript

    @classmethod
    def remove_punct(cls, text: str) -> str:
        """Prétraitement : retire la ponctuation d'une phrase et met en
        minuscules, pour comparer deux répliques sans tenir compte de la
        ponctuation/casse (différences fréquentes entre OCR du gold et
        transcription automatique)."""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def remove_stopwords(cls):
        """prétraitement : retire les stopwords d'une phrase"""

# Caractère de trou pour l'alignement : peu susceptible d'apparaître
# dans un texte normalisé (remove_punct), pour ne pas être confondu
# avec une réplique réelle.
GAP_CHARACTER = "␀"


class AlignmentAnalyzer:

    @classmethod
    def _to_items(cls, repliques) -> list:
        """Normalise 'repliques' (list ou dict) en liste, comme dans ScenesAnalyzer."""
        return repliques if isinstance(repliques, list) else list(repliques.values())

    @classmethod
    def _build_alignment(cls, seq_gold: list, seq_transcript: list):
        """Construit et exécute un alignement global Needleman-Wunsch entre
        deux séquences de textes déjà normalisés (via ScenesAnalyzer.remove_punct).
        Retourne (alignment, sequence_gold_alignée, sequence_transcript_alignée)."""
        alignment = needle.NeedlemanWunsch(seq_gold, seq_transcript)
        alignment.gap_character = GAP_CHARACTER
        alignment.align()
        al_gold, al_transcript = alignment.get_aligned_sequences(core.AlignmentFormat.list)
        return alignment, al_gold, al_transcript

    # ── Étape 1 : alignement global sur l'ensemble des deux corpus ──────────

    @classmethod
    def run_global_alignment(cls, corpus_gold: dict, corpus_transcript: dict) -> tuple:
        """Étape 1 : alignement global gold vs transcription sur l'ensemble du
        corpus (texte normalisé, sans ponctuation/casse). Ne modifie rien ;
        retourne (alignment, table_correspondance) où table_correspondance est
        une liste de dicts {gold_idx, transcript_idx, gold_line, transcript_line}
        (idx à None côté gold ou transcript en cas de trou/insertion)."""
        gold_items = cls._to_items(corpus_gold.get("repliques", []))
        transcript_items = cls._to_items(corpus_transcript.get("repliques", {}))

        seq_gold = [ScenesAnalyzer.remove_punct(r.get("line", "")) for r in gold_items]
        seq_transcript = [ScenesAnalyzer.remove_punct(r.get("line", "")) for r in transcript_items]

        alignment, al_gold, al_transcript = cls._build_alignment(seq_gold, seq_transcript)

        correspondance = []
        gi, ti = 0, 0
        for g_tok, t_tok in zip(al_gold, al_transcript):
            is_gap_g = g_tok == GAP_CHARACTER
            is_gap_t = t_tok == GAP_CHARACTER
            correspondance.append({
                "gold_idx"       : None if is_gap_g else gi,
                "transcript_idx" : None if is_gap_t else ti,
                "gold_line"      : "" if is_gap_g else gold_items[gi].get("line", ""),
                "transcript_line": "" if is_gap_t else transcript_items[ti].get("line", ""),
            })
            if not is_gap_g:
                gi += 1
            if not is_gap_t:
                ti += 1

        return alignment, correspondance

    # ── Étape 2 : alignement global par scène + correction des speaker_id ──

    @classmethod
    def run_scene_alignment_correction(cls, corpus_transcript: dict, corpus_gold: dict, registry: dict = None) -> tuple:
        """Étape 2 : pour chaque scène, alignement global (NeedlemanWunsch)
        gold vs transcript restreint à cette scène. Corrige le speaker_id des
        répliques non assignées (ScenesAnalyzer.is_unassigned) qui s'alignent
        (sans trou de part et d'autre) avec une réplique gold.

        Comme ScenesAnalyzer.run_correction_speaker, les id de locuteurs
        proviennent d'un registre {name: id} partagé (registry) — ou
        initialisé ici depuis corpus_transcript["speakers"] si appelé seul —
        pour qu'un nom encore inconnu reçoive un id au lieu d'être écrit tel
        quel. En appel autonome, corpus_transcript["speakers"] est complété
        avec les nouveaux noms enregistrés."""
        gold_repliques = corpus_gold.get("repliques", [])

        speakers_transcript = corpus_transcript.get("speakers", [])
        registre_autonome = registry is None
        if registre_autonome:
            registry = {
                s["name"]: s["id"] for s in speakers_transcript
                if isinstance(s, dict) and "name" in s and "id" in s
            }

        gold_by_scene = ScenesAnalyzer.segmentation_by_scene(gold_repliques)

        repliques = corpus_transcript.get("repliques", {})
        items = cls._to_items(repliques)
        transcript_by_scene = ScenesAnalyzer.segmentation_by_scene(items)

        corrections = []
        alignments_par_scene = {}

        for scene_index, transcript_scene_items in transcript_by_scene.items():
            if scene_index is None or scene_index not in gold_by_scene:
                continue

            gold_scene_items = gold_by_scene[scene_index]

            seq_gold = [ScenesAnalyzer.remove_punct(r.get("line", "")) for r in gold_scene_items]
            seq_transcript = [ScenesAnalyzer.remove_punct(r.get("line", "")) for r in transcript_scene_items]

            if not seq_gold or not seq_transcript:
                continue

            alignment, al_gold, al_transcript = cls._build_alignment(seq_gold, seq_transcript)
            alignments_par_scene[scene_index] = alignment

            gi, ti = 0, 0
            for g_tok, t_tok in zip(al_gold, al_transcript):
                is_gap_g = g_tok == GAP_CHARACTER
                is_gap_t = t_tok == GAP_CHARACTER

                if not is_gap_g and not is_gap_t:
                    r_transcript = transcript_scene_items[ti]
                    r_gold = gold_scene_items[gi]
                    ancien_speaker = r_transcript.get("speaker_id", r_transcript.get("speaker"))

                    if ScenesAnalyzer.is_unassigned(ancien_speaker):
                        nom_gold = r_gold.get("speaker", "")
                        if nom_gold and nom_gold not in registry:
                            registry[nom_gold] = ScenesAnalyzer._prochain_id_speaker(registry)
                        nouveau_speaker = registry.get(nom_gold, nom_gold)
                        r_transcript["speaker_id"] = nouveau_speaker
                        corrections.append({
                            "scene_index"    : scene_index,
                            "line"           : r_transcript.get("line", ""),
                            "gold_phrase"    : r_gold.get("line", ""),
                            "ancien_speaker" : ancien_speaker,
                            "nouveau_speaker": nouveau_speaker,
                        })

                if not is_gap_g:
                    gi += 1
                if not is_gap_t:
                    ti += 1

        if isinstance(repliques, list):
            corpus_transcript["repliques"] = items
        else:
            keys = list(corpus_transcript["repliques"].keys())
            corpus_transcript["repliques"] = {keys[i]: items[i] for i in range(len(keys))}

        if registre_autonome:
            noms_existants = {s["name"] for s in speakers_transcript if isinstance(s, dict) and "name" in s}
            for nom, id_ in registry.items():
                if nom not in noms_existants:
                    speakers_transcript.append({"name": nom, "id": id_})
            corpus_transcript["speakers"] = speakers_transcript

        return corpus_transcript, corrections, alignments_par_scene

    # ── Affichage Streamlit ──────────────────────────────────────────────

    @classmethod
    def display_global_alignment(cls, correspondance: list) -> None:
        """Affiche (Streamlit) le résultat de l'alignement global corpus entier."""
        st.markdown("#### 🧬 Alignement global (Needleman-Wunsch) — gold vs transcription")
        if correspondance:
            df = pd.DataFrame(correspondance)
            n_insertions_transcript = df["gold_idx"].isna().sum()
            n_insertions_gold = df["transcript_idx"].isna().sum()
            st.info(
                f"{len(df)} positions alignées | "
                f"{n_insertions_transcript} insertion(s) côté transcription | "
                f"{n_insertions_gold} insertion(s) côté gold"
            )
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Rien à aligner.")

    @classmethod
    def display_scene_corrections(cls, corrections: list) -> None:
        """Affiche (Streamlit) les corrections de speaker_id issues de
        l'alignement global par scène."""
        st.markdown("#### 🧬 Corrections de speaker_id par alignement global (par scène)")
        if corrections:
            df = pd.DataFrame(corrections)
            for scene_index in sorted(df["scene_index"].unique(), key=lambda x: (x is None, x)):
                scene_df = df[df["scene_index"] == scene_index]
                with st.expander(f"Scène {scene_index} — {len(scene_df)} correction(s)"):
                    st.dataframe(scene_df, use_container_width=True)
        else:
            st.info("Aucune correction apportée par alignement global.")


def run_alignment_analysis(file_gold, corpus_in: dict, return_corpus: bool = False):
    """Point d'entrée indépendant, du même esprit que
    ScenesAnalyzer.scene_analyze, mais basé sur minineedle. À utiliser
    seul, ou en complément de ScenesAnalyzer.scene_analyze (par ex. sur
    le corpus déjà partiellement corrigé par ce dernier)."""
    corpus_gold = Corpus.load_json_file(file_gold)
    if not corpus_gold:
        st.error("Impossible de charger le fichier gold.")
        return corpus_in if return_corpus else None

    corpus_transcript = json.loads(json.dumps(corpus_in))

    with st.spinner("Alignement global du corpus (gold vs transcription)…"):
        _, correspondance = AlignmentAnalyzer.run_global_alignment(corpus_gold, corpus_transcript)
    AlignmentAnalyzer.display_global_alignment(correspondance)

    with st.spinner("Alignement global par scène + correction des speaker_id…"):
        corpus_transcript, corrections, _ = AlignmentAnalyzer.run_scene_alignment_correction(
            corpus_transcript, corpus_gold
        )
    AlignmentAnalyzer.display_scene_corrections(corrections)

    if not return_corpus:
        Corpus.to_download_json(corpus_transcript, "transcription_corrigee_alignement.json")

    if return_corpus:
        return corpus_transcript
