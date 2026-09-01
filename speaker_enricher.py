from dataclasses import dataclass


@dataclass
class SpeakerEnricher:

    @classmethod
    def enrich_speakers(cls, speakers_raw) -> list:
        """
        Enrichit le champ speakers avec les métadonnées name et id.
        Accepte une liste ou un dict.
        Retourne toujours une liste : [{"name": ..., "id": ...}, ...]
        """
        if isinstance(speakers_raw, list):
            pairs = list(enumerate(speakers_raw, start=1))
        else:
            pairs = list(speakers_raw.items())
        result = []
        for seq, (k, v) in enumerate(pairs, start=1):
            # v peut être un str (juste le nom) ou déjà un dict
            name = v if isinstance(v, str) else v.get("name", str(v))
            result.append({
                "name": name,
                "id"  : f"S{seq}",
            })
        return result

    @classmethod
    def enrich_speakers_with_registry(cls, speakers_raw, registry: dict) -> list:
        """
        Comme enrich_speakers, mais s'appuie sur un registre persistant
        {name: id} fourni par l'appelant, pour garantir des id stables
        et cohérents à travers plusieurs fichiers/appels successifs.

        L'égalité entre noms est stricte (chaîne de caractères) : deux
        variantes comme "CLAIRE" et "CLAIRE OFF" sont donc toujours
        traitées comme deux individus distincts, avec deux id différents,
        et aucune normalisation ne viendra les fusionner.

        `registry` est modifié en place, afin de pouvoir être réutilisé
        (persisté) lors d'appels ultérieurs.
        """
        if isinstance(speakers_raw, list):
            pairs = list(enumerate(speakers_raw, start=1))
        else:
            pairs = list(speakers_raw.items())

        result = []
        for _, (k, v) in enumerate(pairs, start=1):
            # v peut être un str (juste le nom) ou déjà un dict
            name = v if isinstance(v, str) else v.get("name", str(v))

            if name not in registry:
                registry[name] = f"S{len(registry) + 1}"

            result.append({
                "name": name,
                "id"  : registry[name],
            })
        return result

    @classmethod
    def enrich_speakers_from_repliques(cls, speakers_raw, repliques, registry: dict = None) -> tuple:
        """
        Complète le champ speakers à partir des noms de locuteurs présents
        dans repliques, pour couvrir les cas où un speaker n'apparaît que
        sous une variante (ex: "THOMAS OFF") absente de speakers_raw.

        S'appuie sur enrich_speakers_with_registry pour traiter d'abord
        speakers_raw, puis parcourt repliques : tout nom rencontré qui
        n'est pas déjà dans le registre (l'égalité reste stricte, donc
        "THOMAS" et "THOMAS OFF" sont bien deux noms différents) est
        ajouté comme speaker distinct avec un nouvel id.

        repliques est modifié en place : chaque réplique reçoit le
        speaker_id correspondant à son nom.

        Retourne (speakers, repliques).
        """
        if registry is None:
            registry = {}

        speakers = cls.enrich_speakers_with_registry(speakers_raw, registry)

        # Normalise repliques en dict indexé par clé → réplique
        if isinstance(repliques, list):
            repliques_by_id = {i: r for i, r in enumerate(repliques)}
        else:
            repliques_by_id = repliques

        for rid, r in repliques_by_id.items():
            name = r.get("speaker", r.get("speaker_id", None))
            if not isinstance(name, str):
                continue

            if name not in registry:
                registry[name] = f"S{len(registry) + 1}"
                speakers.append({"name": name, "id": registry[name]})

            r["speaker_id"] = registry[name]

        return speakers, repliques

    @classmethod
    def assign_scene_index_to_words(cls, transcript_corrige: dict) -> dict:
        """
        Attribue le scene_index à chaque mot du champ 'words' en se basant
        directement sur le champ 'repl_id' du mot, qui référence l'index/clé
        de la réplique correspondante dans 'repliques'.
        """
        words     = transcript_corrige.get("words", {})
        repliques = transcript_corrige.get("repliques", {})

        if not words or not repliques:
            return transcript_corrige

        # Normalise repliques en dict indexé par clé → réplique
        if isinstance(repliques, list):
            repliques_by_id = {i: r for i, r in enumerate(repliques)}
        else:
            repliques_by_id = repliques

        # Construit un index : repl_id (int) → scene_index (peut être None)
        scene_index_by_repl_id = {}
        speaker_id_by_repl_id = {}

        for rid, r in repliques_by_id.items():
            try:
                key = int(rid)
            except (ValueError, TypeError):
                key = rid

            scene_index_by_repl_id[key] = r.get("scene_index", None)
            speaker_id_by_repl_id[key] = r.get(
                "speaker_id",
                r.get("speaker", None)
            )
        # Normalise words en liste ordonnée de (key, dict)
        if isinstance(words, list):
            words_pairs = list(enumerate(words))
        else:
            words_pairs = [(k, words[k]) for k in sorted(words.keys(), key=lambda x: int(x))]

        # Attribution directe via repl_id
        for key, word in words_pairs:
            repl_id = word.get("repl_id", None)
            if repl_id is None:
                continue

            scene_index = scene_index_by_repl_id.get(repl_id, None)
            speaker_id = speaker_id_by_repl_id.get(repl_id, None)

            if scene_index is not None:
                word["scene_index"] = scene_index

            if speaker_id is not None:
                word["speaker_id"] = speaker_id

        return transcript_corrige
