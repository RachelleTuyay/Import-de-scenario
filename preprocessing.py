import json
import os
import re

def load_json(input_file):
    """Charge un fichier JSON"""
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Erreur lors du chargement du fichier JSON: {e}")
        return {}

def normaliser_noms(text):
    replacements = [
        (r"\ble cosmétique\b", "L. Cosmétiques"),
        (r"\belle cosmétique\b", "L. Cosmétiques"),
        (r"\bl cosmétique\b", "L. Cosmétiques"),
        (r"\bl cosmétiques\b", "L. Cosmétiques")
    ]
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)
    return text

def to_int_id(value):
    """Convertit un identifiant (ex: 'S19', '19') en int en ne gardant que les chiffres"""
    digits = re.sub(r"\D", "", str(value))
    return int(digits) if digits else None

def normaliser_phrases(replique):
    if not isinstance(replique, str):
        replique = str(replique) if repliques is not None else ""
    replique_clean = re.sub(r"\s+", " ", replique)
    return replique_clean

def words_concatenation(dictionary: dict):
    repliques_finales = {}
    final_id = 1
    current_words = []
    current_keys = []      # garde les clés des mots en cours de regroupement
    current_speaker = None #convertir en int

    for key in dictionary.keys():
        element = dictionary[key]
        content = element["content"]
        speaker_id = to_int_id(element["speaker_id"])

        if current_speaker is None:
            current_speaker = speaker_id

        current_words.append(content)
        current_keys.append(key)

        if re.search(r'[.!?]$', content.strip()):  #split en phrases
            phrase = " ".join(current_words).strip()
            phrase = normaliser_phrases(phrase)
            phrase = normaliser_noms(phrase)

            repliques_finales[final_id] = {
                "line": phrase, #rename de content en line
                "speaker_id": current_speaker,
            }

            # attribue le repl_id (identifiant unique de la réplique) à chaque mot l'ayant composée
            for k in current_keys:
                dictionary[k]["repl_id"] = final_id

            current_words = []
            current_keys = []
            current_speaker = None
            final_id += 1

    return repliques_finales


def words_concatenation_newline(dictionary: dict):
    """Variante de words_concatenation() pour l'architecture otto : forme
    les répliques à partir du champ "newline" (retour à la ligne de
    sous-titrage) plutôt que de la ponctuation de fin de phrase. Un mot
    avec newline=True démarre une nouvelle réplique — la réplique en
    cours est donc clôturée juste avant lui."""
    repliques_finales = {}
    final_id = 1
    current_words = []
    current_keys = []
    current_speaker = None

    def cloturer_replique():
        nonlocal final_id, current_words, current_keys, current_speaker
        phrase = " ".join(current_words).strip()
        phrase = normaliser_phrases(phrase)
        phrase = normaliser_noms(phrase)
        repliques_finales[final_id] = {
            "line": phrase,
            "speaker_id": current_speaker,
        }
        for k in current_keys:
            dictionary[k]["repl_id"] = final_id
        current_words = []
        current_keys = []
        current_speaker = None
        final_id += 1

    for key in dictionary.keys():
        element = dictionary[key]
        content = element["content"]
        speaker_id = to_int_id(element["speaker_id"])

        if element.get("newline") and current_words:
            cloturer_replique()

        if current_speaker is None:
            current_speaker = speaker_id

        current_words.append(content)
        current_keys.append(key)

    if current_words:  # dernier segment, pas forcément suivi d'un newline
        cloturer_replique()

    return repliques_finales

if __name__ == "__main__":
    ##### CHARGEMENT DU FICHIER JSON
    transcription = "../data/raw/USGS_1804_daia.json"
    corpus_transcript = load_json(transcription)

    ##### CONVERSION "speakers" en int
    if "speakers" in corpus_transcript:
        for speaker in corpus_transcript["speakers"]:
            if "id" in speaker:
                speaker["id"] = to_int_id(speaker["id"])

    ##### CONVERSION des speaker_id dans "words" en int
    if "words" in corpus_transcript:
        for word in corpus_transcript["words"]:
            if "speaker_id" in word:
                word["speaker_id"] = to_int_id(word["speaker_id"])

    ##### CONVERSION EN DICT
    words_dict = {
        i: dict(element)
        for i, element in enumerate(corpus_transcript.get("words", []))
    }

    ##### FORMATION des répliques
    repliques = words_concatenation(words_dict)

    ##### RÉINJECTION des words enrichis (avec repl_id)
    corpus_transcript["words"] = list(words_dict.values())

    ##### IMPLEMENTATION du champ "repliques"
    corpus_transcript["repliques"] = repliques

    ##### Sauvegarde
    output_path = "../data/clean/transcription_USGS_clean.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus_transcript, f, indent=4, ensure_ascii=False)
    print(f"Fichier sauvegardé : {output_path}")
