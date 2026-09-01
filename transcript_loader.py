"""
Pipeline de chargement et normalisation des fichiers de transcription
pour les deux architectures rencontrées : "daia" et "otto".

Format normalisé en sortie :
{
    "speakers": [{"id": str, "name": str | None}, ...],
    "words": [
        {
            "text": str,
            "speaker_id": str | None,
            "start_time": float,   # en secondes
            "end_time": float,     # en secondes
            "confidence": float | None,
            "newline": bool | None,
        },
        ...
    ]
}
"""

import json


def load_daia(filepath, fps=25):
    """
    Charge un fichier au format "daia".

    Architecture source :
    - clé racine "words", champs "content" / "speaker_id"
    - "start_time" / "end_time" exprimés en frames (int)
    - "speakers" = liste d'ids seuls (pas de nom)

    fps : fréquence utilisée pour convertir les frames en secondes.
    À vérifier/ajuster si le fichier source n'est pas en 25 fps.
    """
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    speakers = [{"id": s["id"], "name": None} for s in data.get("speakers", [])]

    words = []
    for w in data.get("words", []):
        words.append({
            "text": w["content"].strip(),
            "speaker_id": w.get("speaker_id"),
            "start_time": w["start_time"] / fps,
            "end_time": w["end_time"] / fps,
            "confidence": w.get("confidence"),
            "newline": None,
        })

    return {"speakers": speakers, "words": words}


def load_otto(filepath):
    """
    Charge un fichier au format "otto".

    Architecture source :
    - clé racine "transcription", champs "text" / "speakerId"
    - "startTime" / "endTime" déjà en secondes (float)
    - "speakers" = liste d'ids avec "name"
    """
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    speakers = [{"id": s["id"], "name": s.get("name")} for s in data.get("speakers", [])]

    words = []
    for w in data.get("transcription", []):
        words.append({
            "text": w["text"],
            "speaker_id": w.get("speakerId") or None,
            "start_time": w["startTime"],
            "end_time": w["endTime"],
            "confidence": None,
            "newline": w.get("newline", False),
        })

    return {"speakers": speakers, "words": words}


def load_transcript(filepath, architecture):
    """
    Point d'entrée unique de la pipeline.

    architecture : "daia" ou "otto" — détermine quelle fonction
    de parsing/normalisation est utilisée.
    """
    if architecture == "daia":
        return load_daia(filepath)
    elif architecture == "otto":
        return load_otto(filepath)
    else:
        raise ValueError(
            f"Architecture inconnue: {architecture!r}. Attendu 'daia' ou 'otto'."
        )


def export_daia(corpus_final):
    """
    Reconvertit le corpus final vers le format "daia".

    Sans effet : le format natif utilisé en interne par la pipeline
    (words/content/speaker_id) EST déjà le format daia. Fournie pour
    symétrie avec export_otto et pour que l'appelant n'ait pas à savoir
    quelle architecture nécessite réellement une conversion.
    """
    return corpus_final


def export_otto(corpus_final, chemin_otto_original):
    """
    Reconvertit le corpus final (format natif daia : "words"/"content"/
    "speaker_id") vers le format "otto" ("transcription"/"text"/
    "speakerId" + champs de mise en forme).

    Les champs de mise en forme (newline, x, y, width, height, color,
    italic, markWord) ne sont pas conservés par la pipeline (ils ne sont
    pas convertis à l'entrée) : ils sont donc repris ici du fichier otto
    original, réappariés par position avec les mots du corpus final. Le
    texte, les temps et le speaker_id (potentiellement corrigés par la
    pipeline) proviennent eux du corpus final.

    Suppose que le nombre et l'ordre des mots n'ont pas changé entre le
    fichier otto original et le corpus final (vrai pour cette pipeline :
    seules des "repliques" peuvent être ajoutées/retirées/modifiées,
    jamais des mots individuels) — sinon lève une ValueError explicite.
    """
    with open(chemin_otto_original, encoding="utf-8") as f:
        original = json.load(f)

    mots_originaux = original.get("transcription", [])
    mots_finaux = corpus_final.get("words", [])

    if len(mots_originaux) != len(mots_finaux):
        raise ValueError(
            f"Nombre de mots incohérent entre le fichier otto original "
            f"({len(mots_originaux)}) et le corpus final ({len(mots_finaux)}) "
            "— impossible de réassocier les champs de mise en forme."
        )

    transcription = []
    for mot_original, mot_final in zip(mots_originaux, mots_finaux):
        speaker_id = mot_final.get("speaker_id")
        transcription.append({
            "text": mot_final["content"],
            "startTime": mot_final["start_time"],
            "endTime": mot_final["end_time"],
            "newline": mot_original.get("newline", False),
            "x": mot_original.get("x", 0),
            "y": mot_original.get("y", 0),
            "width": mot_original.get("width", 0),
            "height": mot_original.get("height", 0),
            "color": mot_original.get("color", "#FFFFFF"),
            "italic": mot_original.get("italic", False),
            "markWord": mot_original.get("markWord", 0),
            "speakerId": "" if speaker_id in (None, "") else str(speaker_id),
        })

    resultat = dict(corpus_final)
    resultat["transcription"] = transcription
    resultat["speakers"] = [
        {"id": s["id"], "name": s.get("name", "")}
        for s in corpus_final.get("speakers", [])
    ]
    resultat.pop("words", None)

    return resultat


if __name__ == "__main__":
    daia_result = load_transcript("BR023K43_ep1954.json", "daia")
    otto_result = load_transcript("un_si_beau_soleil_1954.json", "otto")
    print(daia_result["words"][:3])
    print(otto_result["words"][:3])
