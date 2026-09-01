# Import de scénario

## DEPENDANCES
Toutes les dépendance utilisées sont dans `requirements.txt`

Pour ce workflow, j'utilise `python 3.10`

Il faudra une clé API Mistral.


## LANCEMENT
Pour lancer le workflow, voilà des exemples de commande :

```
python3 main.py fichier_gold.json fichier_transcript.json --port XXXX --asid XXXX --architecture {daia, otto} [--api-key XXXX] [--tirets] [-o output_name.json]
```

*Notes : suivre l'ordre des arguments obligatoires : PDF puis Trasncript*


**Le script `main.py` lancera tout le workflow.**



### ARGUMENTS

L'option `-h` affiche le message help

L'option `-o` ou `--sortie` permet de spécifier le nom de l'output, autre que celui défini par défault (transcription_finale.json).

L'option `--tirets` permet d'ajouter des tirets au début d'une phrase à chaque changement de speaker.

L'option `--api-key` permet d'insérer une clé API Mistral directement dans la commande.

L'option `--port` et `--asid` permet de spécifier les identifiants.

L'option `--architecture` a uniquement 2 arguments possibles `otto` ou `daia`, cette option permet de spécifier l'architecture d'un fichier json



## INPUTS/OUTPUT

Ce workflow nécessite deux **inputs** différents :

- un scénario validé d'*Un Si Grand Soleil* en pdf.

- un fichier de transcription brut en json de ce scénario. Le fichier de tanscription peut avoir 2 architectures possibles :

1) Architecture **DAIA** :

        {
            "speakers": [
                { "id": "S1" },
                { "id": "S2" }
            ],
            "words": [
                {
                "content": "Bonjour ",
                "speaker_id": "S1",
                "start_time": 5,
                "end_time": 25,
                "confidence": 1
                },
                {
                "content": "à ",
                "speaker_id": "S1",
                "start_time": 25,
                "end_time": 30,
                "confidence": 1
                }
            ]
        }

2) Architecture **OTTO** :

        {
            "transcription": [
                {
                "text": "Bonjour",
                "startTime": 0.2,
                "endTime": 1.0,
                "newline": true,
                "x": 0, "y": 0, "width": 0, "height": 0,
                "color": "#FFFFFF",
                "italic": false,
                "markWord": 0,
                "speakerId": ""
                },
                {
                "text": "à",
                "startTime": 1.0,
                "endTime": 1.2,
                "newline": false,
                "x": 0, "y": 0, "width": 0, "height": 0,
                "color": "#FFFFFF",
                "italic": false,
                "markWord": 0,
                "speakerId": ""
                }
            ],
            "speakers": [
                { "id": "S1", "name": "MARIE" },
                { "id": "S2", "name": "PAUL" }
            ]
        }


L'**output** du workflow sera toujours en json en conservant l'architecture spécifiée en entré.

---

## CE QUI NE GERE PAS

Toutes les autres architectures de fichiers json ne pourront pas être utilisées.

Les autres formats de fichiers autre que le json ne fonctionneront pas.




