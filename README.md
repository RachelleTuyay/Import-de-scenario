# Import de scénario
Ce projet consiste à développer un plugin d'import de scénario pour un logiciel de sous-titrage/audiodescription OTTO. Son objectif est d'assister les sous-titreurs en corrigeant les transcriptions automatiquement à partir de scénarios validés, notamment en améliorant l'identification des locuteurs et l'association des scènes aux répliques.

Ce plugin **corrige environ 60% des locuteurs** et **indexe 80% les scènes aux répliques** afin d'accélérer la production de relevés de dialogues, de sous-titres SME (sourds et malentendants), ainsi que de faciliter l'audiodescription et l'adaptation multilingue.

Il s'appuie uniquement sur des scénarios validé de *Un si grand soleil* (USGS) et leurs transcriptions automatiques afin d'étudier dans quelle mesure une ressource textuelle fiable peut améliorer la qualité de correction des transcriptions générées automatiquement.


## DEPENDANCES
Toutes les dépendance utilisées sont dans `requirements.txt`

Pour ce workflow, j'utilise `python 3.10`

Il faudra une clé API Mistral.


## LANCEMENT

Il est possible de lancer le projet de deux manières.

### Avec `main.py` : 
**Le script `main.py` lancera tout le workflow dans le terminal.**

Pour lancer le workflow, voilà des exemples de commande :

```
python3 main.py fichier_gold.json fichier_transcript.json --port XXXX --asid XXXX --architecture {daia, otto} [--api-key XXXX] [--tirets] [-o output_name.json]
```

*Notes : suivre l'ordre des arguments obligatoires : PDF puis Trasncript*


#### ARGUMENTS pour `main.py` :

L'option `-h` affiche le message help

L'option `-o` ou `--sortie` permet de spécifier le nom de l'output, autre que celui défini par défault (transcription_finale.json).

L'option `--tirets` permet d'ajouter des tirets au début d'une phrase à chaque changement de speaker.

L'option `--api-key` permet d'insérer une clé API Mistral directement dans la commande.

L'option `--port` et `--asid` permet de spécifier les identifiants.

L'option `--architecture` a uniquement 2 arguments possibles `otto` ou `daia`, cette option permet de spécifier l'architecture d'un fichier json

#### INPUTS/OUTPUT pour `main.py`

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

### Avec `app.py` : 
**Le script `app.py` lancera le workflow sur une interface web utilisateur.**

Pour lancer le workflow avec `app.py`, voilà un exemple de commande :
```
streamlit run app.py
```
#### INPUTS/OUTPUT pour `app.py`
Il est possible d'entré 2 fichiers : un PDF et une transcription brute suivant les architectures précédemment décrite. Ainsi que d'utiliser déjà des fichiers prétraités au préalaable.

---

## CE QUI NE GERE PAS

Toutes les autres architectures de fichiers json ne pourront pas être utilisées.

Les autres formats de fichiers autre que le json ne fonctionneront pas.




