import json
from dataclasses import dataclass

import streamlit as st

from corpus import Corpus


@dataclass
class TiretOption:
    @classmethod
    def run_option_tirets(cls, file_transcript, corpus_in=None, return_corpus=False):
        corpus = json.loads(json.dumps(corpus_in)) if corpus_in is not None else Corpus.load_json_file(file_transcript)
        if not corpus:
            return corpus_in if return_corpus else None

        repliques = corpus.get("repliques", [])
        if not repliques:
            st.warning("Aucune réplique trouvée.")
            return corpus_in if return_corpus else None

        items = repliques if isinstance(repliques, list) else list(repliques.values())

        previous_speaker = None
        count = 0
        for r in items:
            speaker = r.get("speaker", r.get("speaker_id", ""))
            line    = r.get("line", "")
            if speaker != previous_speaker:
                if not line.startswith("–") and not line.startswith("-"):
                    r["line"] = "– " + line
                    count += 1
            previous_speaker = speaker

        if isinstance(repliques, list):
            corpus["repliques"] = items
        else:
            keys = list(corpus["repliques"].keys())
            corpus["repliques"] = {keys[i]: items[i] for i in range(len(keys))}

        with st.expander("Notes – Tirets", expanded=True):
            st.markdown(f"✓ Tirets ajoutés !")
            if not return_corpus:
                Corpus.to_download_json(corpus, "transcription_tirets.json")

        if return_corpus:
            return corpus
