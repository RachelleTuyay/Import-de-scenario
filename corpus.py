import json
from dataclasses import dataclass

import streamlit as st


@dataclass
class Corpus:

    @classmethod
    def format_duration(cls, elapsed: float) -> str:
        if elapsed < 60:
            return f"{elapsed:.2f}s"
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        return f"{minutes}m {seconds:.2f}s"

    @classmethod
    def load_json_file(cls, uploaded_file) -> dict:
        """Charge un corpus depuis un st.uploaded_file."""
        try:
            uploaded_file.seek(0)
            return json.load(uploaded_file)
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier JSON : {e}")
            return {}

    @classmethod
    def to_download_json(cls, data: dict, filename: str):
        """Bouton de téléchargement d'un dict → JSON."""
        output = json.dumps(data, ensure_ascii=False, indent=2)
        st.download_button(
            label=f"📥 Télécharger {filename}",
            data=output,
            file_name=filename,
            mime="application/json",
        )
