import os
from dataclasses import dataclass

import streamlit as st
from sentence_transformers import SentenceTransformer


@dataclass
class ModelLoader:
    @classmethod
    @st.cache_resource(show_spinner="Chargement du modèle SentenceTransformer...")
    def load_model(cls, model_dir: str) -> SentenceTransformer:
        if os.path.exists(model_dir) and os.listdir(model_dir):
            return SentenceTransformer(model_dir, trust_remote_code=True)
        st.warning("Modèle introuvable, téléchargement depuis HuggingFace....")
        model = SentenceTransformer(
            "sentence-transformers/static-similarity-mrl-multilingual-v1",
            trust_remote_code=True,
        )
        model.save(model_dir)
        return model

    @classmethod
    @st.cache_resource(show_spinner="Chargement du pipeline Stanza...")
    def load_stanza_pipeline(cls):
        """
        OPTIMISATION : pipeline Stanza séparé dans son propre cache_resource.
        Avant, stanza.Pipeline(...) était instancié à l'intérieur de _annotate_gold_ner,
        qui est en cache_data : si l'utilisateur change de fichier gold, le cache
        (basé sur le contenu JSON) rate ET le pipeline (chargement lourd, plusieurs
        secondes) est reconstruit inutilement. Ici, le pipeline est chargé une seule
        fois pour toute la session, quel que soit le gold utilisé.
        """
        import stanza
        return stanza.Pipeline(
            lang='fr',
            processors='tokenize,mwt,pos,ner',
            verbose=False,
        )
