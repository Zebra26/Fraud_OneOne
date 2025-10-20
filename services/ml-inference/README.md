# ML Inference Service

- Expose `/infer` (FastAPI) pour calculer la probabilité de fraude + explications SHAP.
- Repose sur l'ensemble multicouche (`ensemble.py`) aligné avec le backend.
- Temps de réponse cible: 50–100 ms (optimisé via cache Redis + modèles en mémoire).
- Traçabilité: renvoie version du modèle et contributions SHAP (AI Act).

