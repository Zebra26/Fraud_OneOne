// Script d'initialisation Redis (exécuté via `redis-cli --eval`).
// Prépare les clés nécessaires pour la traçabilité SHAP.

redis.call('SET', 'xai:note', 'Explicabilité (XAI) : Approche SHAP pour justifier les décisions et garantir la conformité réglementaire (ex: AI Act).')

