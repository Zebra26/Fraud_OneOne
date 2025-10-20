db = db.getSiblingDB(process.env.MONGO_DB || "fraud_db");

db.createCollection("transactions");
db.createCollection("explanations");

db.transactions.createIndex({ transaction_id: 1 }, { unique: true });
db.explanations.createIndex({ transaction_id: 1, recorded_at: 1 });

print("Mongo init done with AI Act compliant SHAP traceability.");

