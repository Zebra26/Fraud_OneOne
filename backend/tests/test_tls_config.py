from backend.database.mongo import MongoDBClient


def test_mongo_client_init_with_tls(tmp_path):
    ca = tmp_path / "ca.crt"
    ca.write_text("TEST_CA")
    client = MongoDBClient("mongodb://localhost:27017", "test", tls_enabled=True, tls_verify=True, tls_ca_path=str(ca))
    # Basic sanity: client object is created
    assert client is not None

