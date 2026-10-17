"""The HTTP surface goes through the same engine: auth, consent and receipts hold over HTTP."""

from fastapi.testclient import TestClient

from pcim.api import create_app
from pcim.engine import PersonalCI


def test_api_end_to_end(tmp_path):
    client = TestClient(create_app(PersonalCI(tmp_path, seed=0)))
    tok = client.post("/v1/onboard", json={"user_id": "nishanth", "providers": ["mock"]}).json()["api_token"]
    assert client.post("/v1/onboard", json={"user_id": "intruder"}).status_code == 403  # closed after first user
    assert client.get("/v1/memory").status_code == 401
    h = {"Authorization": f"Bearer {tok}"}
    assert client.post("/v1/memory", headers=h, json={"content": "No meetings before 10am", "kind": "constraint"}).status_code == 200
    assert client.post("/v1/ingest", headers=h, json={"text": "x", "source": "ingest:notes"}).status_code == 403
    r = client.post("/v1/ask", headers=h, json={"task": "schedule my meetings this week"}).json()
    assert r["receipt"]["sent"] and "No meetings before 10am" in r["text"]
    assert client.post(f"/v1/feedback/{r['interaction_id']}", headers=h, json={"rating": 4}).json()["reward"] == 0.75
    rc = client.get(f"/v1/interactions/{r['interaction_id']}", headers=h).json()
    assert rc["sent"][0]["content"] == "No meetings before 10am"
    assert client.get("/v1/audit", headers=h).json()["chain_ok"]
    assert client.get("/", headers=h).status_code == 200
    assert client.delete("/v1/me", headers=h).json()["items"] == 1
    assert client.get("/v1/memory", headers=h).status_code == 401  # token died with the user
