import pytest


async def test_list_models_empty(client):
    """On startup, seed model should exist."""
    resp = await client.get("/admin/models")
    assert resp.status_code == 200
    models = resp.json()
    # Seed model is created on startup
    assert len(models) >= 1
    assert any(m["name"] == "multispecies_whale_fp16" for m in models)


async def test_create_model(client):
    resp = await client.post("/admin/models", json={
        "name": "test_new_model",
        "display_name": "Test New Model",
        "path": "models/test.tflite",
        "vector_dim": 512,
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "test_new_model"
    assert data["vector_dim"] == 512
    assert data["is_default"] is False


async def test_create_duplicate_name(client):
    await client.post("/admin/models", json={
        "name": "dup_model",
        "display_name": "Dup Model",
        "path": "models/dup.tflite",
    })
    resp = await client.post("/admin/models", json={
        "name": "dup_model",
        "display_name": "Dup Model 2",
        "path": "models/dup2.tflite",
    })
    assert resp.status_code == 400


async def test_update_model(client):
    create = await client.post("/admin/models", json={
        "name": "upd_model",
        "display_name": "Update Me",
        "path": "models/upd.tflite",
    })
    model_id = create.json()["id"]

    resp = await client.put(f"/admin/models/{model_id}", json={
        "display_name": "Updated Name",
        "vector_dim": 256,
    })
    assert resp.status_code == 200
    assert resp.json()["display_name"] == "Updated Name"
    assert resp.json()["vector_dim"] == 256


async def test_update_model_not_found(client):
    resp = await client.put("/admin/models/nonexistent", json={
        "display_name": "X",
    })
    assert resp.status_code == 404


async def test_set_default(client):
    create = await client.post("/admin/models", json={
        "name": "new_default",
        "display_name": "New Default",
        "path": "models/nd.tflite",
    })
    model_id = create.json()["id"]

    resp = await client.post(f"/admin/models/{model_id}/set-default")
    assert resp.status_code == 200
    assert resp.json()["is_default"] is True

    # Check the old default is no longer default
    all_models = (await client.get("/admin/models")).json()
    defaults = [m for m in all_models if m["is_default"]]
    assert len(defaults) == 1
    assert defaults[0]["id"] == model_id


async def test_delete_model(client):
    create = await client.post("/admin/models", json={
        "name": "del_model",
        "display_name": "Delete Me",
        "path": "models/del.tflite",
    })
    model_id = create.json()["id"]

    resp = await client.delete(f"/admin/models/{model_id}")
    assert resp.status_code == 200

    # Verify it's gone
    all_models = (await client.get("/admin/models")).json()
    assert all(m["id"] != model_id for m in all_models)


async def test_scan_models(client, app_settings):
    """Scan endpoint should return list (may be empty if no models dir)."""
    resp = await client.get("/admin/models/scan")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


async def test_create_tf2_saved_model(client):
    """Creating a model with model_type=tf2_saved_model should persist correctly."""
    resp = await client.post("/admin/models", json={
        "name": "surfperch_tf2",
        "display_name": "SurfPerch TF2",
        "path": "models/surfperch-tensorflow2",
        "vector_dim": 1280,
        "model_type": "tf2_saved_model",
        "input_format": "waveform",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["model_type"] == "tf2_saved_model"
    assert data["input_format"] == "waveform"
    assert data["vector_dim"] == 1280

    # Verify it shows up in the list
    list_resp = await client.get("/admin/models")
    models = list_resp.json()
    tf2_models = [m for m in models if m["name"] == "surfperch_tf2"]
    assert len(tf2_models) == 1
    assert tf2_models[0]["model_type"] == "tf2_saved_model"


async def test_default_model_type_in_response(client):
    """Models created without explicit model_type should return tflite defaults."""
    resp = await client.post("/admin/models", json={
        "name": "plain_tflite",
        "display_name": "Plain TFLite",
        "path": "models/plain.tflite",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["model_type"] == "tflite"
    assert data["input_format"] == "spectrogram"


async def test_tables_includes_model_configs(client):
    resp = await client.get("/admin/tables")
    assert resp.status_code == 200
    tables = resp.json()
    table_names = [t["table"] for t in tables]
    assert "model_configs" in table_names
