"""Integration tests for classifier API endpoints."""


async def test_create_training_job_missing_embedding_sets(client):
    """400 when embedding sets don't exist."""
    resp = await client.post(
        "/classifier/training-jobs",
        json={
            "name": "test",
            "positive_embedding_set_ids": ["nonexistent"],
            "negative_embedding_set_ids": ["nonexistent2"],
        },
    )
    assert resp.status_code == 400


async def test_create_training_job_missing_negative_sets(client):
    """400 when negative embedding sets don't exist."""
    resp = await client.post(
        "/classifier/training-jobs",
        json={
            "name": "test",
            "positive_embedding_set_ids": ["nonexistent"],
            "negative_embedding_set_ids": ["also-nonexistent"],
        },
    )
    assert resp.status_code == 400


async def test_list_training_jobs_empty(client):
    resp = await client.get("/classifier/training-jobs")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_get_training_job_not_found(client):
    resp = await client.get("/classifier/training-jobs/nonexistent")
    assert resp.status_code == 404


async def test_list_models_empty(client):
    resp = await client.get("/classifier/models")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_get_model_not_found(client):
    resp = await client.get("/classifier/models/nonexistent")
    assert resp.status_code == 404


async def test_delete_model_not_found(client):
    resp = await client.delete("/classifier/models/nonexistent")
    assert resp.status_code == 404


async def test_list_detection_jobs_empty(client):
    resp = await client.get("/classifier/detection-jobs")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_create_detection_job_bad_model(client):
    resp = await client.post(
        "/classifier/detection-jobs",
        json={
            "classifier_model_id": "nonexistent",
            "audio_folder": "/tmp",
        },
    )
    assert resp.status_code == 400


async def test_get_detection_job_not_found(client):
    resp = await client.get("/classifier/detection-jobs/nonexistent")
    assert resp.status_code == 404


async def test_download_detection_not_found(client):
    resp = await client.get("/classifier/detection-jobs/nonexistent/download")
    assert resp.status_code == 404


async def test_extraction_settings(client):
    resp = await client.get("/classifier/extraction-settings")
    assert resp.status_code == 200
    data = resp.json()
    assert "positive_output_path" in data
    assert "negative_output_path" in data


async def test_extract_nonexistent_jobs(client):
    resp = await client.post(
        "/classifier/detection-jobs/extract",
        json={"job_ids": ["nonexistent"]},
    )
    assert resp.status_code == 404


async def test_extract_empty_job_ids(client):
    resp = await client.post(
        "/classifier/detection-jobs/extract",
        json={"job_ids": []},
    )
    # Empty list: all 0 jobs found, 0 expected → should succeed with count 0
    assert resp.status_code == 200
    assert resp.json()["count"] == 0
