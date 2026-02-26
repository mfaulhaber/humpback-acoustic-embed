async def test_create_clustering_job(client):
    resp = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": ["fake-id-1"]},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "queued"
    assert data["embedding_set_ids"] == ["fake-id-1"]


async def test_get_clustering_job(client):
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": ["fake-id-1"]},
    )
    job_id = create.json()["id"]
    resp = await client.get(f"/clustering/jobs/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == job_id


async def test_get_clustering_job_not_found(client):
    resp = await client.get("/clustering/jobs/nonexistent")
    assert resp.status_code == 404


async def test_list_clusters_empty(client):
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": ["fake-id-1"]},
    )
    job_id = create.json()["id"]
    resp = await client.get(f"/clustering/jobs/{job_id}/clusters")
    assert resp.status_code == 200
    assert resp.json() == []
