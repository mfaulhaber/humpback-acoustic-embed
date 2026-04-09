"""Integration tests for hyperparameter tuning API endpoints."""

from __future__ import annotations


import pytest
from httpx import AsyncClient


BASE = "/classifier/hyperparameter"


# ---------------------------------------------------------------------------
# Manifest CRUD
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_manifest(client: AsyncClient) -> None:
    resp = await client.post(
        f"{BASE}/manifests",
        json={
            "name": "test-manifest",
            "training_job_ids": ["tj-1"],
            "detection_job_ids": [],
            "split_ratio": [70, 15, 15],
            "seed": 42,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "test-manifest"
    assert data["status"] == "queued"
    assert data["training_job_ids"] == ["tj-1"]
    assert data["split_ratio"] == [70, 15, 15]


@pytest.mark.asyncio
async def test_list_manifests(client: AsyncClient) -> None:
    # Create two manifests
    await client.post(
        f"{BASE}/manifests",
        json={"name": "m1", "training_job_ids": ["t1"]},
    )
    await client.post(
        f"{BASE}/manifests",
        json={"name": "m2", "detection_job_ids": ["d1"]},
    )
    resp = await client.get(f"{BASE}/manifests")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    # Ordered by created_at desc — m2 first
    assert data[0]["name"] == "m2"
    assert data[1]["name"] == "m1"


@pytest.mark.asyncio
async def test_get_manifest_detail(client: AsyncClient) -> None:
    create_resp = await client.post(
        f"{BASE}/manifests",
        json={"name": "detail-test", "training_job_ids": ["t1"]},
    )
    manifest_id = create_resp.json()["id"]
    resp = await client.get(f"{BASE}/manifests/{manifest_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == manifest_id
    # Detail fields present
    assert "manifest_path" in data
    assert "split_summary" in data
    assert "detection_job_summaries" in data


@pytest.mark.asyncio
async def test_get_manifest_not_found(client: AsyncClient) -> None:
    resp = await client.get(f"{BASE}/manifests/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_manifest(client: AsyncClient) -> None:
    create_resp = await client.post(
        f"{BASE}/manifests",
        json={"name": "to-delete", "training_job_ids": ["t1"]},
    )
    manifest_id = create_resp.json()["id"]
    resp = await client.delete(f"{BASE}/manifests/{manifest_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"
    # Confirm gone
    resp2 = await client.get(f"{BASE}/manifests/{manifest_id}")
    assert resp2.status_code == 404


@pytest.mark.asyncio
async def test_delete_manifest_blocked_by_search(client: AsyncClient) -> None:
    create_resp = await client.post(
        f"{BASE}/manifests",
        json={"name": "referenced", "training_job_ids": ["t1"]},
    )
    manifest_id = create_resp.json()["id"]
    # Create a search referencing the manifest
    await client.post(
        f"{BASE}/searches",
        json={"name": "s1", "manifest_id": manifest_id, "n_trials": 5},
    )
    resp = await client.delete(f"{BASE}/manifests/{manifest_id}")
    assert resp.status_code == 409
    assert "referenced" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Search CRUD
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_search(client: AsyncClient) -> None:
    m_resp = await client.post(
        f"{BASE}/manifests",
        json={"name": "m-for-search", "training_job_ids": ["t1"]},
    )
    manifest_id = m_resp.json()["id"]
    resp = await client.post(
        f"{BASE}/searches",
        json={
            "name": "search-1",
            "manifest_id": manifest_id,
            "n_trials": 10,
            "seed": 7,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "search-1"
    assert data["status"] == "queued"
    assert data["manifest_id"] == manifest_id
    assert data["manifest_name"] == "m-for-search"
    assert data["n_trials"] == 10
    assert data["trials_completed"] == 0


@pytest.mark.asyncio
async def test_create_search_invalid_manifest(client: AsyncClient) -> None:
    resp = await client.post(
        f"{BASE}/searches",
        json={"name": "bad", "manifest_id": "nonexistent"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_create_search_uses_default_search_space(
    client: AsyncClient,
) -> None:
    m_resp = await client.post(
        f"{BASE}/manifests",
        json={"name": "m-defaults", "training_job_ids": ["t1"]},
    )
    manifest_id = m_resp.json()["id"]
    s_resp = await client.post(
        f"{BASE}/searches",
        json={"name": "s-defaults", "manifest_id": manifest_id},
    )
    search_id = s_resp.json()["id"]
    detail = await client.get(f"{BASE}/searches/{search_id}")
    assert detail.status_code == 200
    # Should have default search space populated
    ss = detail.json()["search_space"]
    assert "classifier" in ss
    assert "threshold" in ss


@pytest.mark.asyncio
async def test_list_searches(client: AsyncClient) -> None:
    m_resp = await client.post(
        f"{BASE}/manifests",
        json={"name": "m-list", "training_job_ids": ["t1"]},
    )
    manifest_id = m_resp.json()["id"]
    await client.post(
        f"{BASE}/searches",
        json={"name": "s1", "manifest_id": manifest_id, "n_trials": 5},
    )
    resp = await client.get(f"{BASE}/searches")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    assert data[0]["manifest_name"] == "m-list"


@pytest.mark.asyncio
async def test_get_search_detail(client: AsyncClient) -> None:
    m_resp = await client.post(
        f"{BASE}/manifests",
        json={"name": "m-detail", "training_job_ids": ["t1"]},
    )
    manifest_id = m_resp.json()["id"]
    s_resp = await client.post(
        f"{BASE}/searches",
        json={"name": "s-detail", "manifest_id": manifest_id, "n_trials": 5},
    )
    search_id = s_resp.json()["id"]
    resp = await client.get(f"{BASE}/searches/{search_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == search_id
    assert "search_space" in data
    assert "best_config" in data


@pytest.mark.asyncio
async def test_get_search_not_found(client: AsyncClient) -> None:
    resp = await client.get(f"{BASE}/searches/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_search(client: AsyncClient) -> None:
    m_resp = await client.post(
        f"{BASE}/manifests",
        json={"name": "m-del-search", "training_job_ids": ["t1"]},
    )
    manifest_id = m_resp.json()["id"]
    s_resp = await client.post(
        f"{BASE}/searches",
        json={"name": "s-del", "manifest_id": manifest_id, "n_trials": 5},
    )
    search_id = s_resp.json()["id"]
    resp = await client.delete(f"{BASE}/searches/{search_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"
    resp2 = await client.get(f"{BASE}/searches/{search_id}")
    assert resp2.status_code == 404


@pytest.mark.asyncio
async def test_search_history_not_available(client: AsyncClient) -> None:
    m_resp = await client.post(
        f"{BASE}/manifests",
        json={"name": "m-hist", "training_job_ids": ["t1"]},
    )
    manifest_id = m_resp.json()["id"]
    s_resp = await client.post(
        f"{BASE}/searches",
        json={"name": "s-hist", "manifest_id": manifest_id, "n_trials": 5},
    )
    search_id = s_resp.json()["id"]
    # History not available yet (no results_dir)
    resp = await client.get(f"{BASE}/searches/{search_id}/history")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Search space defaults
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_space_defaults(client: AsyncClient) -> None:
    resp = await client.get(f"{BASE}/search-space-defaults")
    assert resp.status_code == 200
    data = resp.json()
    assert "search_space" in data
    ss = data["search_space"]
    assert "classifier" in ss
    assert "threshold" in ss
    assert "context_pooling" in ss
    # No hard_negative_fraction
    assert "hard_negative_fraction" not in ss


# ---------------------------------------------------------------------------
# Import candidate from search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_import_candidate_requires_complete_search(
    client: AsyncClient,
) -> None:
    m_resp = await client.post(
        f"{BASE}/manifests",
        json={"name": "m-import", "training_job_ids": ["t1"]},
    )
    manifest_id = m_resp.json()["id"]
    s_resp = await client.post(
        f"{BASE}/searches",
        json={"name": "s-import", "manifest_id": manifest_id, "n_trials": 5},
    )
    search_id = s_resp.json()["id"]
    # Search is queued, not complete
    resp = await client.post(f"{BASE}/searches/{search_id}/import-candidate")
    assert resp.status_code == 400
    assert "complete" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Candidate endpoints (relocated)
# ---------------------------------------------------------------------------

OLD_CANDIDATES = "/classifier/autoresearch-candidates"
NEW_CANDIDATES = f"{BASE}/candidates"


@pytest.mark.asyncio
async def test_new_candidate_list_endpoint(client: AsyncClient) -> None:
    resp = await client.get(f"{NEW_CANDIDATES}")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_old_candidate_list_still_works(client: AsyncClient) -> None:
    resp = await client.get(OLD_CANDIDATES)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_new_candidate_not_found(client: AsyncClient) -> None:
    resp = await client.get(f"{NEW_CANDIDATES}/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_old_candidate_not_found(client: AsyncClient) -> None:
    resp = await client.get(f"{OLD_CANDIDATES}/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_both_paths_return_same_list(client: AsyncClient) -> None:
    old_resp = await client.get(OLD_CANDIDATES)
    new_resp = await client.get(NEW_CANDIDATES)
    assert old_resp.status_code == 200
    assert new_resp.status_code == 200
    assert old_resp.json() == new_resp.json()


# ---------------------------------------------------------------------------
# Manifest summary counts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manifest_summary_has_null_counts_when_queued(
    client: AsyncClient,
) -> None:
    resp = await client.post(
        f"{BASE}/manifests",
        json={"name": "counts-test", "training_job_ids": ["t1"]},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["positive_count"] is None
    assert data["negative_count"] is None


@pytest.mark.asyncio
async def test_manifest_summary_has_counts_when_complete(
    app_settings,
) -> None:
    import json

    from httpx import ASGITransport

    from humpback.api.app import create_app
    from humpback.models.hyperparameter import HyperparameterManifest

    app = create_app(app_settings)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        await app.router.startup()

        # Create a manifest
        resp = await ac.post(
            f"{BASE}/manifests",
            json={"name": "counts-complete", "training_job_ids": ["t1"]},
        )
        manifest_id = resp.json()["id"]

        # Simulate worker completion by updating DB directly
        async with app.state.session_factory() as session:
            m = await session.get(HyperparameterManifest, manifest_id)
            assert m is not None
            m.status = "complete"
            m.example_count = 100
            m.split_summary = json.dumps(
                {
                    "train": {"total": 70, "positive": 20, "negative": 50},
                    "val": {"total": 15, "positive": 5, "negative": 10},
                    "test": {"total": 15, "positive": 3, "negative": 12},
                }
            )
            await session.commit()

        # Verify list endpoint returns counts
        resp = await ac.get(f"{BASE}/manifests")
        data = resp.json()
        manifest = next(m for m in data if m["id"] == manifest_id)
        assert manifest["positive_count"] == 28
        assert manifest["negative_count"] == 72

        # Verify detail endpoint also returns counts
        resp = await ac.get(f"{BASE}/manifests/{manifest_id}")
        detail = resp.json()
        assert detail["positive_count"] == 28
        assert detail["negative_count"] == 72

        await app.router.shutdown()


# ---------------------------------------------------------------------------
# Candidate delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_candidate(app_settings) -> None:
    import json

    from httpx import ASGITransport

    from humpback.api.app import create_app
    from humpback.models.classifier import AutoresearchCandidate

    app = create_app(app_settings)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        await app.router.startup()

        # Seed a candidate directly in the DB
        async with app.state.session_factory() as session:
            candidate = AutoresearchCandidate(
                id="test-candidate-del",
                name="test-candidate",
                status="imported",
                manifest_path="/tmp/fake/manifest.json",
                best_run_path="/tmp/fake/best_run.json",
                promoted_config=json.dumps({"classifier": "logreg"}),
            )
            session.add(candidate)
            await session.commit()

        # Verify it exists
        resp = await ac.get(f"{NEW_CANDIDATES}/test-candidate-del")
        assert resp.status_code == 200

        # Delete it
        resp = await ac.delete(f"{NEW_CANDIDATES}/test-candidate-del")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # Verify it's gone
        resp = await ac.get(f"{NEW_CANDIDATES}/test-candidate-del")
        assert resp.status_code == 404

        await app.router.shutdown()


@pytest.mark.asyncio
async def test_delete_candidate_not_found(client: AsyncClient) -> None:
    resp = await client.delete(f"{NEW_CANDIDATES}/nonexistent")
    assert resp.status_code == 404
