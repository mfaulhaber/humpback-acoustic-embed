"""Route-level checks for retired legacy workflow APIs."""


async def test_removed_legacy_route_prefixes_return_404(client):
    for path in (
        "/audio/",
        "/processing/jobs",
        "/processing/embedding-sets",
        "/search/similar",
        "/label-processing/jobs",
        "/clustering/jobs",
    ):
        resp = await client.get(path)
        assert resp.status_code == 404, path


async def test_retained_classifier_audio_media_route_is_registered(client):
    resp = await client.get(
        "/classifier/audio/not-a-real-audio-id/window",
        params={"start_seconds": 0, "duration_seconds": 1},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Audio file not found"


async def test_vocalization_clustering_route_surface_remains(client):
    resp = await client.get("/vocalization/clustering-jobs")
    assert resp.status_code == 200
    assert resp.json() == []
