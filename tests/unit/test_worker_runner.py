import inspect

from humpback.workers import runner


def test_runner_no_longer_references_retired_worker_paths():
    source = inspect.getsource(runner)

    retired_symbols = (
        "processing_worker",
        "search_worker",
        "label_processing_worker",
        "claim_processing_job",
        "claim_search_job",
        "claim_label_processing_job",
        "run_processing_job",
        "run_search_job",
        "run_label_processing_job",
    )

    for symbol in retired_symbols:
        assert symbol not in source
