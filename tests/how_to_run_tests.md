python -m pytest tests/test_ablate_batch.py::TestAblateBatchCore::test_ablation_lowers_log_prob -v
python -m pytest tests/test_ablate_batch.py::TestAblateBatchCore -v
python -m pytest tests/test_ablate_batch.py -v

python -m pytest tests/test_ablate_batch.py::TestAblateBatchCore::test_ablation_lowers_log_prob tests/test_ablate_batch.py::TestAblateBatchCore::test_random_baseline_less_effective -v
