# nccl-allreduce-topology-bench

Work in progress. The initial scaffold checks that the CPU Gloo backend is available.

```bash
pip install -e ".[dev]"
ruff check .
pytest -q
python -m nccl_bench.smoke
```
