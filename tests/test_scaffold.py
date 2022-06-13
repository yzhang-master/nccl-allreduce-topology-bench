import json
import subprocess
import sys

from nccl_bench import __version__, scaffold_info


def test_scaffold_reports_real_gloo_availability():
    info = scaffold_info()
    assert info["version"] == __version__
    assert info["gloo_available"] is True
    proc = subprocess.run(
        [sys.executable, "-m", "nccl_bench.smoke"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(proc.stdout) == info
