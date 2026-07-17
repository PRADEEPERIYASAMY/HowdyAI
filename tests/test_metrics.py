import os
import pytest
from src.metrics import MetricsLogger

def test_metrics_logger(tmp_path):
    log_file = tmp_path / "metrics.jsonl"
    logger = MetricsLogger(log_path=str(log_file))
    
    # Initially empty
    assert logger.get_metrics() == []
    
    # Log a request
    logger.log_request("test query", True, False, 1.5, 3)
    
    # Verify logged
    metrics = logger.get_metrics()
    assert len(metrics) == 1
    assert metrics[0]["query"] == "test query"
    assert metrics[0]["cache_hit"] is True
    assert metrics[0]["latency"] == 1.5
    
    # Clear
    logger.clear()
    assert logger.get_metrics() == []
    assert not os.path.exists(str(log_file))
