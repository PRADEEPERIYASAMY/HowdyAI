import json
import os
from datetime import datetime

class MetricsLogger:
    def __init__(self, log_path="logs/metrics.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
    def log_request(self, query: str, cache_hit: bool, guardrail_hit: bool, latency: float, num_sources: int):
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "cache_hit": cache_hit,
            "guardrail_hit": guardrail_hit,
            "latency": latency,
            "num_sources": num_sources
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            
    def get_metrics(self):
        # TODO: Reading the entire JSONL into memory on every refresh does not scale for 
        # production traffic. In a real deployment, migrate to a timeseries database 
        # (like Prometheus) or use a log tailer to fetch only recent records.
        if not os.path.exists(self.log_path):
            return []
            
        with open(self.log_path, "r") as f:
            lines = f.readlines()
            
        return [json.loads(line) for line in lines]

    def clear(self):
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

