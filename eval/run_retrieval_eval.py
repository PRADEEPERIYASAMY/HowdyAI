import json
import logging
import os
import sys

# Ensure the root directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import AppConfig
from src.search.hybrid_retriever import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_retrieval():
    config = AppConfig()
    # Suppress verbose debug logs during eval
    logging.getLogger("src.search.hybrid_retriever").setLevel(logging.WARNING)
    logging.getLogger("src.search.cross_encoder_ranker").setLevel(logging.WARNING)
    
    logger.info("Initializing Hybrid Retriever...")
    retriever = HybridRetriever(config)
    
    qa_file = os.path.join(os.path.dirname(__file__), "retrieval_qa_pairs.json")
    with open(qa_file, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    
    hits = 0
    total = len(pairs)
    logger.info(f"Loaded {total} retrieval QA pairs. Starting evaluation...")
    
    for idx, pair in enumerate(pairs, 1):
        query = pair["query"]
        expected_url = pair["expected_url"]
        
        results = retriever.retrieve(query)
        found = False
        
        for rank, r in enumerate(results, 1):
            if expected_url in r["url"]:
                found = True
                logger.info(f"[{idx}/{total}] PASS | Rank: {rank} | Query: {query[:40]}...")
                break
        
        if found:
            hits += 1
        else:
            logger.error(f"[{idx}/{total}] FAIL | Query: {query[:40]}... | Expected URL snippet: {expected_url}")
    
    recall = hits / total if total > 0 else 0
    logger.info(f"\n{'='*40}")
    logger.info(f"Retrieval Eval Complete")
    logger.info(f"Recall@10: {recall:.2%} ({hits}/{total})")
    logger.info(f"{'='*40}")

if __name__ == "__main__":
    evaluate_retrieval()
