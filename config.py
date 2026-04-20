import logging
import logging.config
import os
from dotenv import load_dotenv


class AppConfig:
    def __init__(self, log_level="INFO"):
        load_dotenv()

        self.BASE_PATH = os.path.dirname(__file__)
        self.DATA_PATH = os.path.join(self.BASE_PATH, "data")
        self.CRAWLED_DATA_PATH = os.path.join(self.DATA_PATH, "crawled")
        self.DOCUMENTS_PATH = os.path.join(self.DATA_PATH, "documents")
        self.DATABASE_PATH = os.path.join(self.DATA_PATH, "database")

        # ── Template paths (original) ─────────────────────────────────────
        self.TEMPLATE_PATH = os.path.join(
            self.BASE_PATH, "src", "templates", "chat_template.txt")
        self.SUMMARY_TEMPLATE_PATH = os.path.join(
            self.BASE_PATH, "src", "templates", "summary_template.txt")
        self.RANK_TEMPLATE_PATH = os.path.join(
            self.BASE_PATH, "src", "templates", "rank_template.txt")

        # ── New template paths (HowdyAI enhancements) ────────────────────
        self.REWRITE_TEMPLATE_PATH = os.path.join(
            self.BASE_PATH, "src", "templates", "rewrite_template.txt")
        self.GUARDRAIL_TEMPLATE_PATH = os.path.join(
            self.BASE_PATH, "src", "templates", "guardrail_template.txt")

        # ── API keys (original) ───────────────────────────────────────────
        self.OPENAPI_API_KEY = os.environ["OPENAI_API_KEY"]

        # ── HowdyAI enhancement config ────────────────────────────────────
        self.CACHE_DB_PATH = os.path.join(self.BASE_PATH, "howdyai_cache.db")
        self.CHROMA_PATH = self.DATABASE_PATH
        self.NUM_SEARCH_RESULTS = 5
        self.CHROMA_NUM_RESULTS = 5
        self.FUSION_TOP_N = 8
        self.RANKER_TOP_K = 5
        self.MEMORY_MAX_TURNS = 6
        self.FAST_MODEL = "gpt-4o-mini"
        self.STRONG_MODEL = "gpt-4o"

        self.logging_config = self.get_logger(log_level)

    def get_logger(self, level="INFO"):
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s:%(lineno)d] %(message)s"
                }
            },
            "handlers": {
                "console": {"level": level, "class": "logging.StreamHandler", "formatter": "standard"},
                "debug-logfile": {"level": "DEBUG", "class": "logging.FileHandler",
                                  "filename": os.path.join(self.BASE_PATH, "logs", "debug.log"), "formatter": "standard"},
                "info-logfile": {"level": "INFO", "class": "logging.FileHandler",
                                 "filename": os.path.join(self.BASE_PATH, "logs", "info.log"), "formatter": "standard"},
                "error-logfile": {"level": "ERROR", "class": "logging.FileHandler",
                                  "filename": os.path.join(self.BASE_PATH, "logs", "error.log"), "formatter": "standard"},
            },
            "loggers": {
                "__main__": {"level": level, "handlers": ["console", "debug-logfile", "info-logfile", "error-logfile"], "propagate": True},
                "src": {"level": level, "handlers": ["console", "debug-logfile", "info-logfile", "error-logfile"], "propagate": True},
            }
        }
        return logging_config
