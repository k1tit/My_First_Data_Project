import os

APP_NAME = os.getenv("APP_NAME", "Home Credit Scoring API")
API_VERSION = os.getenv("API_VERSION", "0.1.0")

DB_HOST = os.getenv("DB_HOST", "database")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "scoring")
DB_USER = os.getenv("DB_USER", "scoring")
DB_PASS = os.getenv("DB_PASS", "scoring")

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
SCORING_QUEUE = os.getenv("SCORING_QUEUE", "scoring_jobs")

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "/artifacts")
MODEL_VERSION = os.getenv("MODEL_VERSION", "hw7-improved-v1")


def database_url() -> str:
    return f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
