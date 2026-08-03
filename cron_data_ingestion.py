import shutil

from apscheduler.schedulers.blocking import BlockingScheduler
from constants import PROCESSING_DIR, TRANSCRIPTS_DIR, RECORDINGS_DIR, PRESIST_DIRECTORY, EMBEDDING_MODEL_NAME, COLLECTION_NAME
from data_ingestion import DataIngestion
from vector_store import VectorStore
from embedding_manager import EmbeddingManager

def job():
    # Skip if another processing job is still running
    if PROCESSING_DIR.exists():
        print("Processing directory already exists. Skipping...")
        return

    # Create transcripts directory if it doesn't exist
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    # Skip if there are no files to process
    if not any(TRANSCRIPTS_DIR.iterdir()):
        print("No transcripts to process.")
        return

    # Move transcripts to processing
    shutil.move(str(TRANSCRIPTS_DIR), str(PROCESSING_DIR))

    # Create a fresh transcripts directory for new files
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        print("Processing transcripts...")
        vector_store = VectorStore(
            presist_directory=PRESIST_DIRECTORY, collection_name=COLLECTION_NAME
        )
        embedding_manager = EmbeddingManager(model_name=EMBEDDING_MODEL_NAME)
        data_ingestion = DataIngestion(
            vector_store=vector_store,
            embedding_manager=embedding_manager
        )
        data_ingestion.ingest_data()

        print("Processing complete.")

    finally:
        # Clean up processing directory
        if PROCESSING_DIR.exists():
            shutil.rmtree(PROCESSING_DIR)
        if RECORDINGS_DIR.exists():
            shutil.rmtree(RECORDINGS_DIR)
            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)


scheduler = BlockingScheduler()

scheduler.add_job(
    job,
    trigger="cron",
    minute="*/2",
    max_instances=1,      # Prevent overlapping runs
    coalesce=True,        # Merge missed executions
)

print("Scheduler started...")
scheduler.start()