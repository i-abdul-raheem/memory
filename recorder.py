from pathlib import Path
from datetime import datetime
from queue import Queue, Full, Empty
from threading import Event, Thread
import time

import numpy as np
import sounddevice as sd
import whisper
from scipy.io.wavfile import write

from constants import RECORDINGS_DIR, TRANSCRIPTS_DIR, WHISPER_MODEL_NAME

SAMPLE_RATE = 16_000
CHANNELS = 1
CHUNK_SECONDS = 30                 # Lower = quicker partial results
CHUNK_FRAMES = SAMPLE_RATE * CHUNK_SECONDS
MODEL_NAME = WHISPER_MODEL_NAME                # Use "tiny" on very limited hardware
TASK = "translate"                 # Change to "transcribe" to keep original language
MAX_PENDING_CHUNKS = 6             # About one minute of queued audio

audio_dir = RECORDINGS_DIR
text_dir = TRANSCRIPTS_DIR
audio_dir.mkdir(parents=True, exist_ok=True)
text_dir.mkdir(parents=True, exist_ok=True)

audio_queue = Queue(maxsize=MAX_PENDING_CHUNKS)
stop_event = Event()

# Pre-allocate the current audio chunk so the recording callback stays light.
current_chunk = np.empty((CHUNK_FRAMES, CHANNELS), dtype=np.int16)
filled_frames = 0


def transcribe_worker():
    model = whisper.load_model(MODEL_NAME)

    while not stop_event.is_set() or not audio_queue.empty():
        try:
            timestamp, audio = audio_queue.get(timeout=0.5)
        except Empty:
            continue

        audio_path = audio_dir / f"{timestamp}.wav"
        text_path = text_dir / f"{timestamp}.txt"

        try:
            write(audio_path, SAMPLE_RATE, audio)
            result = model.transcribe(
                str(audio_path),
                task=TASK,
                fp16=False,        # Required for most CPU-only IoT devices
                verbose=False,
            )
            text_path.write_text(result["text"].strip(), encoding="utf-8")
            print(f"Saved transcript: {text_path}")
        except Exception as error:
            print(f"Could not process {timestamp}: {error}")
        finally:
            audio_queue.task_done()


def audio_callback(indata, frames, time_info, status):
    global filled_frames, current_chunk

    if status:
        print(f"Audio warning: {status}")

    start = 0
    while start < frames:
        space = CHUNK_FRAMES - filled_frames
        take = min(space, frames - start)

        current_chunk[filled_frames:filled_frames + take] = indata[start:start + take]
        filled_frames += take
        start += take

        if filled_frames == CHUNK_FRAMES:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            try:
                audio_queue.put_nowait((timestamp, current_chunk.copy()))
            except Full:
                # Recording continues; this chunk is dropped rather than blocking audio capture.
                print("Warning: translation queue is full; dropped one audio chunk.")

            filled_frames = 0


worker = Thread(target=transcribe_worker, daemon=True)
worker.start()

try:
    print("Continuous recording started. Press Ctrl+C to stop.")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        callback=audio_callback,
        blocksize=1600,  # 100 ms audio blocks
    ):
        while True:
            time.sleep(0.2)

except KeyboardInterrupt:
    print("\nStopping—finishing queued recordings...")
    stop_event.set()
    worker.join()
    print("Stopped.")