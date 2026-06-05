import threading
import time

class ScoreBuffer:
    """
    Thread-safe buffer holding the latest stress score per modality.
    Each modality writes independently. Fusion reads from all simultaneously.
    Scores older than STALE_THRESHOLD_S are excluded from fusion.
    """
    STALE_THRESHOLD_S = 15  # seconds before a score is considered stale

    def __init__(self):
        self._lock  = threading.Lock()
        self._store = {}  # modality -> {score, indicators, timestamp}

    def write(self, modality: str, score: float, indicators: dict = None):
        with self._lock:
            self._store[modality] = {
                'score':      score,
                'indicators': indicators or {},
                'timestamp':  time.time(),
            }

    def read(self, modality: str):
        with self._lock:
            entry = self._store.get(modality)
            if entry is None:
                return None
            if time.time() - entry['timestamp'] > self.STALE_THRESHOLD_S:
                return None  # stale
            return entry

    def read_all(self):
        now = time.time()
        with self._lock:
            return {
                k: v for k, v in self._store.items()
                if now - v['timestamp'] <= self.STALE_THRESHOLD_S
            }

    def clear(self, modality: str = None):
        with self._lock:
            if modality:
                self._store.pop(modality, None)
            else:
                self._store.clear()

# Singleton instance shared across all Flask routes
score_buffer = ScoreBuffer()
