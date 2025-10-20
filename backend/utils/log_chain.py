import hashlib
from pathlib import Path
from typing import Iterable


class LogChain:
    """Simple append-only hash chain for tamper-evidence (optional).

    hash_n = sha256(hash_{n-1} || batch_n)
    """

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self._last = b""
        if state_file.exists():
            self._last = bytes.fromhex(state_file.read_text().strip())

    def update(self, lines: Iterable[bytes]) -> str:
        h = hashlib.sha256()
        h.update(self._last)
        for line in lines:
            h.update(line)
        digest = h.digest()
        self._last = digest
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(digest.hex())
        return digest.hex()

