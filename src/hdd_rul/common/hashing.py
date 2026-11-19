"""Hachage déterministe : intégrité des archives et échantillonnage reproductible.

`hashlib.md5` est préféré au `hash()` intégré pour l'échantillonnage : ce
dernier est randomisé par processus (PYTHONHASHSEED), donc non reproductible
d'un run à l'autre, ce qui casserait la reproductibilité exigée. md5 n'est
utilisé ici que comme fonction de répartition, jamais à des fins de sécurité.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """sha256 d'un fichier, lu par blocs (les archives pèsent ~2 Go)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_bucket(key: str, *, salt: str = "", buckets: int = 100) -> int:
    """Associé `key` à un entier de [0, buckets) de manière stable et portable.

    Sert à tirer l'échantillon de disques sains sans stocker la liste des
    numéros de série retenus : `stable_bucket(serial) < taux` suffit, et donne
    le même échantillon à chaque exécution et dans chaque langage.
    """
    if buckets < 1:
        raise ValueError("buckets doit être >= 1")
    digest = hashlib.md5(f"{salt}:{key}".encode(), usedforsecurity=False).hexdigest()
    return int(digest[:8], 16) % buckets
