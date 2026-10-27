import os

# Tests use the deterministic, offline lexical embedder even where the `semantic` extra is
# installed. Tests of the semantic embedder ask for it explicitly and skip without it.
os.environ.setdefault("PCIM_EMBEDDER", "hashing-512")
