# These scripts train small SMT models from PE corpora, that are used to generate
# synthetic parallel corpora (so as to inflate the original PE corpus).
# This generates large amounts of noisy (EN, EN) and (FR, EN) data that can be used to pre-train an NPE model (the
# target EN is clean, so it can help train the decoder parameters).
