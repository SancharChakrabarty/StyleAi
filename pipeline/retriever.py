"""
pipeline/retriever.py
Fashion Trend Retriever — No-Docker Version

Uses a local file-based vector store (numpy + JSON) instead of Qdrant.
No Docker, no external services required. Everything saved to data/vector_store/.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
STORE_DIR       = Path("data/vector_store")
VECTORS_FILE    = STORE_DIR / "vectors.npy"
DOCUMENTS_FILE  = STORE_DIR / "documents.json"


class FashionTrendRetriever:
    def __init__(
        self,
        store_dir:       str = str(STORE_DIR),
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        self.store_dir      = Path(store_dir)
        self.vectors_file   = self.store_dir / "vectors.npy"
        self.documents_file = self.store_dir / "documents.json"
        self.store_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading embedding model: %s", embedding_model)
        self.encoder = SentenceTransformer(embedding_model)

        self._vectors:   Optional[np.ndarray] = None
        self._documents: List[Dict[str, str]] = []
        self._load()

    def _load(self) -> None:
        if self.vectors_file.exists() and self.documents_file.exists():
            self._vectors = np.load(str(self.vectors_file))
            with open(self.documents_file, "r", encoding="utf-8") as f:
                self._documents = json.load(f)
            logger.info("Loaded vector store: %d documents", len(self._documents))
        else:
            logger.info("No existing vector store found at %s.", self.store_dir)

    def _save(self) -> None:
        np.save(str(self.vectors_file), self._vectors)
        with open(self.documents_file, "w", encoding="utf-8") as f:
            json.dump(self._documents, f, ensure_ascii=False, indent=2)
        logger.info("Vector store saved: %d documents", len(self._documents))

    def collection_count(self) -> int:
        return len(self._documents)

    def delete_collection(self) -> None:
        self._vectors   = None
        self._documents = []
        self.vectors_file.unlink(missing_ok=True)
        self.documents_file.unlink(missing_ok=True)
        logger.info("Vector store cleared.")

    def ingest(self, documents: List[Dict[str, str]], batch_size: int = 64) -> int:
        if not documents:
            return 0

        texts = [doc["text"] for doc in documents]
        logger.info("Embedding %d documents …", len(texts))

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            embs  = self.encoder.encode(batch, show_progress_bar=True)
            all_embeddings.append(embs)

        new_vectors = np.vstack(all_embeddings).astype(np.float32)
        norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
        new_vectors = new_vectors / np.maximum(norms, 1e-10)

        if self._vectors is not None and len(self._vectors) > 0:
            self._vectors   = np.vstack([self._vectors, new_vectors])
            self._documents = self._documents + list(documents)
        else:
            self._vectors   = new_vectors
            self._documents = list(documents)

        self._save()
        return len(documents)

    def _build_query(self, attributes: Dict[str, str]) -> str:
        """
        Convert CelebA-style attribute dict → natural language search query.
        Handles our trained attribute names: facial_hair, hair_color, eyeglasses,
        bald, wavy_hair, straight_hair, high_cheekbones, pale_skin, male.
        """
        parts = []

        # Gender
        gender = "man" if attributes.get("male") == "yes" else "woman"
        parts.append(f"a {gender}")

        # Hair
        if attributes.get("bald") == "yes":
            parts.append("who is bald")
        else:
            hair_color   = attributes.get("hair_color", "")
            wavy         = attributes.get("wavy_hair") == "yes"
            straight     = attributes.get("straight_hair") == "yes"
            texture      = "wavy" if wavy else ("straight" if straight else "")
            hair_desc    = " ".join(filter(None, [hair_color, texture, "hair"]))
            if hair_desc.strip() != "hair":
                parts.append(f"with {hair_desc}")

        # Facial hair
        if attributes.get("facial_hair") == "yes":
            parts.append("with facial hair")
        elif attributes.get("no_beard") == "yes":
            parts.append("clean-shaven")

        # Eyeglasses
        if attributes.get("eyeglasses") == "yes":
            parts.append("who wears glasses")

        # Cheekbones / skin
        if attributes.get("high_cheekbones") == "yes":
            parts.append("with high cheekbones")
        if attributes.get("pale_skin") == "yes":
            parts.append("with fair skin")

        subject = " ".join(parts) if parts else "a person"
        return (
            f"Fashion trends and style recommendations for {subject}. "
            "What clothing styles, colors, grooming tips, and current trends suit them?"
        )

    def retrieve(
        self,
        attributes:      Dict[str, str],
        top_k:           int = 5,
        score_threshold: float = 0.0,
        filter_season:   Optional[str] = None,
    ) -> List[str]:
        if self._vectors is None or len(self._documents) == 0:
            logger.warning("Vector store is empty. Run bootstrap first.")
            return []

        query_text = self._build_query(attributes)
        query_vec  = self.encoder.encode([query_text], show_progress_bar=False)[0].astype(np.float32)

        norm = np.linalg.norm(query_vec)
        if norm > 1e-10:
            query_vec = query_vec / norm

        scores = self._vectors @ query_vec

        if filter_season:
            for i, doc in enumerate(self._documents):
                season = doc.get("season", "all-seasons")
                if season not in (filter_season, "all-seasons"):
                    scores[i] = -1.0

        scores[scores < score_threshold] = -1.0
        top_indices = np.argsort(scores)[::-1][:top_k]

        snippets = []
        for idx in top_indices:
            if scores[idx] < 0:
                continue
            snippets.append(self._documents[idx]["text"])

        logger.info("Retrieved %d trend snippets for query: '%s…'", len(snippets), query_text[:60])
        return snippets

    def health(self) -> Dict[str, Any]:
        return {
            "status":         "ok" if self.collection_count() > 0 else "empty",
            "backend":        "local-file",
            "store_dir":      str(self.store_dir),
            "document_count": self.collection_count(),
        }


def bootstrap_sample_trends() -> None:
    sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "fashion_trends"))
    from sample_trends import TREND_DOCUMENTS  # noqa: PLC0415

    retriever = FashionTrendRetriever()
    if retriever.collection_count() == 0:
        logger.info("Bootstrapping vector store with sample trend documents …")
        retriever.ingest(TREND_DOCUMENTS)
    else:
        logger.info("Vector store already has %d documents.", retriever.collection_count())


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("bootstrap")
    r = sub.add_parser("retrieve")
    r.add_argument("--attributes", required=True)
    r.add_argument("--top_k", type=int, default=5)
    sub.add_parser("status")
    sub.add_parser("reset")
    args = parser.parse_args()

    if args.cmd == "bootstrap":
        bootstrap_sample_trends()
    elif args.cmd == "retrieve":
        ret      = FashionTrendRetriever()
        attrs    = json.loads(args.attributes)
        snippets = ret.retrieve(attrs, top_k=args.top_k)
        for i, s in enumerate(snippets, 1):
            print(f"\n── Trend {i} ──\n{s}")
    elif args.cmd == "status":
        print(json.dumps(FashionTrendRetriever().health(), indent=2))
    elif args.cmd == "reset":
        FashionTrendRetriever().delete_collection()
        print("Vector store reset.")
    else:
        parser.print_help()
