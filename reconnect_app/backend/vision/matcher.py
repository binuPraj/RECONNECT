import os
import numpy as np

from config import (
    GALLERY_PATH,
    KNOWN_MATCH_THRESHOLD
)


class FaceMatcher:

    def __init__(self):

        self.gallery = {}

        self.load_gallery()

    def load_gallery(self):

        if not os.path.exists(
            GALLERY_PATH
        ):
            return

        for identity in os.listdir(
            GALLERY_PATH
        ):

            identity_path = os.path.join(
                GALLERY_PATH,
                identity
            )

            if not os.path.isdir(
                identity_path
            ):
                continue

            embeddings = []

            for filename in os.listdir(
                identity_path
            ):

                if filename.endswith(".npy"):

                    embedding_path = os.path.join(
                        identity_path,
                        filename
                    )

                    embedding = np.load(
                        embedding_path
                    )

                    embeddings.append(
                        embedding
                    )

            if embeddings:

                self.gallery[identity] = (
                    np.array(
                        embeddings,
                        dtype=np.float32
                    )
                )

    def match(self, embedding):

        best_identity = None

        best_similarity = -1.0

        for identity, embeddings in (
            self.gallery.items()
        ):

            for stored_embedding in embeddings:

                similarity = float(
                    np.dot(
                        embedding,
                        stored_embedding
                    )
                )

                if similarity > best_similarity:

                    best_similarity = similarity

                    best_identity = identity

        if (
            best_identity is not None
            and best_similarity
            >= KNOWN_MATCH_THRESHOLD
        ):

            return {

                "matched": True,

                "identity": best_identity,

                "similarity": best_similarity

            }

        return {

            "matched": False,

            "identity": None,

            "similarity": best_similarity
        }