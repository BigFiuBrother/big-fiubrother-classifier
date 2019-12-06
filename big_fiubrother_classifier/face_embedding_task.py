from big_fiubrother_core import QueueTask
from big_fiubrother_core.db import (
    Database,
    FaceEmbedding
)
from big_fiubrother_core.messages import (
    FaceEmbeddingMessage,
    FaceClassificationMessage
)
from big_fiubrother_classifier.face_embedder_factory import FaceEmbedderFactory
import cv2
import numpy as np


class FaceEmbeddingTask(QueueTask):

    def __init__(self, configuration, input_queue, output_queue):
        super().__init__(input_queue)
        self.output_queue = output_queue
        self.configuration = configuration

        self.db = None
        self.face_embedder = None

    def init(self):
        self.face_embedder = FaceEmbedderFactory.build(self.configuration['face_embedder'])
        self.db = Database(self.configuration['db'])

    def close(self):
        self.face_embedder.close()
        self.db.close()

    def execute_with(self, message):
        face_embedding_message: FaceEmbeddingMessage = message

        # Get face
        face = face_embedding_message.face_bytes
        face_id = face_embedding_message.detected_face_id
        video_chunk_id = face_embedding_message.video_chunk_id

        print(face)
        # Perform face embedding
        #print("- Performing embedding - face_id: " + str(face_embedding_message.detected_face_id))
        embedding = self.face_embedder.get_embedding_mem(face)
        #embedding = np.array([1, 2, 3])

        # Insert result into database
        face_embedding = FaceEmbedding(face_id=face_id, embedding=list(embedding.astype(float)))
        self.db.add(face_embedding)

        # Queue face classification job
        face_classification_message = FaceClassificationMessage(video_chunk_id, face_id, embedding)
        self.output_queue.put(face_classification_message)

    def _stop(self):
        self.input_queue.put(None)
