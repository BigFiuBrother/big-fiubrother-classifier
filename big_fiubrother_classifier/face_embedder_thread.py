from big_fiubrother_core import StoppableThread
from big_fiubrother_core.messages import FaceEmbeddingMessage, FaceClassificationMessage
from big_fiubrother_classifier.face_embedder_factory import FaceEmbedderFactory
import cv2
import numpy as np


class FaceClassifierThread(StoppableThread):

    def __init__(self, configuration, input_queue, output_queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.face_embedder = FaceEmbedderFactory.build(configuration['face_embedder'])

    def _execute(self):
        face_embedding_message = self.input_queue.get()

        if face_embedding_message is not None:

            # Get frame
            frame = face_embedding_message.frame_bytes

            face_embeddings = []
            for face_box in face_embedding_message.face_boxes:
                # Get face from frame
                face = frame[face_box[1]:face_box[3], face_box[0]:face_box[2]]

                # Perform face classification
                embedding = self.face_embedder.get_embedding_mem(face)
                face_embeddings.append(embedding)

            # Queue face classification job
            face_classification_message = FaceClassificationMessage(face_embedding_message.camera_id,
                                                                    face_embedding_message.timestamp,
                                                                    face_embedding_message.frame_id,
                                                                    face_embedding_message.frame_bytes,
                                                                    face_embedding_message.face_boxes,
                                                                    face_embeddings)
            self.output_queue.put(face_classification_message)

    def _stop(self):
        self.input_queue.put(None)
