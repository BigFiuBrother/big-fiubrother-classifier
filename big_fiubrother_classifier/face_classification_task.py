from big_fiubrother_core import QueueTask
from big_fiubrother_core.db import (
    Database,
    Face
)
from big_fiubrother_core.messages import (
    FaceClassificationMessage,
    ProcessedFaceMessage
)
import big_fiubrother_classifier.classifier_support_vector
from big_fiubrother_classifier.classifier_support_vector import SVClassifier
import cv2
import numpy as np


class FaceClassificationTask(QueueTask):

    def __init__(self, configuration, input_queue, output_queue):
        super().__init__(input_queue)
        self.output_queue = output_queue
        self.configuration = configuration

        self.db = None
        self.face_classifier = None

    def init(self):
        # load pre-trained classifier from local file?
        # load from database?
        self.face_classifier = SVClassifier.load(self.configuration['face_classifier'])
        self.db = Database(self.configuration['db'])

    def close(self):
        self.db.close()

    def execute_with(self, message):
        face_classification_message: FaceClassificationMessage = self.input_queue.get()

        if face_classification_message is not None:

            # Get message
            embedding = face_classification_message.face_embedding

            # Do face classification
            print("- Performing classification")
            classification_index, classification_probability = self.face_classifier.predict(embedding)

            # Update database face row with result
            face_id = face_classification_message.face_id
            face: Face = self.db.get(Face, face_id)
            face.classification_id = int(classification_index)
            face.probability_classification = float(classification_probability)
            print(type(face.classification_id))
            print(type(face.probability_classification))
            self.db.update()

            # Notify of face analysis completion
            scheduler_message = ProcessedFaceMessage(face_id) # cambiar a frame id?
            self.output_queue.put(scheduler_message)

    def _stop(self):
        self.input_queue.put(None)
