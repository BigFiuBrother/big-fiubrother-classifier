from big_fiubrother_core import QueueTask
from big_fiubrother_core.db import (
    Database,
    Face
)
from big_fiubrother_core.messages import (
    FaceClassificationMessage,
    AnalyzedVideoChunkMessage
)
import big_fiubrother_classifier.classifier_support_vector
from big_fiubrother_classifier.classifier_support_vector import SVClassifier
from big_fiubrother_core.synchronization import ProcessSynchronizer
import cv2
import numpy as np


class FaceClassificationTask(QueueTask):

    def __init__(self, configuration, input_queue, output_queue):
        super().__init__(input_queue)
        self.output_queue = output_queue
        self.configuration = configuration

        self.db = None
        self.face_classifier = None
        self.process_synchronizer = None

        self.threshold = 0.0

    def init(self):
        # load pre-trained classifier from local file?
        # load from database?
        self.face_classifier = SVClassifier.load(self.configuration['face_classifier']['model'])
        self.db = Database(self.configuration['db'])
        self.threshold = self.configuration['face_classifier']['threshold']
        self.process_synchronizer = ProcessSynchronizer(self.configuration['synchronization'])

    def close(self):
        self.db.close()

    def execute_with(self, message):
        face_classification_message: FaceClassificationMessage = message
        video_chunk_id = face_classification_message.video_chunk_id

        # Get message
        embedding = face_classification_message.face_embedding
        print()

        # Do face classification
        print("- Performing classification - face_id: " + str(face_classification_message.face_id))
        classification_index, classification_probability = self.face_classifier.predict(embedding)
        #classification_index, classification_probability = [0, 0.9]
        is_match = classification_probability > self.threshold

        # Update database face row with result
        face_id = face_classification_message.face_id
        face: Face = self.db.get(Face, face_id)
        face.classification_id = int(classification_index)
        face.probability_classification = float(classification_probability)
        face.is_match = is_match
        #print(type(face.classification_id))
        #print(type(face.probability_classification))
        self.db.update()

        print("- Classification id: " + str(face.classification_id) + ", prob: " + str(face.probability_classification))

        # Face analysis sync completion
        should_notify_scheduler = self.process_synchronizer.complete_face_task(
            video_chunk_id,
            face.frame_id,
            face_id
        )
        print(should_notify_scheduler)

        if should_notify_scheduler:
            print("- Notifying interpolator of video chunk analysis completion, id: " + str(video_chunk_id))
            # Notify scheduler of video chunk analysis completion
            scheduler_notification = AnalyzedVideoChunkMessage(video_chunk_id)
            self.output_queue.put(scheduler_notification)

    def _stop(self):
        self.input_queue.put(None)
