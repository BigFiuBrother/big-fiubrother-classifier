from big_fiubrother_core import StoppableThread
from big_fiubrother_core.messages import FaceClassificationMessage, FrameInterpolatorNotificationMessage
from big_fiubrother_classifier.classifier_support_vector import SVClassifier
import cv2
import numpy as np


class FaceClassifierThread(StoppableThread):

    def __init__(self, configuration, input_queue, output_queue):
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue

        # load pre-trained classifier from local file?
        # load from database?
        self.face_classifier = SVClassifier.load(configuration['face_classifier'])

    def _execute(self):
        face_classification_message = self.input_queue.get()

        if face_classification_message is not None:

            # Get message
            embeddings = face_classification_message.face_embeddings

            # Do face classification
            predictions = self.face_classifier.predict_proba(embeddings)
            best_class_indices = np.argmax(predictions, axis=1)
            #best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

            # Save result to database for clustering?

            # Send notification to frame interpolator
            #self.output_queue.put(interpolator_notification_message)

    def _stop(self):
        self.input_queue.put(None)
