from queue import Queue
from big_fiubrother_core import setup
from big_fiubrother_core import SignalHandler
from big_fiubrother_classifier import FaceEmbedderThread, FaceClassifierThread, ClassifiedFaceMessagePublisher, DetectedFaceConsumer

if __name__ == "__main__":
    configuration = setup('Big Fiubrother Face Classifier Application')

    print('[*] Configuring big_fiubrother_classifier')

    consumer_to_embedder_queue = Queue()
    embedder_to_classifier_queue = Queue()
    classifier_to_publisher_queue = Queue()

    embedder_thread = FaceEmbedderThread(configuration['face_embedder'], consumer_to_embedder_queue,
                                         embedder_to_classifier_queue)
    classifier_thread = FaceClassifierThread(configuration['face_classifier'], embedder_to_classifier_queue,
                                             classifier_to_publisher_queue)
    publisher_thread = ClassifiedFaceMessagePublisher(configuration['publisher'], classifier_to_publisher_queue)
    consumer = DetectedFaceConsumer(configuration['consumer'], consumer_to_embedder_queue)

    signal_handler = SignalHandler(callback=consumer.stop)

    print('[*] Configuration finished. Starting big_fiubrother_classifier!')

    embedder_thread.start()
    classifier_thread.start()
    publisher_thread.start()
    consumer.start()

    # Signal Handled STOP

    embedder_thread.stop()
    classifier_thread.stop()
    publisher_thread.stop()

    embedder_thread.wait()
    classifier_thread.wait()
    publisher_thread.wait()

    print('[*] big_fiubrother_classifier stopped!')