from queue import Queue
from big_fiubrother_core import (
    SignalHandler,
    StoppableThread,
    PublishToRabbitMQ,
    ConsumeFromRabbitMQ,
    runtime_context
)
from big_fiubrother_classifier import (
    FaceEmbeddingTask,
    FaceClassificationTask
)
# magia negra para que funcione pickle
from big_fiubrother_classifier.classifier_support_vector import SVClassifier


if __name__ == "__main__":
    with runtime_context('Big Fiubrother Face Classifier Application') as configuration:

        print('[*] Configuring big_fiubrother_classifier')

        consumer_to_embedder_queue = Queue()
        embedder_to_classifier_queue = Queue()
        classifier_to_publisher_queue = Queue()

        consumer = ConsumeFromRabbitMQ(configuration=configuration['consumer'],
                                       output_queue=consumer_to_embedder_queue)

        embedder_thread = StoppableThread(
            FaceEmbeddingTask(configuration=configuration['face_embedder'],
                              input_queue=consumer_to_embedder_queue,
                              output_queue=embedder_to_classifier_queue))

        classifier_thread = StoppableThread(
            FaceClassificationTask(configuration=configuration['face_classifier'],
                                   input_queue=embedder_to_classifier_queue,
                                   output_queue=classifier_to_publisher_queue))

        publisher_thread = StoppableThread(
            PublishToRabbitMQ(configuration=configuration['publisher_to_scheduler'],
                              input_queue=classifier_to_publisher_queue))

        signal_handler = SignalHandler(callback=consumer.stop)

        print('[*] Configuration finished. Starting big_fiubrother_classifier!')

        # Start worker threads
        embedder_thread.start()
        classifier_thread.start()
        publisher_thread.start()

        # Start consumer on main thread
        consumer.init()
        consumer.execute()

        # Signal Handled STOP
        consumer.close()

        # Stop worker threads
        embedder_thread.stop()
        classifier_thread.stop()
        publisher_thread.stop()

        # Wait for worker threads
        embedder_thread.wait()
        classifier_thread.wait()
        publisher_thread.wait()

        print('[*] big_fiubrother_classifier stopped!')
