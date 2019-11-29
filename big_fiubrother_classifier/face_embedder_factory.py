import os
import yaml


class FaceEmbedderFactory:

    @staticmethod
    def build(configuration):

        face_embedder_type = configuration['type']

        if face_embedder_type == "movidius_facenet":
            from big_fiubrother_classifier.face_embedder_mvds_facenet import FaceEmbedderMovidiusFacenet
            return FaceEmbedderMovidiusFacenet(configuration['movidius_id'])

        elif face_embedder_type == "tensorflow_facenet":
            from big_fiubrother_classifier.face_embedder_tensorflow_facenet import FaceEmbedderTensorflowFacenet
            return FaceEmbedderTensorflowFacenet()

    @staticmethod
    def build_mvds_facenet():
        config_path = os.path.dirname(os.path.realpath(__file__)) + "/../config/config_mvds_facenet.yaml"
        return FaceEmbedderFactory.build(FaceEmbedderFactory._read_config_file(config_path))

    @staticmethod
    def build_tensorflow_facenet():
        config_path = os.path.dirname(os.path.realpath(__file__)) + "/../config/config_tensorflow_facenet.yaml"
        return FaceEmbedderFactory.build(FaceEmbedderFactory._read_config_file(config_path))

    @staticmethod
    def _read_config_file(path):
        with open(path) as config_file:
            configuration = yaml.load(config_file)
            return configuration['face_embedder']
