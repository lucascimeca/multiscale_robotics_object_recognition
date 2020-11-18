import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


class WrsRecognitionCNN:

    def __init__(self, filename="recognition_CNN/networks/gray_deep-gpu_net.pb", img_type='gray'):
        self.height = self.width = 300
        if img_type == 'gray':
            self.channels = 1
        else:
            self.channels = 3
        self.graph = load_graph(filename)

        self.inputs = self.graph.get_tensor_by_name('prefix/inputs:0')
        self.outputs = self.graph.get_tensor_by_name('prefix/predictions/Softmax:0')
        self.batch_num = self.graph.get_tensor_by_name('prefix/Placeholder:0')

        # We launch a Session
        self.sess = tf.Session(graph=self.graph)

    def predict(self, input_batch):
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants
        probabilities = self.sess.run(self.outputs, feed_dict={
            self.inputs: input_batch,
            self.batch_num: input_batch.shape[0]
        })

        classes = np.argmax(probabilities, axis=1)+2
        return classes, probabilities
