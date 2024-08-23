import numpy as np
import tensorflow as tf
import os


class SOM:
    def __init__(self, width, height, input_dim, learning_rate=0.1, radius=None, num_iterations=100):
        self.width = width
        self.height = height
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        if radius is None:
            self.radius = max(width, height) / 2
        else:
            self.radius = radius
        self.num_iterations = num_iterations
        self.weights = tf.Variable(tf.random.normal([width*height, input_dim]))
        self.locations = self._create_locations() # Checkpointing
        self.checkpoint = tf.train.Checkpoint(weights=self.weights)
        self.checkpoint_manager = None  # To be defined in save method
    def _create_locations(self):
        return np.array([[y, x] for y in range(self.height) for x in range(self.width)])
    def _find_bmu(self, input_vec):
        squared_differences = tf.reduce_sum(tf.square(self.weights - tf.stack([input_vec for i in range(self.width*self.height)])), axis=1)
        bmu_index = tf.argmin(squared_differences, axis=0)
        bmu_loc = tf.cast(tf.stack([tf.math.mod(bmu_index, self.width), tf.math.floordiv(bmu_index, self.width)]), tf.float32)
        return bmu_loc
    def _update_weights(self, input_vec, bmu_loc, iteration):
        learning_rate = self.learning_rate * (1 - iteration / self.num_iterations)
        sigma = self.radius * (1 - iteration / self.num_iterations)
        squared_distance_from_bmu = tf.reduce_sum(tf.square(self.locations - tf.stack([bmu_loc for i in range(self.width*self.height)])), axis=1)
        neighbourhood_func = tf.exp(-squared_distance_from_bmu / (2 * sigma * sigma))
        learning_rate_times_neighbourhood = learning_rate * neighbourhood_func
        delta_weights = tf.stack([learning_rate_times_neighbourhood[i] * (input_vec - self.weights[i]) for i in range(self.width*self.height)])
        new_weights = self.weights + delta_weights
        return new_weights
    def train(self, input_data):
        for iteration in range(self.num_iterations):
            for input_vec in input_data:
                bmu_loc = self._find_bmu(input_vec)
                self.weights.assign(self._update_weights(input_vec, bmu_loc, iteration))
        print(f"Epoch {iteration+1}/{self.num_iterations} completed.")
    def save(self, directory="som_checkpoints"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        if self.checkpoint_manager is None:
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory, max_to_keep=3)
        save_path = self.checkpoint_manager.save()  # Let the manager handle naming
        print(f"Checkpoint saved at {save_path}")
    def load(self, directory="som_checkpoints"):
        if self.checkpoint_manager is None:
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory, max_to_keep=3)
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        if self.checkpoint_manager.latest_checkpoint:
            print("Restored from {}".format(self.checkpoint_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")










