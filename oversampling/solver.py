import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf

import cudf, cuml
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
from cuml.cluster import KMeans

from autoencdoer_model import *
from siamese_network_model import SiameseNetwork

from tqdm import tqdm


class Solver:
    def __init__(self, train_ds, test_ds, epochs=32, features_dim=76, knn_data=None, siamese_data=None):
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.num_epochs = epochs
        self.features_dim = features_dim

        # using the SimpleAE model
        self.encoder = Encoder(input_shape=features_dim)
        self.decoder = Decoder(original_dim=features_dim)

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam()

        # loss function
        self.loss_func = tf.keras.losses.MeanSquaredError()

        # set percentile thresh
        self.percentile = 70

        # show network architecrute
        # setup tensorboard

        if knn_data is not None and siamese_data is not None:
           self.train_siamese_model(siamese_data)
           self.train_knn_model(knn_data) 

    
    def train_knn_model(self, knn_data):
        self.knn_data = knn_data
        X_rapids = cudf.DataFrame(knn_data)

        # defining the distance metric to be using the siamese network
        def siamese_distance(sample1, sample2):
            sample1 = tf.expand_dims(sample1, axis=0)
            sample2 = tf.expand_dims(sample2, axis=0)
            output, distance_vector = self.siamese_network((sample1, sample2))
            return tf.math.reduce_sum(distance_vector, axis=-1)

        # self.knn_model = cuNearestNeighbors(metric=siamese_distance)
        self.knn_model = NearestNeighbors(metric=siamese_distance, n_jobs=-1)
        print("Start fitting KNN")
        #self.knn_model.fit(X_rapids)
        self.knn_model.fit(knn_data)
        print("KNN model is fitted")

    def train_siamese_model(self, siamese_data):
        X_pairs, y_pairs = siamese_data
        # using the siamese model
        self.siamese_network = SiameseNetwork()
        # compile the siamese network
        self.siamese_network.compile(optimizer='adam', loss='binary_crossentropy')

        # train the siamese network
        self.siamese_network.fit(
            [X_pairs[0], X_pairs[1]],
            y_pairs,
            batch_size=64,
            epochs=10
        )

    def save_weights(self, path, dataset_name):
        encoder_path = path + f"/epochs_{self.num_epochs}_{dataset_name}_encoder_weights"
        decoder_path = path + f"/epochs_{self.num_epochs}_{dataset_name}_decoder_weights"

        np.save(encoder_path, self.encoder.get_weights())
        np.save(decoder_path, self.decoder.get_weights())

    def load_weights(self, encoder_path, decoder_path):
        self.encoder = Encoder(input_shape=self.features_dim)
        self.decoder = Decoder(original_dim=self.features_dim)

        encoder_trained_weights = np.load(encoder_path, allow_pickle=True)
        decoder_trained_weights = np.load(decoder_path, allow_pickle=True)

        # calculating just one feed-forward in order to set the layers weights' shapes
        for batch_X_train, batch_y_train in self.train_ds:
            latent_var = self.encoder(batch_X_train)
            self.decoder(latent_var)
            break

        self.encoder.set_weights(encoder_trained_weights)
        self.decoder.set_weights(decoder_trained_weights)

    def train(self):

        # loss function
        self.loss_func = tf.keras.losses.MeanSquaredError()
        for epoch in range(self.num_epochs):
            epoch_loss_mean = tf.keras.metrics.Mean()

            for step, (x_batch_train, y_batch_train) in enumerate(self.train_ds):
                # if step%1000 == 0: print(f'Step: {step} in epoch {epoch}')
                loss = self.train_step(x_batch_train)

                # keep track of the metrics
                epoch_loss_mean.update_state(loss)

            # update metrics after each epoch
            print(f'Epoch {epoch+1} loss mean: {epoch_loss_mean.result()}')


    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            latent_var = self.encoder(inputs)
            outputs = self.decoder(latent_var)
            loss = self.loss_func(inputs, outputs)

            trainable_vars = self.encoder.trainable_variables \
                             + self.decoder.trainable_variables

        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        return loss


    def test(self):
        # loss function - with reduction equals to `NONE` in order to get the loss of every test example
        self.loss_func = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        # iterating through the train to calculate the loss of every train instance
        train_loss = []
        for step, (x_batch_train, y_batch_train) in enumerate(self.train_ds):
            loss = self.test_step(x_batch_train)
            train_loss.append(loss.numpy())

        # iterating through the test to calculate the loss of every test instance
        test_loss = []
        test_labels = []
        for step, (x_batch_test, y_batch_test) in enumerate(self.test_ds):
            reconstruction_loss = self.test_step(x_batch_test)
            test_loss.append(reconstruction_loss.numpy())
            test_labels.append(y_batch_test.numpy())

        train_loss = np.concatenate(train_loss, axis=0)

        test_loss = np.concatenate(test_loss, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        combined_loss = np.concatenate([train_loss, test_loss], axis=0)

        # setting the threshold to be a value of a 80% of the loss of all the examples
        # give a reference from the GMM papper
        thresh = np.percentile(combined_loss, self.percentile)
        print("Threshold :", thresh)
        # thresh = np.mean(combined_loss) + np.std(combined_loss)

        y_pred = tf.where(test_loss > thresh, 1, 0).numpy().astype(int)
        y_true = np.asarray(test_labels).astype(int)

        from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score, roc_auc_score
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f_score, support = prf(y_true, y_pred, average='binary')
        auc = roc_auc_score(y_true, y_pred)

        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, AUC : {:0.4f}".format(
            accuracy, precision,
            recall, f_score, auc))

        return accuracy, precision, recall, f_score, auc
        

    def test_tta(self, num_neighbors, num_augmentations):

        # loss function - with reduction equals to `NONE` in order to get the loss of every test example
        self.loss_func = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        # iterating through the train to calculate the loss of every train instance
        train_loss = []
        for step, (x_batch_train, y_batch_train) in enumerate(self.train_ds):
            loss = self.test_step(x_batch_train)
            train_loss.append(loss.numpy())

        print("Done iterating through all training set")

        # iterating through the test to calculate the loss of every test instance
        test_loss = []
        test_labels = []
        for step, (x_batch_test, y_batch_test) in tqdm(enumerate(self.test_ds)):
            # x_batch_test: ndarray of shape (batch_size, num_dataset_features)
            # y_batch_test: ndarray of shape (batch_size,)

            reconstruction_loss = self.test_step(x_batch_test).numpy()
            test_labels.append(y_batch_test.numpy())

            # neighbors_indices: ndarray of shape (batch_size, num_of_augmentations)
            neighbors_indices = self.knn_model.kneighbors(X=np.array(x_batch_test), n_neighbors=num_neighbors, return_distance=False)
            neighbors_indices = np.squeeze(neighbors_indices)

            # tta_features_batch: ndarray of shape (batch_size, num_dataset_features)
            neighbors_batch_features = knn_data[neighbors_indices]

            tta_reconstruction = []
            for neighbors_features in neighbors_batch_features:
                # tta_features = self.get_oversampling_tta_sample(neighbors_features, neighbors_labels, num_augmentations, oversampling_method, fake_tta_features, fake_tta_labels)
                k_means = KMeans(n_clusters=num_augmentations)
                neighbors_features = neighbors_features.astype(np.float32)
                k_means.fit(neighbors_features)
                tta_features = k_means.cluster_centers_

                # making the test phase for the tta sample
                tta_reconstruction_loss = self.test_step(tta_features).numpy()
                tta_reconstruction.append(tta_reconstruction_loss)

            # calculating the final loss - mean over primary sample and tta samples
            for primary_loss, tta_loss in list(zip(reconstruction_loss, tta_reconstruction)):
                combined_tta_loss = np.concatenate([[primary_loss], tta_loss])
                test_loss.append(np.mean(combined_tta_loss))

        train_loss = np.concatenate(train_loss, axis=0)

        # test_loss = np.concatenate(test_loss, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        combined_loss = np.concatenate([train_loss, test_loss], axis=0)

        # setting the threshold to be a value of a 80% of the loss of all the examples
        # give a reference from the GMM papper
        thresh = np.percentile(combined_loss, self.percentile)
        print("Threshold :", thresh)
        # thresh = np.mean(combined_loss) + np.std(combined_loss)

        y_pred = tf.where(test_loss > thresh, 1, 0).numpy().astype(int)
        y_true = np.asarray(test_labels).astype(int)

        from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score, roc_auc_score
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f_score, support = prf(y_true, y_pred, average='binary')
        auc = roc_auc_score(y_true, y_pred)

        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, AUC : {:0.4f}".format(
            accuracy, precision,
            recall, f_score, auc))

        return accuracy, precision, recall, f_score, auc

    @tf.function
    def test_step(self, inputs):
        latent_var = self.encoder(inputs)
        reconstructed = self.decoder(latent_var)
        reconstruction_loss = self.loss_func(inputs, reconstructed)

        return reconstruction_loss
