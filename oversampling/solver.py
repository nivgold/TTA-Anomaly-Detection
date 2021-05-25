import numpy as np
import tensorflow as tf

import cudf, cuml
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
from cuml.cluster import KMeans as cuKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

from sklearn.metrics import roc_curve
import scikitplot as skplt
import matplotlib.pyplot as plt

from autoencdoer_model import *
from siamese_network_model import SiameseNetwork

from tqdm import tqdm


class Solver:
    def __init__(self, train_ds, test_ds, dataset_name, epochs=32, features_dim=76, knn_data=None, siamese_data=None, with_cuml=True):
        self.dataset_name = dataset_name

        self.train_ds = train_ds
        self.test_ds = test_ds
        self.num_epochs = epochs
        self.features_dim = features_dim

        self.with_cuml = with_cuml

        # using the SimpleAE model
        self.encoder = Encoder(input_shape=features_dim)
        self.decoder = Decoder(original_dim=features_dim)

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam()

        # loss function
        self.loss_func = tf.keras.losses.MeanSquaredError()

        # set percentile thresh
        self.percentile = 80

        # show network architecrute
        # setup tensorboard

        if knn_data is not None and siamese_data is not None:
           self.train_siamese_model(siamese_data)
           self.train_knn_model(knn_data) 

    
    def train_knn_model(self, knn_data):
        self.knn_data = knn_data

        # defining the distance metric to be using the siamese network
        def siamese_distance(sample1, sample2):
            sample1 = tf.expand_dims(sample1, axis=0)
            sample2 = tf.expand_dims(sample2, axis=0)
            output, distance_vector = self.siamese_network((sample1, sample2))
            return tf.math.reduce_sum(distance_vector, axis=-1)

        if self.with_cuml:
            print("--- Using cuML ---")
            # define and train siamese-distance based KNN
            self.knn_model_siamese = cuNearestNeighbors(metric='cosine')
            latent_features = self.siam_internal_model(self.knn_data).numpy()
            self.knn_model_siamese.fit(latent_features)
            # define and train regular KNN
            self.knn_model_regular = cuNearestNeighbors()
            self.knn_model_regular.fit(self.knn_data)
        else:
            print("--- Not using cuML")
            # define and train siamese-distance based KNN
            self.knn_model_siamese = NearestNeighbors(metric=siamese_distance, n_jobs=-1)
            self.knn_model_siamese.fit(self.knn_data)
            # define and train regular KNN
            self.knn_model_regular = NearestNeighbors()
            self.knn_model_regular.fit(self.knn_data)

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
            epochs=10,
            # validation_split=0.2,
        )

        # save the latent features of every sample in a dict
        self.siam_internal_model = self.siamese_network.internal_model


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
        print("Baseline threshold :", thresh)
        # thresh = np.mean(combined_loss) + np.std(combined_loss)

        y_pred = tf.where(test_loss > thresh, 1, 0).numpy().astype(int)
        y_true = np.asarray(test_labels).astype(int)

        from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score, roc_auc_score
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f_score, support = prf(y_true, y_pred, average='binary')
        # auc = roc_auc_score(y_true, y_pred)
        auc = roc_auc_score(y_true, test_loss)

        # save test metrics in self
        self.baseline_metrics = (accuracy, precision, recall, f_score, auc)
        self.baseline_y_pred_loss = test_loss

        self.y_true = y_true

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
        regular_knn_test_loss = []
        siamese_knn_test_loss = []
        # test_loss = []
        test_labels = []
        tqdm_total_bar = self.test_ds.cardinality().numpy()
        for step, (x_batch_test, y_batch_test) in tqdm(enumerate(self.test_ds), total=tqdm_total_bar):
            # x_batch_test: ndarray of shape (batch_size, num_dataset_features)
            # y_batch_test: ndarray of shape (batch_size,)

            reconstruction_loss = self.test_step(x_batch_test).numpy()
            test_labels.append(y_batch_test.numpy())

            # calculate regular knn indices
            regular_knn_batch_neighbors_indices = self.knn_model_regular.kneighbors(X=x_batch_test.numpy(), n_neighbors=num_neighbors, return_distance=False)
            # calculate siamese-knn indices
            test_batch_latent_features = self.siam_internal_model(x_batch_test).numpy()
            siamese_knn_batch_neighbors_indices = self.knn_model_siamese.kneighbors(X=test_batch_latent_features, n_neighbors=num_neighbors, return_distance=False)
            # neighbors_indices: ndarray of shape (batch_size, num_neighbors)

            # batch_neighbors_features: ndarray of shape (batch_size, num_neighbors, num_dataset_features)
            regular_knn_batch_neighbors_features = self.knn_data[regular_knn_batch_neighbors_indices]
            siamese_knn_batch_neighbors_features = self.knn_data[siamese_knn_batch_neighbors_indices]

            # batch_tta_samples: ndarray of shape: (batch_size, num_augmentations, num_dataset_features)
            regular_knn_batch_tta_samples = self.generate_tta_samples(regular_knn_batch_neighbors_features, num_augmentations=num_augmentations)
            siamese_knn_batch_tta_samples = self.generate_tta_samples(siamese_knn_batch_neighbors_features, num_augmentations=num_augmentations)
            
            # batch_tta_reconstruction: ndarray of shape: (batch_size, num_augmentations)
            regular_knn_batch_tta_reconstruction = self.test_step(regular_knn_batch_tta_samples).numpy()
            siamese_knn_batch_tta_reconstruction = self.test_step(siamese_knn_batch_tta_samples).numpy()
            
            for primary_loss, tta_loss in list(zip(reconstruction_loss, regular_knn_batch_tta_reconstruction)):
                combined_tta_loss = np.concatenate([[primary_loss], tta_loss])
                regular_knn_test_loss.append(np.mean(combined_tta_loss))


            for primary_loss, tta_loss in list(zip(reconstruction_loss, siamese_knn_batch_tta_reconstruction)):
                combined_tta_loss = np.concatenate([[primary_loss], tta_loss])
                siamese_knn_test_loss.append(np.mean(combined_tta_loss))

        train_loss = np.concatenate(train_loss, axis=0)

        test_labels = np.concatenate(test_labels, axis=0)

        combined_regular_knn_loss = np.concatenate([train_loss, regular_knn_test_loss], axis=0)
        combined_siamese_knn_loss = np.concatenate([train_loss, siamese_knn_test_loss], axis=0)

        # setting the threshold to be a value of a 80% of the loss of all the examples
        # give a reference from the GMM papper
        regular_knn_thresh = np.percentile(combined_regular_knn_loss, self.percentile)
        siamese_knn_thresh = np.percentile(combined_siamese_knn_loss, self.percentile)
        print("Regular TTA threshold :", regular_knn_thresh)
        print("Siamese TTA threshold :", siamese_knn_thresh)
        # thresh = np.mean(combined_loss) + np.std(combined_loss)

        regular_knn_y_pred = tf.where(regular_knn_test_loss > regular_knn_thresh, 1, 0).numpy().astype(int)
        siamese_knn_y_pred = tf.where(siamese_knn_test_loss > siamese_knn_thresh, 1, 0).numpy().astype(int)
        y_true = np.asarray(test_labels).astype(int)

        from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score, roc_auc_score

        regular_knn_accuracy = accuracy_score(y_true, regular_knn_y_pred)
        siamese_knn_accuracy = accuracy_score(y_true, siamese_knn_y_pred)

        regular_knn_precision, regular_knn_recall, regular_knn_f_score, regular_knn_support = prf(y_true, regular_knn_y_pred, average='binary')
        siamese_knn_precision, siamese_knn_recall, siamese_knn_f_score, siamese_knn_support = prf(y_true, siamese_knn_y_pred, average='binary')

        # regular_knn_auc = roc_auc_score(y_true, regular_knn_y_pred)
        regular_knn_auc = roc_auc_score(y_true, regular_knn_test_loss)
        # siamese_knn_auc = roc_auc_score(y_true, siamese_knn_y_pred)
        siamese_knn_auc = roc_auc_score(y_true, siamese_knn_test_loss)

        # save both test metrics in self
        self.regular_tta_metrics = (regular_knn_accuracy, regular_knn_precision, regular_knn_recall, regular_knn_f_score, regular_knn_auc)
        self.siamese_tta_metrics = (siamese_knn_accuracy, siamese_knn_precision, siamese_knn_recall, siamese_knn_f_score, siamese_knn_auc)

        self.regular_tta_y_pred_loss = regular_knn_test_loss
        self.siamese_tta_y_pred_loss = siamese_knn_test_loss

    @tf.function
    def test_step(self, inputs):
        latent_var = self.encoder(inputs)
        reconstructed = self.decoder(latent_var)
        reconstruction_loss = self.loss_func(inputs, reconstructed)

        return reconstruction_loss

    def generate_tta_samples(self, batch_neighbors_features, num_augmentations):
        batch_tta_samples = []
        for neighbors_features in batch_neighbors_features:
            kmeans_model = cuKMeans(n_clusters=num_augmentations, random_state=1234)
            neighbors_features = neighbors_features.astype(np.float32)
            kmeans_model.fit(X=neighbors_features)
            tta_samples = kmeans_model.cluster_centers_
            # appending to the batch tta samples
            batch_tta_samples.append(tta_samples)
        
        return np.array(batch_tta_samples)

    def print_test_results(self):
        if self.baseline_metrics is None or self.regular_tta_metrics is None or self.siamese_tta_metrics is None:
            raise ValueError("run test and tta_test functions and then call print_test_results function")
        
        print("*"*100)
        print("---- Baseline Test ----")
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, AUC : {:0.4f}".format(*self.baseline_metrics))

        print("*"*100)
        print("---- Regular TTA Test ----")
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, AUC : {:0.4f}".format(*self.regular_tta_metrics))

        print("*"*100)
        print("---- Siamese TTA Test ----")
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, AUC : {:0.4f}".format(*self.siamese_tta_metrics))
        print("*"*100)

        # plot roc curve
        # skplt.metrics.plot_roc_curve(self.y_true, self.baseline_y_pred)
        baseline_accuracy, baseline_precision, baseline_recall, baseline_f_score, baseline_auc = self.baseline_metrics 
        baseline_fpr, baseline_tpr, threshold = roc_curve(self.y_true, self.baseline_y_pred_loss)
        plt.plot(baseline_fpr, baseline_tpr, label='AUC = {:.3f}'.format(baseline_auc))
        regular_knn_accuracy, regular_knn_precision, regular_knn_recall, regular_knn_f_score, regular_knn_auc = self.regular_tta_metrics
        regular_tta_fpr, regular_tta_tpr, threshold = roc_curve(self.y_true, self.regular_tta_y_pred_loss)
        plt.plot(regular_tta_fpr, regular_tta_tpr, label='AUC = {:.3f}'.format(regular_knn_auc))
        siamese_knn_accuracy, siamese_knn_precision, siamese_knn_recall, siamese_knn_f_score, siamese_knn_auc = self.siamese_tta_metrics 
        siamese_tta_fpr, siamese_tta_tpr, threshold = roc_curve(self.y_true, self.siamese_tta_y_pred_loss)
        plt.plot(siamese_tta_fpr, siamese_tta_tpr, label='AUC = {:.3f}'.format(siamese_knn_auc))
        plt.plot([0, 1], [0, 1], linestyle='dashed', color='gray')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.title("ROC Curves")
        # plt.show()
        plt.savefig(f"/home/nivgold/plots/{self.dataset_name}_ROC_curve.png")