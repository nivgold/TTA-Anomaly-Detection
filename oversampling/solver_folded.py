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

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE


class Solver:
    def __init__(self, X, y, dataset_name, epochs=32, features_dim=76, n_folds=5, siamese_data=None, with_cuml=True):
        self.dataset_name = dataset_name

        self.num_epochs = epochs
        self.features_dim = features_dim
        self.n_folds = n_folds

        self.with_cuml = with_cuml

        # set percentile thresh
        self.percentile = 80

        # show network architecrute
        # setup tensorboard

        # if knn_data is not None and siamese_data is not None:
        self.train_siamese_model(siamese_data)
        self.train_knn_model(X) 

        self.make_folded_dataset(X, y)

    
    def train_knn_model(self, knn_data):
        self.X = knn_data

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
            latent_features = self.siam_internal_model(self.X).numpy()
            self.knn_model_siamese.fit(latent_features)
            # define and train regular KNN
            self.knn_model_regular = cuNearestNeighbors()
            self.knn_model_regular.fit(self.X)
        else:
            print("--- Not using cuML")
            # define and train siamese-distance based KNN
            self.knn_model_siamese = NearestNeighbors(metric=siamese_distance, n_jobs=-1)
            self.knn_model_siamese.fit(self.X)
            # define and train regular KNN
            self.knn_model_regular = NearestNeighbors()
            self.knn_model_regular.fit(self.X)

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
            verbose=0
        )

        # save the latent features of every sample in a dict
        self.siam_internal_model = self.siamese_network.internal_model

    def make_folded_dataset(self, X, y):
        from sklearn.model_selection import StratifiedKFold
        from sklearn.model_selection import KFold
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        def gen():
            for step, (train_index, test_index) in enumerate(skf.split(X, y)):
                train_features, train_labels = X[train_index], y[train_index]

                # FILTERING THE ANOMALY SAMPLES FROM THE TRAIN DATA !!!!!!!!!!!!!!!!!!!!!
                train_normal_mask = train_labels == 0
                train_features = train_features[train_normal_mask]
                train_labels = train_labels[train_normal_mask]

                test_features, test_labels = X[test_index], y[test_index]
                yield train_features, train_labels, test_features, test_labels
        
        # creating the folded dataset that contains all the train and test folds
        folded_dataset = tf.data.Dataset.from_generator(gen, (tf.float64, tf.int16, tf.float64, tf.int16))
        
        # adding seperatly the train folds and test folds to lists
        self.folded_train_datasets_list = []
        self.folded_test_datasets_list = []
        for X_train, y_train, X_test, y_test in folded_dataset:
            train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
                .cache()
                .batch(32)
            )
            
            test_ds = (tf.data.Dataset.from_tensor_slices((X_test, y_test))
                        .cache()
                        .batch(32)
            )
            
            # define the train ds and test ds lists
            self.folded_train_datasets_list.append(train_ds)
            self.folded_test_datasets_list.append(test_ds)

        # define the models list
        self.models_list = [(Encoder(input_shape=self.features_dim), Decoder(original_dim=self.features_dim)) for i in range(self.n_folds)]

    def save_weights(self, path, dataset_name):
        encoder_path = path + f"/epochs_{self.num_epochs}_{dataset_name}_encoder_weights"
        decoder_path = path + f"/epochs_{self.num_epochs}_{dataset_name}_decoder_weights"

        np.save(encoder_path, self.encoder.get_weights())
        np.save(decoder_path, self.decoder.get_weights())
    
    def save_weights_folded(self, path):
        for fold_index in range(self.n_folds):
            encoder_path = path + f"/epochs_{self.num_epochs}_{self.dataset_name}_fold_{fold_index}_encoder_weights"
            decoder_path = path + f"/epochs_{self.num_epochs}_{self.dataset_name}_fold_{fold_index}_decoder_weights"

            np.save(encoder_path, self.models_list[fold_index][0].get_weights())
            np.save(decoder_path, self.models_list[fold_index][1].get_weights())

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
    
    def load_weights_folded(self, path):
        self.models_list = [(Encoder(input_shape=self.features_dim), Decoder(original_dim=self.features_dim)) for i in range(self.n_folds)]
        
        for fold_index in range(self.n_folds):
            fold_encoder_path = path + f"/epochs_{self.num_epochs}_{self.dataset_name}_fold_{fold_index}_encoder_weights.npy"
            fold_decoder_path = path + f"/epochs_{self.num_epochs}_{self.dataset_name}_fold_{fold_index}_decoder_weights.npy"
            
            encoder_trained_weights = np.load(fold_encoder_path, allow_pickle=True)
            decoder_trained_weights = np.load(fold_decoder_path, allow_pickle=True)

            for batch_X_train, batch_y_train in self.folded_train_datasets_list[fold_index]:
                latent_var = self.models_list[fold_index][0](batch_X_train)
                self.models_list[fold_index][1](latent_var)
                break
                
            self.models_list[fold_index][0].set_weights(encoder_trained_weights)
            self.models_list[fold_index][1].set_weights(decoder_trained_weights)


    def train_folded(self):
        for fold_index in range(self.n_folds):
            # train set-up
            train_ds = self.folded_train_datasets_list[fold_index]
            encoder, decoder = self.models_list[fold_index]
            train_step_func = Solver.train_step()
            print(f"--- Training K-Fold Split index: {fold_index+1}")
            self._train(train_ds, encoder, decoder, train_step_func)


    def _train(self, train_ds, encoder, decoder, train_step_func):

        # loss function
        loss_func = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam()
        for epoch in range(1, self.num_epochs+1):
            epoch_loss_mean = tf.keras.metrics.Mean()

            for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
                # if step%1000 == 0: print(f'Step: {step} in epoch {epoch}')
                # model_train_step = Solver.train_step()
                loss = train_step_func(x_batch_train, encoder, decoder, optimizer, loss_func)

                # keep track of the metrics
                epoch_loss_mean.update_state(loss)

            # update metrics after each epoch
            if epoch==1 or epoch%100 == 0:
                print(f'Epoch {epoch} loss mean: {epoch_loss_mean.result()}')

    @staticmethod
    def train_step():
        @tf.function
        def train_one_step(inputs, encoder, decoder, optimizer, loss_func):
            with tf.GradientTape() as tape:
                latent_var = encoder(inputs)
                outputs = decoder(latent_var)
                loss = loss_func(inputs, outputs)

                trainable_vars = encoder.trainable_variables \
                                + decoder.trainable_variables

            grads = tape.gradient(loss, trainable_vars)
            optimizer.apply_gradients(zip(grads, trainable_vars))

            return loss
        return train_one_step

    def test_folded(self):
        self.baseline_metrics_folds_list = []
        self.baseline_y_pred_loss_folds_list = []
        self.y_true_folds_list = []

        for fold_index in range(self.n_folds):
            # test set-up
            train_ds = self.folded_train_datasets_list[fold_index]
            test_ds = self.folded_test_datasets_list[fold_index]
            trained_encoder, trained_decoder = self.models_list[fold_index]
            test_step_func = Solver.test_step()
            print(f"--- Testing K-Fold Split index: {fold_index+1}")
            self._test(train_ds, test_ds, trained_encoder, trained_decoder, test_step_func)
        
        self.baseline_metrics_folds_list = np.array(self.baseline_metrics_folds_list)

    def _test(self, train_ds, test_ds, trained_encoder, trained_decoder, test_step_func):
        # loss function - with reduction equals to `NONE` in order to get the loss of every test example
        loss_func = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        # iterating through the train to calculate the loss of every train instance
        train_loss = []
        for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
            loss = test_step_func(x_batch_train, trained_encoder, trained_decoder, loss_func)
            train_loss.append(loss.numpy())

        # iterating through the test to calculate the loss of every test instance
        test_loss = []
        test_labels = []
        for step, (x_batch_test, y_batch_test) in enumerate(test_ds):
            reconstruction_loss = test_step_func(x_batch_test, trained_encoder, trained_decoder, loss_func)
            test_loss.append(reconstruction_loss.numpy())
            test_labels.append(y_batch_test.numpy())

        train_loss = np.concatenate(train_loss, axis=0)

        test_loss = np.concatenate(test_loss, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        combined_loss = np.concatenate([train_loss, test_loss], axis=0)

        # setting the threshold to be a value of a 80% of the loss of all the examples
        # give a reference from the GMM papper
        thresh = np.percentile(combined_loss, self.percentile)
        # print("Baseline threshold :", thresh)
        # thresh = np.mean(combined_loss) + np.std(combined_loss)

        y_pred = tf.where(test_loss > thresh, 1, 0).numpy().astype(int)
        y_true = np.asarray(test_labels).astype(int)

        from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score, roc_auc_score
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f_score, support = prf(y_true, y_pred, average='binary')
        # auc = roc_auc_score(y_true, y_pred)
        auc = roc_auc_score(y_true, test_loss)

        # save test metrics in self
        self.baseline_metrics_folds_list.append((accuracy, precision, recall, f_score, auc))
        self.baseline_y_pred_loss_folds_list.append(test_loss)

        self.y_true_folds_list.append(y_true)

    def test_tta_folded(self, num_neighbors, num_augmentations):
        self.kmeans_regular_knn_tta_metrics_folds_list = []
        self.kmeans_siamese_knn_tta_metrics_folds_list = []
        self.SMOTE_regular_knn_tta_metrics_folds_list = []
        self.SMOTE_siamese_knn_tta_metrics_folds_list = []
        self.BorderlineSMOTE_regular_knn_tta_metrics_folds_list = []
        self.BorderlineSMOTE_siamese_knn_tta_metrics_folds_list = []
        self.gaussian_tta_metrics_folds_list = []

        self.kmeans_regular_knn_tta_y_pred_loss_folds_list = []
        self.kmeans_siamese_knn_tta_y_pred_loss_folds_list = []
        self.SMOTE_regular_knn_tta_y_pred_loss_folds_list = []
        self.SMOTE_siamese_knn_tta_y_pred_loss_folds_list = []
        self.BorderlineSMOTE_regular_knn_tta_y_pred_loss_folds_list = []
        self.BorderlineSMOTE_siamese_knn_tta_y_pred_loss_folds_list = []
        self.gaussian_tta_y_pred_loss_folds_list = []

        for fold_index in range(self.n_folds):
            # tta test set-up
            train_ds = self.folded_train_datasets_list[fold_index]
            test_ds = self.folded_test_datasets_list[fold_index]
            trained_encoder, trained_decoder = self.models_list[fold_index][0], self.models_list[fold_index][1]
            test_step_func = Solver.test_step()
            print(f"--- TTA Testing Fold index: {fold_index+1}")
            self._test_tta(train_ds, test_ds, trained_encoder, trained_decoder, num_neighbors, num_augmentations, test_step_func)

        self.kmeans_regular_knn_tta_metrics_folds_list = np.array(self.kmeans_regular_knn_tta_metrics_folds_list)
        self.kmeans_siamese_knn_tta_metrics_folds_list = np.array(self.kmeans_siamese_knn_tta_metrics_folds_list)
        self.SMOTE_regular_knn_tta_metrics_folds_list = np.array(self.SMOTE_regular_knn_tta_metrics_folds_list)
        self.SMOTE_siamese_knn_tta_metrics_folds_list = np.array(self.SMOTE_siamese_knn_tta_metrics_folds_list)
        self.BorderlineSMOTE_regular_knn_tta_metrics_folds_list = np.array(self.BorderlineSMOTE_regular_knn_tta_metrics_folds_list)
        self.BorderlineSMOTE_siamese_knn_tta_metrics_folds_list = np.array(self.BorderlineSMOTE_siamese_knn_tta_metrics_folds_list)
        self.gaussian_tta_metrics_folds_list = np.array(self.gaussian_tta_metrics_folds_list)

    def _test_tta(self, train_ds, test_ds, trained_encoder, trained_decoder, num_neighbors, num_augmentations, test_step_func):

        # loss function - with reduction equals to `NONE` in order to get the loss of every test example
        loss_func = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        # iterating through the train to calculate the loss of every train instance
        train_loss = []
        for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
            loss = test_step_func(x_batch_train, trained_encoder, trained_decoder, loss_func)
            train_loss.append(loss.numpy())

        # iterating through the test to calculate the loss of every test instance
        kmeans_regular_knn_test_loss = []
        kmeans_siamese_knn_test_loss = []
        SMOTE_regular_knn_tta_test_loss = []
        SMOTE_siamese_knn_tta_test_loss = []
        BorderlineSMOTE_regular_knn_tta_test_loss = []
        BorderlineSMOTE_siamese_knn_tta_test_loss = []
        gaussian_tta_test_loss = []
        
        test_labels = []
        tqdm_total_bar = test_ds.cardinality().numpy()
        for step, (x_batch_test, y_batch_test) in tqdm(enumerate(test_ds), total=tqdm_total_bar):
            # x_batch_test: ndarray of shape (batch_size, num_dataset_features)
            # y_batch_test: ndarray of shape (batch_size,)

            reconstruction_loss = test_step_func(x_batch_test, trained_encoder, trained_decoder, loss_func).numpy()
            test_labels.append(y_batch_test.numpy())

            # calculate regular knn indices
            regular_knn_batch_neighbors_indices = self.knn_model_regular.kneighbors(X=x_batch_test.numpy(), n_neighbors=num_neighbors, return_distance=False)
            # calculate siamese-knn indices
            test_batch_latent_features = self.siam_internal_model(x_batch_test).numpy()
            siamese_knn_batch_neighbors_indices = self.knn_model_siamese.kneighbors(X=test_batch_latent_features, n_neighbors=num_neighbors, return_distance=False)
            # neighbors_indices: ndarray of shape (batch_size, num_neighbors)

            # batch_neighbors_features: ndarray of shape (batch_size, num_neighbors, num_dataset_features)
            regular_knn_batch_neighbors_features = self.X[regular_knn_batch_neighbors_indices]
            siamese_knn_batch_neighbors_features = self.X[siamese_knn_batch_neighbors_indices]

            # batch_tta_samples: ndarray of shape: (batch_size, num_augmentations, num_dataset_features)
            kmeans_regular_knn_batch_tta_samples = self.generate_kmeans_tta_samples(regular_knn_batch_neighbors_features, num_augmentations=num_augmentations)
            kmeans_siamese_knn_batch_tta_samples = self.generate_kmeans_tta_samples(siamese_knn_batch_neighbors_features, num_augmentations=num_augmentations)
            SMOTE_regular_knn_batch_tta_samples = self.generate_oversampling_tta_samples(regular_knn_batch_neighbors_features, oversampling_method=SMOTE, num_neighbors=num_neighbors, num_augmentations=num_augmentations)
            SMOTE_siamese_knn_batch_tta_samples = self.generate_oversampling_tta_samples(siamese_knn_batch_neighbors_features, oversampling_method=SMOTE, num_neighbors=num_neighbors, num_augmentations=num_augmentations)
            BorderlineSMOTE_regular_knn_batch_tta_samples = self.generate_oversampling_tta_samples(regular_knn_batch_neighbors_features, oversampling_method=BorderlineSMOTE, num_neighbors=num_neighbors, num_augmentations=num_augmentations)
            BorderlineSMOTE_siamese_knn_batch_tta_samples = self.generate_oversampling_tta_samples(siamese_knn_batch_neighbors_features, oversampling_method=BorderlineSMOTE, num_neighbors=num_neighbors, num_augmentations=num_augmentations)
            gaussian_batch_tta_samples = self.generate_random_noise_tta_samples(x_batch_test.numpy(), num_augmentations=num_augmentations)
            
            # batch_tta_reconstruction: ndarray of shape: (batch_size, num_augmentations)
            kmeans_regular_knn_batch_tta_reconstruction = test_step_func(kmeans_regular_knn_batch_tta_samples, trained_encoder, trained_decoder, loss_func).numpy()
            kmeans_siamese_knn_batch_tta_reconstruction = test_step_func(kmeans_siamese_knn_batch_tta_samples, trained_encoder, trained_decoder, loss_func).numpy()
            SMOTE_regular_knn_batch_tta_reconstruction = test_step_func(SMOTE_regular_knn_batch_tta_samples, trained_encoder, trained_decoder, loss_func).numpy()
            SMOTE_siamese_knn_batch_tta_reconstruction = test_step_func(SMOTE_siamese_knn_batch_tta_samples, trained_encoder, trained_decoder, loss_func).numpy()
            BorderlineSMOTE_regular_knn_batch_tta_reconstruction = test_step_func(BorderlineSMOTE_regular_knn_batch_tta_samples, trained_encoder, trained_decoder, loss_func).numpy()
            BorderlineSMOTE_siamese_knn_batch_tta_reconstruction = test_step_func(BorderlineSMOTE_siamese_knn_batch_tta_samples, trained_encoder, trained_decoder, loss_func).numpy()
            gaussian_batch_tta_reconstruction = test_step_func(gaussian_batch_tta_samples, trained_encoder, trained_decoder, loss_func).numpy()
            
            # combine original test samples' reconstruction with the kmeans regular-NN TTA samples' reconstruction
            for primary_loss, tta_loss in list(zip(reconstruction_loss, kmeans_regular_knn_batch_tta_reconstruction)):
                combined_tta_loss = np.concatenate([[primary_loss], tta_loss])
                kmeans_regular_knn_test_loss.append(np.mean(combined_tta_loss))

            # combine original test samples' reconstruction with the kmeans siamese-NN TTA samples' reconstruction
            for primary_loss, tta_loss in list(zip(reconstruction_loss, kmeans_siamese_knn_batch_tta_reconstruction)):
                combined_tta_loss = np.concatenate([[primary_loss], tta_loss])
                kmeans_siamese_knn_test_loss.append(np.mean(combined_tta_loss))

            # combine original test samples' reconstruction with the SMOTE regular-NN TTA samples' reconstruction
            for primary_loss, tta_loss in list(zip(reconstruction_loss, SMOTE_regular_knn_batch_tta_reconstruction)):
                combined_tta_loss = np.concatenate([[primary_loss], tta_loss])
                SMOTE_regular_knn_tta_test_loss.append(np.mean(combined_tta_loss))

            # combine original test samples's reconstruction with the SMOTE siamese-NN TTA samples' reconstruction
            for primary_loss, tta_loss in list(zip(reconstruction_loss, SMOTE_siamese_knn_batch_tta_reconstruction)):
                combined_tta_loss = np.concatenate([[primary_loss], tta_loss])
                SMOTE_siamese_knn_tta_test_loss.append(np.mean(combined_tta_loss))
                
            # combine original test samples' reconstruction with the BorderlineSMOTE regular-NN TTA samples' reconstruction
            for primary_loss, tta_loss in list(zip(reconstruction_loss, BorderlineSMOTE_regular_knn_batch_tta_reconstruction)):
                combined_tta_loss = np.concatenate([[primary_loss], tta_loss])
                BorderlineSMOTE_regular_knn_tta_test_loss.append(np.mean(combined_tta_loss))
            
            # combine original test samples' reconstruction with the BorderlineSMOTE siamese-NN TTA samples' reconstruction
            for primary_loss, tta_loss in list(zip(reconstruction_loss, BorderlineSMOTE_siamese_knn_batch_tta_reconstruction)):
                combined_tta_loss = np.concatenate([[primary_loss], tta_loss])
                BorderlineSMOTE_siamese_knn_tta_test_loss.append(np.mean(combined_tta_loss))
            
            # combine original test sampels' reconstuction with the gaussian TTA samples' reconstruction
            for primary_loss, tta_loss, in list(zip(reconstruction_loss, gaussian_batch_tta_reconstruction)):
                combined_tta_loss = np.concatenate([[primary_loss], tta_loss])
                gaussian_tta_test_loss.append(np.mean(combined_tta_loss))

        # flatten the train_loss and the test_labels vectors
        train_loss = np.concatenate(train_loss, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        # combine the train_loss with each of every tta method
        combined_kmeans_regular_knn_loss = np.concatenate([train_loss, kmeans_regular_knn_test_loss], axis=0)
        combined_kmeans_siamese_knn_loss = np.concatenate([train_loss, kmeans_siamese_knn_test_loss], axis=0)
        combined_SMOTE_regular_knn_tta_loss = np.concatenate([train_loss, SMOTE_regular_knn_tta_test_loss], axis=0)
        combined_SMOTE_siamese_knn_tta_loss = np.concatenate([train_loss, SMOTE_siamese_knn_tta_test_loss], axis=0)
        combined_BorderlineSMOTE_regular_knn_tta_loss = np.concatenate([train_loss, BorderlineSMOTE_regular_knn_tta_test_loss], axis=0)
        combined_BorderlineSMOTE_siamese_knn_tta_loss = np.concatenate([train_loss, BorderlineSMOTE_siamese_knn_tta_test_loss], axis=0)
        combined_gaussian_tta_loss = np.concatenate([train_loss, gaussian_tta_test_loss], axis=0)

        # setting the threshold to be a value of a 80% of the loss of all the examples
        # give a reference from the GMM papper
        kmeans_regular_knn_thresh = np.percentile(combined_kmeans_regular_knn_loss, self.percentile)
        kmeans_siamese_knn_thresh = np.percentile(combined_kmeans_siamese_knn_loss, self.percentile)
        SMOTE_regular_knn_tta_thresh = np.percentile(combined_SMOTE_regular_knn_tta_loss, self.percentile)
        SMOTE_siamese_knn_tta_thresh = np.percentile(combined_SMOTE_siamese_knn_tta_loss, self.percentile)
        BorderlineSMOTE_regular_knn_tta_thresh = np.percentile(combined_BorderlineSMOTE_regular_knn_tta_loss, self.percentile)
        BorderlineSMOTE_siamese_knn_tta_thresh = np.percentile(combined_BorderlineSMOTE_siamese_knn_tta_loss, self.percentile)
        gaussian_tta_thresh = np.percentile(combined_gaussian_tta_loss, self.percentile)

        # getting the 0,1 vector of each of every tta method by applying the threshold
        kmeans_regular_knn_y_pred = tf.where(kmeans_regular_knn_test_loss > kmeans_regular_knn_thresh, 1, 0).numpy().astype(int)
        kmeans_siamese_knn_y_pred = tf.where(kmeans_siamese_knn_test_loss > kmeans_siamese_knn_thresh, 1, 0).numpy().astype(int)
        SMOTE_regular_knn_tta_y_pred = tf.where(SMOTE_regular_knn_tta_test_loss > SMOTE_regular_knn_tta_thresh, 1, 0).numpy().astype(int)
        SMOTE_siamese_knn_tta_y_pred = tf.where(SMOTE_siamese_knn_tta_test_loss > SMOTE_siamese_knn_tta_thresh, 1, 0).numpy().astype(int)
        BorderlineSMOTE_regular_knn_tta_y_pred = tf.where(BorderlineSMOTE_regular_knn_tta_test_loss > BorderlineSMOTE_regular_knn_tta_thresh, 1, 0).numpy().astype(int)
        BorderlineSMOTE_siamese_knn_tta_y_pred = tf.where(BorderlineSMOTE_siamese_knn_tta_test_loss > BorderlineSMOTE_siamese_knn_tta_thresh, 1, 0).numpy().astype(int)
        gaussian_tta_y_pred = tf.where(gaussian_tta_test_loss > gaussian_tta_thresh, 1, 0).numpy().astype(int)

        y_true = np.asarray(test_labels).astype(int)

        from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score, roc_auc_score

        # calculating accuracy
        kmeans_regular_knn_accuracy = accuracy_score(y_true, kmeans_regular_knn_y_pred)
        kmeans_siamese_knn_accuracy = accuracy_score(y_true, kmeans_siamese_knn_y_pred)
        SMOTE_regular_knn_tta_accuracy = accuracy_score(y_true, SMOTE_regular_knn_tta_y_pred)
        SMOTE_siamese_knn_tta_accuracy = accuracy_score(y_true, SMOTE_siamese_knn_tta_y_pred)
        BorderlineSMOTE_regular_knn_tta_accuracy = accuracy_score(y_true, BorderlineSMOTE_regular_knn_tta_y_pred)
        BorderlineSMOTE_siamese_knn_tta_accuracy = accuracy_score(y_true, BorderlineSMOTE_siamese_knn_tta_y_pred)
        guassian_tta_accuracy = accuracy_score(y_true, gaussian_tta_y_pred)

        # calculating precision, recall and f1 scores
        kmeans_regular_knn_precision, kmeans_regular_knn_recall, kmeans_regular_knn_f_score, kmeans_regular_knn_support = prf(y_true, kmeans_regular_knn_y_pred, average='binary')
        kmeans_siamese_knn_precision, kmeans_siamese_knn_recall, kmeans_siamese_knn_f_score, kmeans_siamese_knn_support = prf(y_true, kmeans_siamese_knn_y_pred, average='binary')
        SMOTE_regular_knn_tta_precision, SMOTE_regular_knn_tta_recall, SMOTE_regular_knn_tta_f_score, SMOTE_regular_knn_tta_support = prf(y_true, SMOTE_regular_knn_tta_y_pred, average='binary')
        SMOTE_siamese_knn_tta_precision, SMOTE_siamese_knn_tta_recall, SMOTE_siamese_knn_tta_f_score, SMOTE_siamese_knn_tta_support = prf(y_true, SMOTE_siamese_knn_tta_y_pred, average='binary')
        BorderlineSMOTE_regular_knn_tta_precision, BorderlineSMOTE_regular_knn_tta_recall, BorderlineSMOTE_regular_knn_tta_f_score, BorderlineSMOTE_regular_knn_tta_support = prf(y_true, BorderlineSMOTE_regular_knn_tta_y_pred, average='binary')
        BorderlineSMOTE_siamese_knn_tta_precision, BorderlineSMOTE_siamese_knn_tta_recall, BorderlineSMOTE_siamese_knn_tta_f_score, BorderlineSMTOE_siamese_knn_tta_support = prf(y_true, BorderlineSMOTE_siamese_knn_tta_y_pred, average='binary')
        gaussian_tta_precision, gaussian_tta_recall, gaussian_tta_f_score, gaussian_tta_suppoer = prf(y_true, gaussian_tta_y_pred, average='binary')

        # calculating AUC
        kmeans_regular_knn_auc = roc_auc_score(y_true, kmeans_regular_knn_test_loss)
        kmeans_siamese_knn_auc = roc_auc_score(y_true, kmeans_siamese_knn_test_loss)
        SMOTE_regular_knn_tta_auc = roc_auc_score(y_true, SMOTE_regular_knn_tta_test_loss)
        SMOTE_siamese_knn_tta_auc = roc_auc_score(y_true, SMOTE_siamese_knn_tta_test_loss)
        BorderlineSMOTE_regular_knn_tta_auc = roc_auc_score(y_true, BorderlineSMOTE_regular_knn_tta_test_loss)
        BorderlineSMOTE_siamese_knn_tta_auc = roc_auc_score(y_true, BorderlineSMOTE_siamese_knn_tta_test_loss)
        gaussian_tta_auc = roc_auc_score(y_true, gaussian_tta_test_loss)

        # save both test metrics and y_pred_loss vectors in self
        self.kmeans_regular_knn_tta_metrics_folds_list.append((kmeans_regular_knn_accuracy, kmeans_regular_knn_precision, kmeans_regular_knn_recall, kmeans_regular_knn_f_score, kmeans_regular_knn_auc))
        self.kmeans_siamese_knn_tta_metrics_folds_list.append((kmeans_siamese_knn_accuracy, kmeans_siamese_knn_precision, kmeans_siamese_knn_recall, kmeans_siamese_knn_f_score, kmeans_siamese_knn_auc))
        self.SMOTE_regular_knn_tta_metrics_folds_list.append((SMOTE_regular_knn_tta_accuracy, SMOTE_regular_knn_tta_precision, SMOTE_regular_knn_tta_recall, SMOTE_regular_knn_tta_f_score, SMOTE_regular_knn_tta_auc))
        self.SMOTE_siamese_knn_tta_metrics_folds_list.append((SMOTE_siamese_knn_tta_accuracy, SMOTE_siamese_knn_tta_precision, SMOTE_siamese_knn_tta_recall, SMOTE_siamese_knn_tta_f_score, SMOTE_siamese_knn_tta_auc))
        self.BorderlineSMOTE_regular_knn_tta_metrics_folds_list.append((BorderlineSMOTE_regular_knn_tta_accuracy, BorderlineSMOTE_regular_knn_tta_precision, BorderlineSMOTE_regular_knn_tta_recall, BorderlineSMOTE_regular_knn_tta_f_score, BorderlineSMOTE_regular_knn_tta_auc))
        self.BorderlineSMOTE_siamese_knn_tta_metrics_folds_list.append((BorderlineSMOTE_siamese_knn_tta_accuracy, BorderlineSMOTE_siamese_knn_tta_precision, BorderlineSMOTE_siamese_knn_tta_recall, BorderlineSMOTE_siamese_knn_tta_f_score, BorderlineSMOTE_siamese_knn_tta_auc))
        self.gaussian_tta_metrics_folds_list.append((guassian_tta_accuracy, gaussian_tta_precision, gaussian_tta_recall, gaussian_tta_f_score, gaussian_tta_auc))

        self.kmeans_regular_knn_tta_y_pred_loss_folds_list.append(kmeans_regular_knn_test_loss)
        self.kmeans_siamese_knn_tta_y_pred_loss_folds_list.append(kmeans_siamese_knn_test_loss)
        self.SMOTE_regular_knn_tta_y_pred_loss_folds_list.append(SMOTE_regular_knn_tta_test_loss)
        self.SMOTE_siamese_knn_tta_y_pred_loss_folds_list.append(SMOTE_siamese_knn_tta_test_loss)
        self.BorderlineSMOTE_regular_knn_tta_y_pred_loss_folds_list.append(BorderlineSMOTE_regular_knn_tta_test_loss)
        self.BorderlineSMOTE_siamese_knn_tta_y_pred_loss_folds_list.append(BorderlineSMOTE_siamese_knn_tta_test_loss)
        self.gaussian_tta_y_pred_loss_folds_list.append(gaussian_tta_test_loss)

    @staticmethod
    def test_step():
        # @tf.function
        def test_one_step(inputs, encoder, decoder, loss_func):
            latent_var = encoder(inputs)
            reconstructed = decoder(latent_var)
            reconstruction_loss = loss_func(inputs, reconstructed)

            return reconstruction_loss
        return test_one_step

    def generate_kmeans_tta_samples(self, batch_neighbors_features, num_augmentations):
        batch_tta_samples = []
        for neighbors_features in batch_neighbors_features:
            kmeans_model = cuKMeans(n_clusters=num_augmentations, random_state=1234)
            neighbors_features = neighbors_features.astype(np.float32)
            kmeans_model.fit(X=neighbors_features)
            tta_samples = kmeans_model.cluster_centers_
            # appending to the batch tta samples
            batch_tta_samples.append(tta_samples)
        
        return np.array(batch_tta_samples)

    def generate_oversampling_tta_samples(self, oversampling_batch_neighbors_features, num_neighbors, num_augmentations, oversampling_method):
        oversampling_batch_tta_samples = np.zeros((32, num_augmentations, self.features_dim))
        for index_in_batch, original_neighbors_features in enumerate(oversampling_batch_neighbors_features):
            original_neighbors_labels = np.zeros((original_neighbors_features.shape[0],))
            
            # create fake samples for the imblearn dataset
            fake_neighbors_features = np.zeros((num_neighbors + num_augmentations, self.features_dim))
            fake_neighbors_labels = np.ones((fake_neighbors_features.shape[0],))

            # create the imblearn dataset
            imblearn_features = np.concatenate([original_neighbors_features, fake_neighbors_features])
            imblearn_labels = np.concatenate([original_neighbors_labels, fake_neighbors_labels])

            oversampling_obj = oversampling_method(k_neighbors=num_neighbors-1, random_state=42)
            X_res, y_res = oversampling_obj.fit_resample(imblearn_features, imblearn_labels)

            current_augmentations = X_res[-num_augmentations:]
            oversampling_batch_tta_samples[index_in_batch] = current_augmentations
        
        return oversampling_batch_tta_samples

    def generate_random_noise_tta_samples(self, x_batch_test, num_augmentations):
        size = x_batch_test.shape
        # scale = 0.2
        random_noise = np.random.normal(size=size)
        # adding the noise to the original batch test samples. expanding the middle dim of x_batch_test to make it (batch_size, 1, dataset_features_dim)
        gaussian_tta_samples = np.expand_dims(x_batch_test, axis=1) + random_noise

        return gaussian_tta_samples

    def print_test_results(self):
        if self.baseline_metrics_folds_list is None or self.kmeans_regular_knn_tta_metrics_folds_list is None or self.kmeans_siamese_knn_tta_metrics_folds_list is None or self.SMOTE_regular_knn_tta_metrics_folds_list is None or self.SMOTE_siamese_knn_tta_metrics_folds_list is None or self.BorderlineSMOTE_regular_knn_tta_metrics_folds_list is None or self.BorderlineSMOTE_siamese_knn_tta_metrics_folds_list is None or self.gaussian_tta_metrics_folds_list is None:
            raise ValueError("run test and tta_test functions and then call print_test_results function")
        
        print("*"*100)
        print("---- Baseline Test ----")
        print_list = []
        for i in range(len(self.baseline_metrics_folds_list.mean(axis=0))):
            print_list.append(self.baseline_metrics_folds_list.mean(axis=0)[i])
            print_list.append(self.baseline_metrics_folds_list.std(axis=0)[i])
        print("Accuracy : {:0.4f}+-{:0.4f}, Precision : {:0.4f}+-{:0.4f}, Recall : {:0.4f}+-{:0.4f}, F-score : {:0.4f}+-{:0.4f}, AUC : {:0.4f}+-{:0.4f}".format(*print_list))

        print("*"*100)
        print("---- Gaussian-Noise TTA Test ----")
        print_list = []
        for i in range(len(self.gaussian_tta_metrics_folds_list.mean(axis=0))):
            print_list.append(self.gaussian_tta_metrics_folds_list.mean(axis=0)[i])
            print_list.append(self.gaussian_tta_metrics_folds_list.std(axis=0)[i])
        print("Accuracy : {:0.4f}+-{:0.4f}, Precision : {:0.4f}+-{:0.4f}, Recall : {:0.4f}+-{:0.4f}, F-score : {:0.4f}+-{:0.4f}, AUC : {:0.4f}+-{:0.4f}".format(*print_list))

        print("*"*100)
        print("---- SMOTE Regular-KNN TTA Test ----")
        print_list = []
        for i in range(len(self.SMOTE_regular_knn_tta_metrics_folds_list.mean(axis=0))):
            print_list.append(self.SMOTE_regular_knn_tta_metrics_folds_list.mean(axis=0)[i])
            print_list.append(self.SMOTE_regular_knn_tta_metrics_folds_list.std(axis=0)[i])
        print("Accuracy : {:0.4f}+-{:0.4f}, Precision : {:0.4f}+-{:0.4f}, Recall : {:0.4f}+-{:0.4f}, F-score : {:0.4f}+-{:0.4f}, AUC : {:0.4f}+-{:0.4f}".format(*print_list))

        print("*"*100)
        print("---- SMOTE Siamese-KNN TTA Test ----")
        print_list = []
        for i in range(len(self.SMOTE_siamese_knn_tta_metrics_folds_list.mean(axis=0))):
            print_list.append(self.SMOTE_siamese_knn_tta_metrics_folds_list.mean(axis=0)[i])
            print_list.append(self.SMOTE_siamese_knn_tta_metrics_folds_list.std(axis=0)[i])
        print("Accuracy : {:0.4f}+-{:0.4f}, Precision : {:0.4f}+-{:0.4f}, Recall : {:0.4f}+-{:0.4f}, F-score : {:0.4f}+-{:0.4f}, AUC : {:0.4f}+-{:0.4f}".format(*print_list))

        print("*"*100)
        print("---- Borderline-SMOTE Regular-KNN TTA Test ----")
        print_list = []
        for i in range(len(self.BorderlineSMOTE_regular_knn_tta_metrics_folds_list.mean(axis=0))):
            print_list.append(self.BorderlineSMOTE_regular_knn_tta_metrics_folds_list.mean(axis=0)[i])
            print_list.append(self.BorderlineSMOTE_regular_knn_tta_metrics_folds_list.std(axis=0)[i])
        print("Accuracy : {:0.4f}+-{:0.4f}, Precision : {:0.4f}+-{:0.4f}, Recall : {:0.4f}+-{:0.4f}, F-score : {:0.4f}+-{:0.4f}, AUC : {:0.4f}+-{:0.4f}".format(*print_list))
        
        print("*"*100)
        print("---- Borderline-SMOTE Siamese-KNN TTA Test ----")
        print_list = []
        for i in range(len(self.BorderlineSMOTE_siamese_knn_tta_metrics_folds_list.mean(axis=0))):
            print_list.append(self.BorderlineSMOTE_siamese_knn_tta_metrics_folds_list.mean(axis=0)[i])
            print_list.append(self.BorderlineSMOTE_siamese_knn_tta_metrics_folds_list.std(axis=0)[i])
        print("Accuracy : {:0.4f}+-{:0.4f}, Precision : {:0.4f}+-{:0.4f}, Recall : {:0.4f}+-{:0.4f}, F-score : {:0.4f}+-{:0.4f}, AUC : {:0.4f}+-{:0.4f}".format(*print_list))

        print("*"*100)
        print("---- k-Means TTA Regular-KNN Test ----")
        print_list = []
        for i in range(len(self.kmeans_regular_knn_tta_metrics_folds_list.mean(axis=0))):
            print_list.append(self.kmeans_regular_knn_tta_metrics_folds_list.mean(axis=0)[i])
            print_list.append(self.kmeans_regular_knn_tta_metrics_folds_list.std(axis=0)[i])
        print("Accuracy : {:0.4f}+-{:0.4f}, Precision : {:0.4f}+-{:0.4f}, Recall : {:0.4f}+-{:0.4f}, F-score : {:0.4f}+-{:0.4f}, AUC : {:0.4f}+-{:0.4f}".format(*print_list))

        print("*"*100)
        print("---- k-Means TTA Siamese-KNN Test ----")
        print_list = []
        for i in range(len(self.kmeans_siamese_knn_tta_metrics_folds_list.mean(axis=0))):
            print_list.append(self.kmeans_siamese_knn_tta_metrics_folds_list.mean(axis=0)[i])
            print_list.append(self.kmeans_siamese_knn_tta_metrics_folds_list.std(axis=0)[i])
        print("Accuracy : {:0.4f}+-{:0.4f}, Precision : {:0.4f}+-{:0.4f}, Recall : {:0.4f}+-{:0.4f}, F-score : {:0.4f}+-{:0.4f}, AUC : {:0.4f}+-{:0.4f}".format(*print_list))
        print("*"*100)

        # # plot roc curve
        # # skplt.metrics.plot_roc_curve(self.y_true, self.baseline_y_pred)
        # baseline_auc = self.baseline_metrics_folds_list[0][-1]
        # baseline_fpr, baseline_tpr, threshold = roc_curve(self.y_true_folds_list[0], self.baseline_y_pred_loss_folds_list[0])
        # plt.plot(baseline_fpr, baseline_tpr, label='No TTA AUC = {:.3f}'.format(baseline_auc))

        # kmeans_regular_knn_tta_auc = self.kmeans_regular_knn_tta_metrics_folds_list[0][-1]
        # kmeans_regular_knn_tta_fpr, kmeans_regular_knn_tta_tpr, threshold = roc_curve(self.y_true_folds_list[0], self.kmeans_regular_knn_tta_y_pred_loss_folds_list[0])
        # plt.plot(kmeans_regular_knn_tta_fpr, kmeans_regular_knn_tta_tpr, label='k-Means Regular-NN TTA AUC = {:.3f}'.format(kmeans_regular_knn_tta_auc))

        # kmeans_siamese_knn_tta_auc = self.kmeans_siamese_knn_tta_metrics_folds_list[0][-1] 
        # kmeans_siamese_knn_tta_fpr, kmeans_siamese_knn_tta_tpr, threshold = roc_curve(self.y_true_folds_list[0], self.kmeans_siamese_knn_tta_y_pred_loss_folds_list[0])
        # plt.plot(kmeans_siamese_knn_tta_fpr, kmeans_siamese_knn_tta_tpr, label='k-Means Siamese-NN TTA AUC = {:.3f}'.format(kmeans_siamese_knn_tta_auc))
        
        # SMOTE_regular_knn_tta_auc = self.SMOTE_regular_knn_tta_metrics_folds_list[0][-1]
        # SMOTE_regular_knn_tta_fpr, SMOTE_regular_knn_tta_tpr, threshold = roc_curve(self.y_true_folds_list[0], self.SMOTE_regular_knn_tta_y_pred_loss_folds_list[0])
        # plt.plot(SMOTE_regular_knn_tta_fpr, SMOTE_regular_knn_tta_tpr, label='SMOTE Regular-NN TTA AUC = {:.3f}'.format(SMOTE_regular_knn_tta_auc))

        # SMOTE_siamese_knn_tta_auc = self.SMOTE_siamese_knn_tta_metrics_folds_list[0][-1]
        # SMOTE_siamese_knn_tta_fpr, SMOTE_siamese_knn_tta_tpr, threshold = roc_curve(self.y_true_folds_list[0], self.SMOTE_siamese_knn_tta_y_pred_loss_folds_list[0])
        # plt.plot(SMOTE_siamese_knn_tta_fpr, SMOTE_siamese_knn_tta_tpr, label='SMOTE Siamese-NN TTA AUC = {:.3f}'.format(SMOTE_siamese_knn_tta_auc))

        # BorderlineSMOTE_regular_knn_tta_auc = self.BorderlineSMOTE_regular_knn_tta_metrics_folds_list[0][-1]
        # BorderlineSMOTE_regular_knn_tta_fpr, BorderlineSMOTE_regular_knn_tta_tpr, threshold = roc_curve(self.y_true_folds_list[0], self.BorderlineSMOTE_regular_knn_tta_y_pred_loss_folds_list[0])
        # plt.plot(BorderlineSMOTE_regular_knn_tta_fpr, BorderlineSMOTE_regular_knn_tta_tpr, label='BorderlineSMOTE Regular-NN TTA AUC = {:.3f}'.format(BorderlineSMOTE_regular_knn_tta_auc))

        # BorderlineSMOTE_siamese_knn_tta_auc = self.BorderlineSMOTE_siamese_knn_tta_metrics_folds_list[0][-1]
        # BorderlineSMOTE_siamese_knn_tta_fpr, BorderlineSMOTE_siamese_knn_tta_tpr, threshold = roc_curve(self.y_true_folds_list[0], self.BorderlineSMOTE_siamese_knn_tta_y_pred_loss_folds_list[0])
        # plt.plot(BorderlineSMOTE_siamese_knn_tta_fpr, BorderlineSMOTE_siamese_knn_tta_tpr, label='BorderlineSMOTE Siamese-NN TTA AUC = {:.3f}'.format(BorderlineSMOTE_siamese_knn_tta_auc))

        # plt.plot([0, 1], [0, 1], linestyle='dashed', color='gray')
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        # plt.legend(loc='lower right')
        # plt.ylabel("True Positive Rate")
        # plt.xlabel("False Positive Rate")
        # plt.title("ROC Curves")
        # # plt.show()
        # plt.savefig(f"/home/nivgold/plots/{self.dataset_name}_ROC_curve.png")