import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf

import cudf, cuml
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors

from autoencdoer_model import *


class Solver:
    def __init__(self, train_ds, test_ds, epochs=32, features_dim=76):
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.num_epochs = epochs

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
        # enable GPU
        # setup tensorboard

    def save_weights(self, path, dataset_name):
        encoder_path = path + f"/epochs_{self.num_epochs}_{dataset_name}_encoder"
        decoder_path = path + f"/epochs_{self.num_epochs}_{dataset_name}_decoder"

        self.encoder.save(encoder_path)
        self.decoder.save(decoder_path)

    def load_weights(self, path):
        encoder_path = path + "/encoder"
        decoder_path = path + "/decoder"
        self.encoder = tf.keras.models.load_model(encoder_path)
        self.decoder = tf.keras.models.load_model(decoder_path)

    def train(self):

        # loss function
        self.loss_func = tf.keras.losses.MeanSquaredError()
        for epoch in range(self.num_epochs):
            epoch_loss_mean = tf.keras.metrics.Mean()

            for step, (x_batch_train, y_batch_train) in enumerate(self.train_ds):
                if step%1000 == 0: print(f'Step: {step} in epoch {epoch}')
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


    def get_oversampling_method(self, method_name):
        methods_dict = {
            'smote': [SMOTE],
            'borderline_smote': [BorderlineSMOTE],
            'both': [SMOTE, BorderlineSMOTE]
        }
        return methods_dict[method_name]


    def test_tta(self, method_name):

        # loss function - with reduction equals to `NONE` in order to get the loss of every test example
        self.loss_func = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        # iterating through the train to calculate the loss of every train instance
        train_loss = []
        X_train_list = []
        y_train_list = []
        for step, (x_batch_train, y_batch_train) in enumerate(self.train_ds):
            loss = self.test_step(x_batch_train)
            train_loss.append(loss.numpy())

            X_train_list.append(x_batch_train)
            y_train_list.append(y_batch_train)

        # adjusting X to be: ndarray of shape (num_dataset_samples, num_dataset_features)
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        # decide the oversampling method
        oversampling_method = self.get_oversampling_method(method_name)
        NUM_OF_AUGMENTATIONS = 5

        # if SMOTE is the augmentation method -> training KNN model with training samples (RAPIDS KNN IMPLEMENTATION)
        X_rapids = cudf.DataFrame(X_train)
        knn_model = cuNearestNeighbors(n_neighbors=NUM_OF_AUGMENTATIONS)

        knn_model.fit(X_rapids)

        # iterating through the test to calculate the loss of every test instance
        test_loss = []
        test_labels = []
        for step, (x_batch_test, y_batch_test) in enumerate(self.test_ds):
            # x_batch_test: ndarray of shape (batch_size, num_dataset_features)
            # y_batch_test: ndarray of shape (batch_size,)

            reconstruction_loss = self.test_step(x_batch_test).numpy()
            test_labels.append(y_batch_test.numpy())

            # TODO: make batch samples using oversampling method, calculate predictions and final predication
            # TODO: the tta_features_batch shape is: (batch_size, num_of_augmentations, num_dataset_features)

            # neighbors_indices: ndarray of shape (batch_size, num_of_augmentations)
            neighbors_indices = knn_model.kneighbors(x_batch_test, return_distance=False)
            neighbors_indices = np.squeeze(neighbors_indices)

            # tta_features_batch: ndarray of shape (batch_size, num_dataset_features)
            neighbors_batch_features = X_train[neighbors_indices]
            neighbors_batch_labels = y_train[neighbors_indices]

            tta_reconstruction = []
            for neighbors_features, neighbors_labels in list(zip(neighbors_batch_features, neighbors_batch_labels)):
                # TODO: dump the tta creation to a function
                fake_tta_features = [tta_feature for tta_feature in neighbors_features] + [neighbors_features[-1] for i in range(NUM_OF_AUGMENTATIONS)]
                fake_tta_labels = [1 for i in fake_tta_features]
                tta_features_full = np.concatenate([neighbors_features, fake_tta_features])
                tta_labels_full = np.concatenate([neighbors_labels, fake_tta_labels])

                osmp_obj = oversampling_method(sampling_strategy='minority', k_neighbors=NUM_OF_AUGMENTATIONS-1)
                X_res, y_res = osmp_obj.fit_resample(tta_features_full, tta_labels_full)

                tta_features = X_res[-NUM_OF_AUGMENTATIONS:]

                # making the test phase for the tta sample
                tta_reconstruction_loss = self.test_step(tta_features).numpy()
                tta_reconstruction.append(tta_reconstruction_loss)


            # # calculating the tta samples reconstructions
            # tta_reconstruction = []
            # for tta_sample in tta_features_batch:
            #     tta_reconstruction_loss = self.test_step(tta_sample).numpy()
            #     tta_reconstruction.append(tta_reconstruction_loss)

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

    def get_oversampling_tta_sample(self, neighbors_features, neighbors_labels, NUM_OF_AUGMENTATIONS):
        fake_tta_features = [tta_feature for tta_feature in neighbors_features] + [neighbors_features[-1] for i in
                                                                                   range(NUM_OF_AUGMENTATIONS)]
        fake_tta_labels = [1 for i in fake_tta_features]
        tta_features_full = np.concatenate([neighbors_features, fake_tta_features])
        tta_labels_full = np.concatenate([neighbors_labels, fake_tta_labels])

        osmp_obj = oversampling_method(sampling_strategy='minority', k_neighbors=NUM_OF_AUGMENTATIONS - 1)
        X_res, y_res = osmp_obj.fit_resample(tta_features_full, tta_labels_full)

        return X_res[-NUM_OF_AUGMENTATIONS:]

    @tf.function
    def test_step(self, inputs):
        latent_var = self.encoder(inputs)
        reconstructed = self.decoder(latent_var)
        reconstruction_loss = self.loss_func(inputs, reconstructed)

        return reconstruction_loss