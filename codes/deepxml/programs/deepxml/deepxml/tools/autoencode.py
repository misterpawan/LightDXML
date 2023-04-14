import numpy as np
import xclib.data.data_utils as data_utils
from models.autoencoder_layer import Autoencoder, train
import torch
import os
import json


class LabelEmbedding(object):
    """
    Generate mapping of labels in lower dimensional space

    Arguments:
    ----------
    label_embed_dims: int, optional (default: 300)
        - Dimensionality of the label space after embedding using random walk
    feature_type: str, optional (default: 'sparse')
        - 'sparse' the label matrix is sparse (csr_matrix)
        - 'dense' the label matrix is dense
    """
    def __init__(self, label_embed_dims=300, feature_type='sparse', method=0):
        if feature_type == 'dense':
            raise NotImplementedError("Label embedding for dense matrices has not been implemented yet")
        self.feature_type = feature_type
        self.label_embed_dims = label_embed_dims
        self.method = method
        self.mapping = None

    # noinspection DuplicatedCode
    @staticmethod
    def remove_documents_wo_features(features, labels):
        if isinstance(features, np.ndarray):
            features = np.power(features, 2)
        else:
            features = features.power(2)
        freq = np.array(features.sum(axis=1)).ravel()
        indices = np.where(freq > 0)[0]
        features = features[indices]
        labels = labels[indices]
        return features, labels

    def gen_label_x(self, labels):
        indices, labels = labels.nonzero()
        formatted_ds = {}
        for idx, index in enumerate(indices):
            if index in formatted_ds:
                formatted_ds[index].append(labels[idx])
            else:
                formatted_ds[index] = [labels[idx]]

        X, y = [], []
        for index in formatted_ds.keys():
            for label in formatted_ds[index]:
                X.append(label)
        return np.array(X)

    def gen_mapping(self, labels, epochs, batch_size):
        X = None
        if self.method == 0:
            X = self.gen_label_x(labels)

        embed = Autoencoder(labels.get_shape()[1], self.label_embed_dims)
        train(embed, X, epochs, batch_size)
        self.mapping = np.hstack((
            self.valid_labels.reshape((self.valid_labels.shape[0], 1)),
            embed.encode(
                torch.LongTensor(
                    list(range(self.valid_labels.shape[0])))
            ).detach().numpy()))

    @staticmethod
    def get_valid_labels(labels):
        freq = np.array(labels.sum(axis=0)).ravel()
        ind = np.where(freq > 0)[0]
        return labels[:, ind], ind

    def fit(self, features, labels, epochs, batch_size):
        self.num_labels = labels.shape[1]
        # Remove documents w/o any feature
        # these may impact the count, if not removed
        features, labels = self.remove_documents_wo_features(features, labels)
        # keep only valid labels; main code will also remove invalid labels
        labels, self.valid_labels = self.get_valid_labels(labels)
        self.gen_mapping(labels, epochs, batch_size)


def run(feat_fname, lbl_fname, feature_type, label_embed_dims, seed, epochs, batch_size, tmp_dir):
    np.random.seed(seed)
    if feature_type == 'dense':
        features = data_utils.read_gen_dense(feat_fname)
    elif feature_type == 'sparse':
        features = data_utils.read_gen_sparse(feat_fname)
    else:
        raise NotImplementedError()
    labels = data_utils.read_sparse_file(lbl_fname)
    assert features.shape[0] == labels.shape[0], \
        "Number of instances must be same in features and labels"
    num_features = features.shape[1]
    stats_obj = {'label_embed_dims': label_embed_dims}

    label_embed = LabelEmbedding(
        label_embed_dims=label_embed_dims, feature_type=feature_type)
    label_embed.fit(features, labels, epochs, batch_size)
    # stats_obj['surrogate'] = "{},{},{}".format(
    #     num_features, sd.num_surrogate_labels, sd.num_surrogate_labels)
    stats_obj['extreme'] = "{},{},{}".format(
        num_features, label_embed.num_labels, len(label_embed.valid_labels))

    json.dump(stats_obj, open(
        os.path.join(tmp_dir, "data_stats.json"), 'w'), indent=4)

    np.savetxt(os.path.join(tmp_dir, "valid_labels.txt"),
               label_embed.valid_labels, fmt='%d')
    np.savetxt(os.path.join(tmp_dir, "label_mapping.txt"),
               label_embed.mapping, fmt='%18e')
