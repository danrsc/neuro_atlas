import numpy
from sklearn import neighbors
from scipy.spatial import distance as spdist
import neuro_atlas_features
from neuro_atlas_util import unique_rows


def find_duplicates(labeled_tracks, identical_track_path):
    """
    Exhaustive search over every pair to find exact duplicates
    :param labeled_tracks: An iterable of (label, index, track) tuples
    :param identical_track_path: The file in which to write duplicates. The output format is tab-separated with fields:
    label_1, index_track_1, label_2, index_track_2
    :return: The number of identical tracks
    """

    numeric_label_to_label = dict()
    label_to_numeric_label = dict()

    # first compute the neighbors on a small number of points to get a candidate set
    # this is much faster than true exhaustive search
    numeric_labels = list()
    relative_indices = list()
    interpolated_tracks = list()

    # we're just going to store these all in memory, probably fine
    original_tracks = list()

    short_tracks = list()
    short_labels = list()
    short_relative_indices = list()

    for label, index, track in labeled_tracks:
        if label not in label_to_numeric_label:
            label_to_numeric_label[label] = len(numeric_label_to_label)
            numeric_label_to_label[len(numeric_label_to_label)] = label
        try:
            interpolated_tracks.append(neuro_atlas_features.interpolate(track, num_resulting_points=3).reshape((1, -1)))
            original_tracks.append(track)
            numeric_labels.append(label_to_numeric_label[label])
            relative_indices.append(index)
        except neuro_atlas_features.TooFewPointsError:
            short_tracks.append(track)
            short_labels.append(label_to_numeric_label[label])
            short_relative_indices.append(index)

    # add information about tracks that were too short to featurize to the end
    original_tracks += short_tracks
    numeric_labels += short_labels
    relative_indices += short_relative_indices

    numeric_labels = numpy.array(numeric_labels)
    relative_indices = numpy.array(relative_indices)
    feature_vectors = numpy.vstack(interpolated_tracks)
    nn = neighbors.NearestNeighbors(n_neighbors=2)
    nn.fit(feature_vectors)

    # find any vector with an exact neighbor that is not itself
    distances, indices = nn.kneighbors(feature_vectors, n_neighbors=2)
    candidates = feature_vectors[distances[:, 1] == 0]
    # expand k until we find all exact neighbors of every vector with at least one exact neighbor
    for k in xrange(3, len(numeric_labels)):
        distances, indices = nn.kneighbors(candidates, n_neighbors=k)
        # noinspection PyTypeChecker
        if numpy.count_nonzero(distances[:, k - 1] == 0) == 0:
            break

    candidate_pairs = list()
    indicator_zero_distance = distances == 0
    for index in xrange(distances.shape[0]):
        indices_zero_distance = indices[index, indicator_zero_distance[index, :]][1:]
        for index_zero in indices_zero_distance:
            if indices[index, 0] < index_zero:
                candidate_pairs.append((indices[index, 0], index_zero))
            else:
                candidate_pairs.append((index_zero, indices[index, 0]))
    for index_1 in range(len(short_tracks)):
        for index_2 in range(index_1 + 1, len(short_tracks)):
            candidate_pairs.append((index_1 + feature_vectors.shape[0], index_2 + feature_vectors.shape[0]))

    candidate_pairs = unique_rows(numpy.array(candidate_pairs))

    exact_pairs = list()
    for index_pair in xrange(candidate_pairs.shape[0]):
        if numpy.array_equal(
                original_tracks[candidate_pairs[index_pair, 0]],
                original_tracks[candidate_pairs[index_pair, 1]]):
            exact_pairs.append(candidate_pairs[index_pair, :])

    exact_pairs = map(lambda p: (
        (numeric_label_to_label[numeric_labels[p[0]]], relative_indices[p[0]]),
        (numeric_label_to_label[numeric_labels[p[1]]], relative_indices[p[1]])),
        exact_pairs)

    num_identical = len(exact_pairs)
    with open(identical_track_path, 'w') as identical_file:
        for pair in exact_pairs:
            identical_file.write('{0}\t{1}\t{2}\t{3}\n'.format(pair[0][0], pair[0][1], pair[1][0], pair[1][1]))

    return num_identical


def label_counts(labeled_tracks):
    """
    Returns a dictionary mapping labels to track counts
    :param labeled_tracks: An iterable of (label, index, track) tuples
    :return:
    """

    counts = dict()
    for label, index, track in labeled_tracks:
        if label not in counts:
            counts[label] = 1
        else:
            counts[label] += 1
    return counts


def distance_investigate(numeric_labels, feature_vectors, metric=None):
    """
    Tool to find the mean distance to the furthest fiber with the same label and the mean distance to the closest
    fiber with a different label for each label type
    :param numeric_labels: The numeric labels assigned to each fiber
    :param feature_vectors: The points in the track, vectorized
    :param metric: What metric to use for computing distance
    :return: A dictionary mapping numeric labels to a tuple of
    (mean distance to furthest same, mean distance to closest different)
    """

    # first find the furthest same label
    unique_labels = numpy.unique(numeric_labels)
    mean_furthest_same = numpy.full(unique_labels.shape, numpy.nan)
    for index_label, label in enumerate(unique_labels):
        print('Finding mean furthest same for {0} of {1} ({2:.2%})'.format(
            index_label, len(unique_labels), float(index_label) / len(unique_labels)))
        current_set = feature_vectors[label == numeric_labels]
        if metric is not None:
            same_distances = spdist.pdist(current_set, metric=metric)
        else:
            same_distances = spdist.pdist(current_set)
        mean_furthest_same[index_label] = numpy.mean(numpy.max(spdist.squareform(same_distances), axis=1))

    knn = neighbors.KNeighborsClassifier(metric=metric, n_neighbors=5000) if metric is not None else \
        neighbors.KNeighborsClassifier(n_neighbors=5000)

    print('Fitting ... ')
    knn.fit(feature_vectors, numeric_labels)

    closest_different = numpy.full(feature_vectors.shape[0], numpy.nan)

    def __update_closest(closest, k, is_same):
        distances, indices = knn.kneighbors(feature_vectors[numpy.isnan(closest)], n_neighbors=k)
        labels = numeric_labels[indices]
        current_closest = numpy.full(indices.shape[0], numpy.nan)
        for index in xrange(labels.shape[0]):
            if is_same:
                # noinspection PyTypeChecker,PyUnresolvedReferences
                matching_indices = numpy.nonzero(labels[index, 1:] == labels[index, 0])[0]
            else:
                # noinspection PyTypeChecker,PyUnresolvedReferences
                matching_indices = numpy.nonzero(labels[index, 1:] != labels[index, 0])[0]
            if len(matching_indices) > 0:
                current_closest[index] = distances[index, matching_indices[0]]
        closest[numpy.isnan(closest)] = current_closest
        return numpy.count_nonzero(numpy.isnan(closest))

    print ('Checking at k={0}'.format(2))
    current_k = 2
    num_different_not_found = __update_closest(closest_different, current_k, False)
    while current_k < feature_vectors.shape[0]:
        print('Still need to find closest different neighbors for {0}'.format(num_different_not_found))
        print ('Checking at k={0}'.format(current_k))
        num_different_not_found = __update_closest(closest_different, current_k, False)
        if num_different_not_found == 0:
            break
        if current_k == feature_vectors.shape[0] - 1:
            break
        current_k *= 2
        if current_k >= feature_vectors.shape[0]:
            current_k = feature_vectors.shape[0] - 1

    if num_different_not_found > 0:
        num_different_not_found = __update_closest(closest_different, feature_vectors.shape[0], False)

    print('Are we done? {0}'.format(num_different_not_found == 0))

    unique_labels = numpy.unique(numeric_labels)
    label_to_mean_closest_same_different = dict()
    # noinspection PyTypeChecker
    for index_label, numeric_label in enumerate(unique_labels):
        mean_distance_to_closest_different = numpy.mean(closest_different[numeric_label == numeric_labels])
        label_to_mean_closest_same_different[numeric_label] = (
            mean_furthest_same[index_label], mean_distance_to_closest_different)
    return label_to_mean_closest_same_different


class SupervisedQuickBundles(neighbors.KNeighborsClassifier):
    """
    You can imagine two ways of doing this.
    1) Since we have labels, just take the centroid of feature vectors with a given label, then run nearest neighbor
       using the resulting centroids as the training data.
    2) Do the unsupervised QuickBundles algorithm, then assign labels to the clusters by vote, and use these as the
    train data.
    When unsupervised_thresh is None this does way (1), when unsupervised_thresh is not None, this does way (2)
    This is simply nearest neighbor using k=1 and centroids to train
    """

    def __init__(self, unsupervised_thresh=None, weights='uniform', algorithm='auto', leaf_size=30, n_jobs=1, **kwargs):
        neighbors.KNeighborsClassifier.__init__(
            self, n_neighbors=1, weights=weights, algorithm=algorithm, leaf_size=leaf_size,
            metric=SupervisedQuickBundles.minimum_average_direct_flip, n_jobs=n_jobs, **kwargs)
        self.__unsupervised_thresh = unsupervised_thresh

    @staticmethod
    def minimum_average_direct_flip(x, y):
        # BallTree calls this with some kind of random values at the beginning to see if you
        # are doing something stupid in your metric, so we need to handle bogus sizes
        # https://github.com/scikit-learn/scikit-learn/issues/6287
        # :(
        try:
            x = numpy.reshape(x, (-1, 3))
            y = numpy.reshape(y, (-1, 3))
        except ValueError:
            # let's hope this is due to BallTree check and not something else
            return numpy.linalg.norm(x - y)
        direct = numpy.mean(numpy.sqrt(numpy.sum(numpy.square(x - y), axis=1)))
        flipped = numpy.mean(numpy.sqrt(numpy.sum(numpy.square(numpy.flipud(x) - y), axis=1)))
        return min(direct, flipped)

    def get_params(self, deep=True):
        base_params = neighbors.KNeighborsClassifier.get_params(self, deep)
        base_params['unsupervised_thresh'] = self.__unsupervised_thresh
        return base_params

    # noinspection PyPep8Naming
    def fit(self, X, y):

        if self.__unsupervised_thresh is not None:

            # user can pass in a negative to let us decide
            if self.__unsupervised_thresh <= 0:
                self.__unsupervised_thresh = 20

            clusters = list()
            cluster_labels = list()
            # permute the data so we don't bias the clusters too much
            X = numpy.random.permutation(X)
            for index_vector in xrange(X.shape[0]):
                is_clustered = False
                for index_cluster in xrange(len(clusters)):
                    if SupervisedQuickBundles.minimum_average_direct_flip(
                            X[index_vector], clusters[index_cluster]) < self.__unsupervised_thresh:
                        is_clustered = True
                        clusters[index_cluster] = (
                            (clusters[index_cluster] * len(cluster_labels[index_cluster]) + X[index_vector]) /
                            (len(cluster_labels[index_cluster]) + 1))
                        cluster_labels[index_cluster].append(y[index_vector])
                if not is_clustered:
                    clusters.append(X[index_vector])
                    cluster_labels.append([y[index_vector]])
            for index_cluster in xrange(len(clusters)):
                current_labels, current_label_counts = numpy.unique(cluster_labels[index_cluster], return_counts=True)
                cluster_labels[index_cluster] = current_labels[numpy.argmax(current_label_counts)]
            clusters = numpy.vstack(clusters)
            cluster_labels = numpy.array(cluster_labels)
            neighbors.KNeighborsClassifier.fit(self, clusters, cluster_labels)
        else:

            unique_labels = numpy.unique(y)
            centroids = numpy.full((len(unique_labels), X.shape[1]), numpy.nan)
            for index_label, unique_label in enumerate(unique_labels):
                centroids[index_label] = numpy.mean(X[y == unique_label], axis=0)
            neighbors.KNeighborsClassifier.fit(self, centroids, unique_labels)
