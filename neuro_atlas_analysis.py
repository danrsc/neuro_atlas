import os
import numpy
from sklearn import neighbors, cross_validation
import neuro_atlas_io
from neuro_atlas_features import read_all_features
from neuro_atlas_io import read_tracks
from neuro_atlas_util import status_iterate, unique_rows


def find_duplicates(input_dir, identical_track_path):
    """
    Exhaustive search over every pair to find exact duplicates
    :param input_dir: The directory containing text files of fiber tracks
    :param identical_track_path: The file in which to write duplicates. The output format is tab-separated with fields:
    file_path_1, index_track_1, file_path_2, index_track_2
    :return: The number of identical tracks
    """

    # first compute the neighbors on a small number of points to get a candidate set
    # this is much faster than true exhaustive search
    numeric_labels, feature_vectors, label_to_string, _ = read_all_features(input_dir)
    nn = neighbors.NearestNeighbors(n_neighbors=2)
    nn.fit(feature_vectors)
    distances, indices = nn.kneighbors(feature_vectors, n_neighbors=2)
    # find any vector with an exact neighbor that is not itself
    candidates = feature_vectors[distances[:, 1] == 0]
    # expand k until we find all exact neighbors of every vector with at least one exact neighbor
    for k in xrange(3, len(numeric_labels)):
        distances, indices = nn.kneighbors(candidates, n_neighbors=k)
        # noinspection PyTypeChecker
        if numpy.count_nonzero(distances[:, k - 1] == 0) == 0:
            break

    candidate_pairs = list()
    indicator_exact = distances == 0
    for index in xrange(distances.shape[0]):
        exact_matches = indices[index, indicator_exact[index, :]][1:]
        for exact_match in exact_matches:
            if indices[index, 0] < exact_match:
                candidate_pairs.append((indices[index, 0], exact_match))
            else:
                candidate_pairs.append((exact_match, indices[index, 0]))

    candidate_pairs = unique_rows(numpy.array(candidate_pairs))

    def __convert_label_index_to_file_index_tuple(index_label):
        numeric_label = numeric_labels[index_label]
        # how many matching labels have we seen up to, but not including, the index of interest
        # noinspection PyTypeChecker
        index_within_label = numpy.sum(numeric_labels[:index_label] == numeric_label)
        return os.path.join(input_dir, label_to_string[numeric_label] + '.txt'), index_within_label

    # noinspection PyTypeChecker
    candidates = map(lambda index_into_exact: (
        __convert_label_index_to_file_index_tuple(candidate_pairs[index_into_exact, 0]),
        __convert_label_index_to_file_index_tuple(candidate_pairs[index_into_exact, 1])),
        range(candidate_pairs.shape[0]))

    # group by file
    file_index_to_track = dict()
    file_to_max_index = dict()
    for ((file_1, index_1), (file_2, index_2)) in candidates:
        if file_1 not in file_to_max_index or file_to_max_index[file_1] < index_1:
            file_to_max_index[file_1] = index_1
        if file_2 not in file_to_max_index or file_to_max_index[file_2] < index_2:
            file_to_max_index[file_2] = index_2
        file_index_to_track[(file_1, index_1)] = None
        file_index_to_track[(file_2, index_2)] = None

    for current_file, max_index in file_to_max_index.iteritems():
        for index_track, track in enumerate(read_tracks(current_file)):
            if index_track > max_index:
                break
            if (current_file, index_track) in file_index_to_track:
                file_index_to_track[(current_file, index_track)] = track

    num_identical = 0
    with open(identical_track_path, 'w') as identical_file:
        for candidate in candidates:
            track_1 = file_index_to_track[candidate[0]]
            track_2 = file_index_to_track[candidate[1]]
            if numpy.array_equal(track_1, track_2):
                num_identical += 1
            identical_file.write(
                '{0}\t{1}\t{2}\t{3}\n'.format(candidate[0][0], candidate[0][1], candidate[1][0], candidate[1][1]))

    return num_identical


def label_counts(input_dir):
    """
    Returns a dictionary mapping labels to track counts
    :param input_dir: The directory in which to
    :return:
    """

    counts = dict()
    for file_name in status_iterate('{item} {fraction_complete:.2%} of files processed', os.listdir(input_dir)):
        label = os.path.splitext(file_name)[0]
        count = 0
        for _ in neuro_atlas_io.read_tracks(os.path.join(input_dir, file_name)):
            count += 1
        counts[label] = count
    return counts


def cv_knn(input_dir, num_points_in_features=3):
    """
    Runs cross-validated knn on the data in the input directory. The name of each file in the input directory
    is taken as the label for the fiber tracks in that file.
    :param input_dir: Directory of the data
    :param num_points_in_features: The number of points to use to represent each track
    :return: The accuracies for each fold, the number of bad tracks
    """

    numeric_labels, feature_vectors, label_to_string, num_bad_tracks = read_all_features(
        input_dir, num_interpolated_points=num_points_in_features)

    print('running cross validation...')
    estimator = neighbors.KNeighborsClassifier(n_neighbors=1)
    scores = cross_validation.cross_val_score(estimator, feature_vectors, numeric_labels,
                                              cv=cross_validation.StratifiedKFold(
                                                  numeric_labels, n_folds=10, shuffle=True))
    return scores, num_bad_tracks


def cv_knn_vs_num_points(input_dir, num_points_list=None):

    if num_points_list is None:
        num_points_list = range(3, 20)
    result_dict = dict()
    for num_points in num_points_list:
        result_dict[num_points] = cv_knn(input_dir, num_points)

    return result_dict
