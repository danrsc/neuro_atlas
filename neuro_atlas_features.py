import numpy


class TooFewPointsError(Exception):
    """
    Thrown by interpolate if a track has too few points for the requested number of interpolated points
    """

    def __init__(self, message):
        Exception(self, message)


def interpolate(track_points, num_resulting_points=None):
    """
    Keeps the first and last point of a track, divides the remaining points equally into num_resulting_points groups
    and keeps the mean of each group
    :param track_points: The points to interpolate, array-like
    :param num_resulting_points: The number of resulting points (including the start/end points)
    :return: A track with interpolated points (ndarray)
    """

    if num_resulting_points is None:
        num_resulting_points = 3

    if num_resulting_points < 3:
        raise ValueError('Must have at least 3 points')

    if len(track_points) < num_resulting_points:
        raise TooFewPointsError('Not enough points in track: {0}'.format(len(track_points)))

    segment_breaks = numpy.linspace(0, len(track_points), num_resulting_points - 1)
    last_segment_break = 0

    interpolated = list()
    interpolated.append(track_points[0, :])
    for index_segment, segment_break in enumerate(segment_breaks[1:]):
        try:
            start = int(numpy.floor(last_segment_break)) + 1
            if index_segment == len(segment_breaks) - 2:
                end = int(segment_break)
            else:
                end = int(numpy.floor(segment_break)) + 1
            interpolated.append(numpy.mean(track_points[start:end], axis=0))
        except:
            raise ValueError(len(track_points))
        last_segment_break = segment_break
    interpolated.append(track_points[-1, :])

    return numpy.vstack(interpolated)


def extract_features_from_track(
        track,
        num_interpolated_points=None,
        num_dropped_start_segments=None,
        num_dropped_end_segments=None,
        is_subtract_start=None):
    """
    Converts a track to a feature representation
    :param track: The track from which to extract features
    :param num_interpolated_points: How many points to use to characterize the track
    :param num_dropped_start_segments: Number of points to drop from the beginning of the track (before interpolation)
    :param num_dropped_end_segments: Number of points to drop from the end of the track (before interpolation)
    :param is_subtract_start: If True, the track is translated by -track[0] before the features are computed
    :return: The features for the track as a list
    """

    if is_subtract_start is None:
        is_subtract_start = False
    if num_dropped_start_segments is None:
        num_dropped_start_segments = 0
    if num_dropped_end_segments is None:
        num_dropped_end_segments = 0

    if is_subtract_start:
        track -= numpy.tile(track[0, :], (track.shape[0], 1))

    if num_dropped_start_segments > 0 or num_dropped_end_segments > 0:
        track = track[num_dropped_start_segments:-num_dropped_end_segments]

    interpolated = interpolate(track, num_interpolated_points)
    features = list()
    for i in range(interpolated.shape[0]):
        features.append(interpolated[i, :])
    features.append(track.shape[0])
    features.append(numpy.sum(numpy.linalg.norm(numpy.diff(track, axis=1))))
    return features


def vectorize(features):
    """
    Convert a feature list to a vector
    :param features: The feature list to convert
    :return: The vector
    """

    vectorized_features = list()
    for feature in features:
        vectorized_features.append(numpy.reshape(feature, (1, -1)))
    return numpy.hstack(vectorized_features)


def convert_to_features_unlabeled(unlabeled_tracks, featurize):

    num_bad_tracks = 0
    feature_vectors = list()
    for index, track in unlabeled_tracks:
        try:
            features = featurize(track)
        except TooFewPointsError:
            num_bad_tracks += 1
            continue
        feature_vectors.append(vectorize(features))
    feature_vectors = numpy.vstack(feature_vectors)
    return feature_vectors, num_bad_tracks


def convert_to_features(labeled_tracks, featurize):
    """
    Gets the features and labels for every track in the input directory
    :param labeled_tracks: An iterable of (label, index, track) tuples
    :param featurize: A callable which converts the track to an iterable of features
    :return:
        numeric_labels: A vector of the numeric labels for each sample
        feature_vectors: A matrix where each row is the feature vector for the sample
        numeric_label_to_label: A dictionary mapping numeric labels to strings
        num_bad_tracks: The number of tracks that could not be featurized
    """

    numeric_label_to_label = dict()
    label_to_numeric_label = dict()

    numeric_labels = list()
    feature_vectors = list()
    num_bad_tracks = 0
    for label, index, track in labeled_tracks:
        if label not in label_to_numeric_label:
            label_to_numeric_label[label] = len(numeric_label_to_label)
            numeric_label_to_label[len(numeric_label_to_label)] = label
        numeric_label = label_to_numeric_label[label]
        try:
            features = featurize(track)
        except TooFewPointsError:
            num_bad_tracks += 1
            continue

        feature_vectors.append(vectorize(features))
        numeric_labels.append(numeric_label)

    numeric_labels = numpy.array(numeric_labels)
    feature_vectors = numpy.vstack(feature_vectors)
    return numeric_labels, feature_vectors, numeric_label_to_label, num_bad_tracks
