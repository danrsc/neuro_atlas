import os
import numpy
from neuro_atlas_io import read_tracks
from neuro_atlas_util import status_iterate


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


def extract_features(
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


def read_all_features(
        input_dir,
        num_interpolated_points=None,
        num_dropped_start_segments=None,
        num_dropped_end_segments=None,
        is_subtract_start=None):
    """
    Gets the features and labels for every track in the input directory
    :param input_dir: The directory where the track data can be found
    :param num_interpolated_points: How many points to use to characterize the track
    :param num_dropped_start_segments: Number of points to drop from the beginning of the track (before interpolation)
    :param num_dropped_end_segments: Number of points to drop from the end of the track (before interpolation)
    :param is_subtract_start: If True, the track is translated by -track[0] before the features are computed
    :return:
        all_labels: A vector of the numeric labels for each sample
        all_vectors: A matrix where each row is the feature vector for the sample
        id_to_label: A dictionary mapping numeric labels to strings
        num_bad_tracks_in_read: The number of tracks that could not be featurized
    """

    id_to_label = dict()
    all_labels = list()
    all_vectors = list()
    num_bad_tracks_in_read = 0
    for index_file, file_name in enumerate(
            status_iterate('{item} {fraction_complete:.2%} of files processed', os.listdir(input_dir))):

        label = os.path.splitext(file_name)[0]
        id_to_label[index_file] = label

        file_path = os.path.join(input_dir, file_name)
        for track in read_tracks(file_path):

            try:
                features = extract_features(
                    track,
                    num_interpolated_points=num_interpolated_points,
                    num_dropped_start_segments=num_dropped_start_segments,
                    num_dropped_end_segments=num_dropped_end_segments,
                    is_subtract_start=is_subtract_start)
            except TooFewPointsError:
                num_bad_tracks_in_read += 1
                continue

            features = vectorize(features)
            all_labels.append(index_file)
            all_vectors.append(features)

    all_labels = numpy.array(all_labels)
    all_vectors = numpy.vstack(all_vectors)
    return all_labels, all_vectors, id_to_label, num_bad_tracks_in_read
