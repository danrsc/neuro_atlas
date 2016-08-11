import argparse
import os
import numpy
from sklearn import neighbors, cross_validation
import neuro_atlas_analysis
import neuro_atlas_visualizations
import neuro_atlas_io
import neuro_atlas_features
import functools


class Analyses:
    """
    Class to hold constants for analysis names
    """

    def __init__(self):
        pass

    cv_knn = 'cv_knn'
    cv_supervised_quick_bundles = 'cv_supervised_quick_bundles'
    cv_knn_vs_num_points = 'cv_knn_vs_num_points'
    label_counts = 'label_counts'
    find_duplicates = 'find_duplicates'
    distance_investigate = 'distance_investigate'

all_analyses = [name for name in vars(Analyses) if not name.startswith('_')]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='This program runs an analysis (for example cross-validated knn) on the input data '
                    'directory of fiber tracks')

    parser.add_argument('--input_path', '-i',
                        help='The input data directory', required=True)
    parser.add_argument('--output_path', '-o',
                        help='The path to the output file. '
                             'For find_duplicates, this file will contain the duplicates. '
                             'For label_counts, this file will contain the counts. '
                             'For cv_knn, this file is not used. '
                             'For cv_knn_vs_num_points, this file must have an image data extension (e.g. \'.png\') '
                             'and will be a plot of the accuracy vs. number of points',
                        required=True)
    parser.add_argument('--analysis', '-a',
                        help='Which analysis to run',
                        choices=all_analyses,
                        required=True)
    parser.add_argument('--num_points_in_features',
                        help='How many points to use to characterize a track. For cv_knn this should be an integer. '
                             'For cv_knn_vs_num_points this should be a comma-delimited list of integers. '
                             'If not provided, defaults to 3 for cv_knn, and range(3, 20) for cv_knn_vs_num_points')
    parser.add_argument('--quick_bundles_threshold',
                        help='When cv_supervised_quick_bundles is active, there are two ways of clustering. '
                             'The clusters can be formed directly by looking at the training labels (i.e. one cluster '
                             'per label, or the normal unsupervised quick bundles can be run and labels can be '
                             'assigned to the resulting clusters by vote. If --quick_bundles_threshold is set to '
                             'something other than None, the unsupervised technique is used. This threshold is used to '
                             'decide when a fiber is added to a cluster in the unsupervised technique. Note that a '
                             'fiber can be assigned to multiple clusters if this threshold is met for multiple '
                             'clusters. If the value of this parameter is negative, the algorithm will choose it. '
                             'Otherwise this value will be used.',
                        type=int)

    parsed_arguments = parser.parse_args()
    for name, value in vars(parsed_arguments).iteritems():
        print '{0}: {1}'.format(name, value)

    input_path = os.path.abspath(parsed_arguments.input_path)
    output_path = os.path.abspath(parsed_arguments.output_path)

    labeled_tracks = neuro_atlas_io.read_all_tracks_with_labels(input_path)

    if parsed_arguments.analysis == Analyses.cv_knn:

        num_points = getattr(parsed_arguments, 'num_points_in_features', None)
        if num_points is not None:
            try:
                num_points = int(num_points)
            except (TypeError, ValueError):
                parser.error('Bad argument for num_points_in_features {0}'.format(
                    parsed_arguments.num_points_in_features))
                exit(1)

        featurize = functools.partial(
            neuro_atlas_features.extract_features_from_track, num_interpolated_points=num_points)

        numeric_labels, feature_vectors, numeric_label_to_label, num_bad_tracks_in_read = \
            neuro_atlas_features.convert_to_features(labeled_tracks, featurize)

        estimator = neighbors.KNeighborsClassifier(n_neighbors=1)
        accuracies = cross_validation.cross_val_score(
            estimator, feature_vectors, numeric_labels, cv=cross_validation.StratifiedKFold(
                numeric_labels, n_folds=10, shuffle=True))

        print('Accuracies:')
        print(accuracies)
        mean_accuracy = numpy.mean(accuracies)
        print('Mean: {0}, Std: {1}'.format(mean_accuracy, numpy.std(accuracies)))
        print('Mean Error: {0}'.format(1 - mean_accuracy))

    elif parsed_arguments.analysis == Analyses.distance_investigate:

        num_points = getattr(parsed_arguments, 'num_points_in_features', None)
        if num_points is not None:
            try:
                num_points = int(num_points)
            except (TypeError, ValueError):
                parser.error('Bad argument for num_points_in_features {0}'.format(
                    parsed_arguments.num_points_in_features))
                exit(1)

        featurize = functools.partial(neuro_atlas_features.interpolate, num_resulting_points=num_points)

        numeric_labels, feature_vectors, numeric_label_to_label, num_bad_tracks_in_read = \
            neuro_atlas_features.convert_to_features(labeled_tracks, featurize)

        numeric_label_to_closest_same_different = neuro_atlas_analysis.distance_investigate(
            numeric_labels, feature_vectors)  # neuro_atlas_analysis.SupervisedQuickBundles.minimum_average_direct_flip)

        for label, numeric in sorted([(value, key) for key, value in numeric_label_to_label.iteritems()]):
            furthest_same, closest_different = numeric_label_to_closest_same_different[numeric]
            print label, furthest_same, closest_different

    elif parsed_arguments.analysis == Analyses.cv_supervised_quick_bundles:

        num_points = getattr(parsed_arguments, 'num_points_in_features', None)
        if num_points is not None:
            try:
                num_points = int(num_points)
            except (TypeError, ValueError):
                parser.error('Bad argument for num_points_in_features {0}'.format(
                    parsed_arguments.num_points_in_features))
                exit(1)

        featurize = functools.partial(neuro_atlas_features.interpolate, num_resulting_points=num_points)

        numeric_labels, feature_vectors, numeric_label_to_label, num_bad_tracks_in_read = \
            neuro_atlas_features.convert_to_features(labeled_tracks, featurize)

        if parsed_arguments.quick_bundles_threshold is not None:
            if parsed_arguments.quick_bundles_threshold < 0:
                qb_threshold = 10
            else:
                qb_threshold = parsed_arguments.quick_bundles_threshold

        estimator = neuro_atlas_analysis.SupervisedQuickBundles(
            unsupervised_thresh=parsed_arguments.quick_bundles_threshold)
        accuracies = cross_validation.cross_val_score(
            estimator, feature_vectors, numeric_labels, cv=cross_validation.StratifiedKFold(
                numeric_labels, n_folds=10, shuffle=True))

        print('Accuracies:')
        print(accuracies)
        mean_accuracy = numpy.mean(accuracies)
        print('Mean: {0}, Std: {1}'.format(mean_accuracy, numpy.std(accuracies)))
        print('Mean Error: {0}'.format(1 - mean_accuracy))

    elif parsed_arguments.analysis == Analyses.cv_knn_vs_num_points:

        num_points = getattr(parsed_arguments, 'num_points_in_features', None)
        if num_points is not None:
            try:
                num_points = [int(x.strip()) for x in num_points.split(',') if len(x.strip()) > 0]
            except (TypeError, ValueError):
                parser.error('Bad argument for num_points_in_features {0}'.format(
                    parsed_arguments.num_points_in_features))
                exit(1)

        if num_points is None:
            num_points = range(3, 20)

        dict_to_plot = dict()
        # make this reusable by initializing a list
        labeled_tracks = list(labeled_tracks)
        for current_num_points in num_points:

            featurize = functools.partial(
                neuro_atlas_features.extract_features_from_track, num_interpolated_points=current_num_points)

            numeric_labels, feature_vectors, numeric_label_to_label, num_bad_tracks_in_read = \
                neuro_atlas_features.convert_to_features(labeled_tracks, featurize)

            estimator = neighbors.KNeighborsClassifier(n_neighbors=1)
            accuracies = cross_validation.cross_val_score(
                estimator, feature_vectors, numeric_labels, cv=cross_validation.StratifiedKFold(
                    numeric_labels, n_folds=10, shuffle=True))

            mean_accuracy = numpy.mean(accuracies)
            print 'Num points: {0}, Mean: {1}, Std: {2} Mean Error: {3}, Bad Tracks: {3}'.format(
                current_num_points, mean_accuracy, numpy.std(accuracies), 1 - mean_accuracy, num_bad_tracks_in_read)

            dict_to_plot[current_num_points] = accuracies
            neuro_atlas_visualizations.plot_dict(dict_to_plot, output_path)

    elif parsed_arguments.analysis == Analyses.label_counts:

        label_counts = neuro_atlas_analysis.label_counts(labeled_tracks)
        sorted_counts = sorted(label_counts.iteritems(), key=lambda (s_l, s_c): s_c)
        total = sum([c for l, c in sorted_counts])
        with open(output_path, 'w') as output_file:
            for label, count in sorted_counts:
                output_file.write('{0}\t{1}\n'.format(label, count))
                print ('{0}: {1} ({2:.2%})'.format(label, count, float(count) / total))

    elif parsed_arguments.analysis == Analyses.find_duplicates:

        num_duplicates = neuro_atlas_analysis.find_duplicates(labeled_tracks, output_path)
        print '{0} duplicates found'.format(num_duplicates)

    else:

        parser.error('Unknown analysis: {0}'.format(parsed_arguments.analysis))
