import argparse
import os
import numpy
from sklearn import neighbors, cross_validation
import neuro_atlas_analysis
import neuro_atlas_visualizations
import neuro_atlas_io
import neuro_atlas_features
from neuro_atlas_util import status_iterate
import functools
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import time


class Analyses:
    """
    Class to hold constants for analysis names
    """

    def __init__(self):
        pass

    cv_knn = 'cv_knn'
    individual_knn = 'individual_knn'
    cv_svm = 'cv_svm'
    individual_svm = 'individual_svm'
    cv_logistic_regression = 'cv_logistic_regression'
    individual_logistic_regression = 'individual_logistic_regression'
    cv_supervised_quick_bundles = 'cv_supervised_quick_bundles'
    individual_supervised_quick_bundles = 'individual_supervised_quick_bundles'
    cv_knn_vs_num_points = 'cv_knn_vs_num_points'
    individual_knn_vs_num_points = 'individual_knn_vs_num_points'
    label_counts = 'label_counts'
    find_duplicates = 'find_duplicates'
    distance_investigate = 'distance_investigate'
    make_atlas = 'make_atlas'
    label = 'label'

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
                             'and will be a plot of the accuracy vs. number of points. '
                             'For make_atlas, this path is used as a directory in which to write .nii files.',
                        required=True)
    parser.add_argument('--individual_path',
                        help='The path to the labeled tracks for an individual to use as the test set.')
    parser.add_argument('--unlabeled_path', '-u', help='The path to the input unlabeled data. Used only by \'label\'')
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
    unlabeled_path = getattr(parsed_arguments, 'unlabeled_path', None)
    if unlabeled_path is not None:
        unlabeled_path = os.path.abspath(parsed_arguments.unlabeled_path)
    individual_path = getattr(parsed_arguments, 'individual_path', None)
    if individual_path is not None:
        individual_path = os.path.abspath(parsed_arguments.individual_path)

    labeled_tracks = neuro_atlas_io.read_all_tracks_with_labels(input_path)

    num_points_in_features = getattr(parsed_arguments, 'num_points_in_features', None)
    if num_points_in_features is not None:
        if (parsed_arguments.analysis == Analyses.cv_knn_vs_num_points or
                parsed_arguments.analysis == Analyses.individual_knn_vs_num_points):
            try:
                num_points_in_features = [
                    int(x.strip()) for x in num_points_in_features.split(',') if len(x.strip()) > 0]
            except (TypeError, ValueError):
                parser.error('Bad argument for num_points_in_features {0}'.format(
                    parsed_arguments.num_points_in_features))
                exit(1)
        else:
            try:
                num_points_in_features = int(num_points_in_features)
            except (TypeError, ValueError):
                parser.error('Bad argument for num_points_in_features {0}'.format(
                    parsed_arguments.num_points_in_features))
                exit(1)

    if parsed_arguments.analysis == Analyses.cv_knn:

        featurize = functools.partial(
            neuro_atlas_features.extract_features_from_track, num_interpolated_points=num_points_in_features)

        numeric_labels, feature_vectors, numeric_label_to_label, num_bad_tracks_in_read = \
            neuro_atlas_features.convert_to_features(labeled_tracks, featurize)

        estimator = neighbors.KNeighborsClassifier(n_neighbors=1)
        t0 = time.time()
        accuracies = cross_validation.cross_val_score(
            estimator, feature_vectors, numeric_labels, cv=cross_validation.StratifiedKFold(
                numeric_labels, n_folds=10, shuffle=True))

        print('Accuracies:')
        print(accuracies)
        mean_accuracy = numpy.mean(accuracies)
        print('Mean: {0}, Std: {1}'.format(mean_accuracy, numpy.std(accuracies)))
        print('Mean Error: {0}'.format(1 - mean_accuracy))
        print('Time(seconds): {0}'.format(time.time() - t0))

    elif parsed_arguments.analysis == Analyses.cv_svm:

        featurize = functools.partial(
            neuro_atlas_features.extract_features_from_track, num_interpolated_points=num_points_in_features)

        numeric_labels, feature_vectors, numeric_label_to_label, num_bad_tracks_in_read = \
            neuro_atlas_features.convert_to_features(labeled_tracks, featurize)

        estimator = SVC(kernel="linear")
        t0 = time.time()
        accuracies = cross_validation.cross_val_score(
            estimator, feature_vectors, numeric_labels, cv=cross_validation.StratifiedKFold(
                numeric_labels, n_folds=10, shuffle=True))

        print('linear accuracies:')
        print(accuracies)
        mean_accuracy = numpy.mean(accuracies)
        print('Mean: {0}, Std: {1}'.format(mean_accuracy, numpy.std(accuracies)))
        print('Mean Error: {0}'.format(1 - mean_accuracy))
        print('Time(seconds): {0}'.format(time.time() - t0))

        estimator = SVC(kernel="rbf", gamma=0.0000082)
        t0 = time.time()
        accuracies = cross_validation.cross_val_score(
            estimator, feature_vectors, numeric_labels, cv=cross_validation.StratifiedKFold(
                numeric_labels, n_folds=10, shuffle=True))

        print('rbf accuracies:')
        print(accuracies)
        mean_accuracy = numpy.mean(accuracies)
        print('Mean: {0}, Std: {1}'.format(mean_accuracy, numpy.std(accuracies)))
        print('Mean Error: {0}'.format(1 - mean_accuracy))
        print('Time(seconds): {0}'.format(time.time() - t0))

    elif parsed_arguments.analysis == Analyses.cv_logistic_regression:

        featurize = functools.partial(
            neuro_atlas_features.extract_features_from_track, num_interpolated_points=num_points_in_features)

        numeric_labels, feature_vectors, numeric_label_to_label, num_bad_tracks_in_read = \
            neuro_atlas_features.convert_to_features(labeled_tracks, featurize)

        estimator = LogisticRegression(multi_class="ovr", penalty='l2', solver='sag', max_iter=1000, n_jobs=-1)
        t0 = time.time()
        accuracies = cross_validation.cross_val_score(
            estimator, feature_vectors, numeric_labels, cv=cross_validation.StratifiedKFold(
                numeric_labels, n_folds=10, shuffle=True))

        print('Accuracies:')
        print(accuracies)
        mean_accuracy = numpy.mean(accuracies)
        print('Mean: {0}, Std: {1}'.format(mean_accuracy, numpy.std(accuracies)))
        print('Mean Error: {0}'.format(1 - mean_accuracy))
        print('Time(seconds): {0}'.format(time.time() - t0))

    elif parsed_arguments.analysis == Analyses.label:

        if unlabeled_path is None:
            parser.error('unlabeled_path is required for analysis==\'label\'')

        if not os.path.exists(unlabeled_path):
            raise ValueError('Path does not exist: {0}'.format(unlabeled_path))

        featurize = functools.partial(
            neuro_atlas_features.extract_features_from_track, num_interpolated_points=num_points_in_features)

        numeric_labels, feature_vectors, numeric_label_to_label, num_bad_tracks_in_read = \
            neuro_atlas_features.convert_to_features(labeled_tracks, featurize)

        estimator = neighbors.KNeighborsClassifier(n_neighbors=1)
        estimator.fit(feature_vectors, numeric_labels)

        unlabeled_tracks = list(status_iterate(
            '{complete_count} unlabeled read',
            neuro_atlas_io.read_tracks(unlabeled_path),
            status_modulus=1000))

        unlabeled_vectors, num_bad_tracks_in_unlabeled = neuro_atlas_features.convert_to_features_unlabeled(
            enumerate(unlabeled_tracks), featurize)

        predictions = estimator.predict(unlabeled_vectors)
        unique_predictions = numpy.unique(predictions)
        indices = numpy.arange(len(unlabeled_tracks))
        # noinspection PyTypeChecker
        for prediction in unique_predictions:
            text_label = numeric_label_to_label[prediction]
            indices_matching_tracks = indices[prediction == predictions]
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            with open(os.path.join(output_path, text_label + '.txt'), 'w') as label_file:
                for index_matching in indices_matching_tracks:
                    neuro_atlas_io.write_track(label_file, unlabeled_tracks[index_matching])

    elif parsed_arguments.analysis == Analyses.individual_knn:

        featurize = functools.partial(
            neuro_atlas_features.extract_features_from_track, num_interpolated_points=num_points_in_features)

        numeric_labels, feature_vectors, numeric_label_to_label, num_bad_tracks_in_read = \
            neuro_atlas_features.convert_to_features(labeled_tracks, featurize)

        individual_labeled_tracks = neuro_atlas_io.read_all_tracks_with_labels(individual_path)

        individual_numeric_labels, individual_feature_vectors, numeric_label_to_label, num_bad_tracks_in_individual = \
            neuro_atlas_features.convert_to_features(individual_labeled_tracks, featurize, numeric_label_to_label)

        estimator = neighbors.KNeighborsClassifier(n_neighbors=1)
        t0 = time.time()
        estimator.fit(feature_vectors, numeric_labels)
        accuracy = estimator.score(individual_feature_vectors, individual_numeric_labels)

        print('Accuracy: {0}'.format(accuracy))
        print('Error: {0}'.format(1 - accuracy))
        print('Time(seconds): {0}'.format(time.time() - t0))

    elif parsed_arguments.analysis == Analyses.individual_svm:

        featurize = functools.partial(
            neuro_atlas_features.extract_features_from_track, num_interpolated_points=num_points_in_features)

        numeric_labels, feature_vectors, numeric_label_to_label, num_bad_tracks_in_read = \
            neuro_atlas_features.convert_to_features(labeled_tracks, featurize)

        individual_labeled_tracks = neuro_atlas_io.read_all_tracks_with_labels(individual_path)

        individual_numeric_labels, individual_feature_vectors, numeric_label_to_label, num_bad_tracks_in_individual = \
            neuro_atlas_features.convert_to_features(individual_labeled_tracks, featurize, numeric_label_to_label)

        estimator = SVC(kernel="linear")
        t0 = time.time()
        estimator.fit(feature_vectors, numeric_labels)
        # noinspection PyUnresolvedReferences
        accuracy = estimator.score(individual_feature_vectors, individual_numeric_labels)

        print('linear accuracy: {0}'.format(accuracy))
        print('Error: {0}'.format(1 - accuracy))
        print('Time(seconds): {0}'.format(time.time() - t0))

        estimator = SVC(kernel="rbf", gamma=0.0000082)
        t0 = time.time()
        estimator.fit(feature_vectors, numeric_labels)
        # noinspection PyUnresolvedReferences
        accuracy = estimator.score(individual_feature_vectors, individual_numeric_labels)

        print('rbf accuracy: {0}'.format(accuracy))
        print('Error: {0}'.format(1 - accuracy))
        print('Time(seconds): {0}'.format(time.time() - t0))

    elif parsed_arguments.analysis == Analyses.individual_logistic_regression:

        featurize = functools.partial(
            neuro_atlas_features.extract_features_from_track, num_interpolated_points=num_points_in_features)

        numeric_labels, feature_vectors, numeric_label_to_label, num_bad_tracks_in_read = \
            neuro_atlas_features.convert_to_features(labeled_tracks, featurize)

        individual_labeled_tracks = neuro_atlas_io.read_all_tracks_with_labels(individual_path)

        individual_numeric_labels, individual_feature_vectors, numeric_label_to_label, num_bad_tracks_in_individual = \
            neuro_atlas_features.convert_to_features(individual_labeled_tracks, featurize, numeric_label_to_label)

        estimator = LogisticRegression(multi_class="ovr", penalty='l2', solver='sag', max_iter=1000, n_jobs=-1)
        t0 = time.time()
        estimator.fit(feature_vectors, numeric_labels)
        accuracy = estimator.score(individual_feature_vectors, individual_numeric_labels)

        print('Accuracy: {0}'.format(accuracy))
        print('Error: {0}'.format(1 - accuracy))
        print('Time(seconds): {0}'.format(time.time() - t0))

    elif parsed_arguments.analysis == Analyses.distance_investigate:

        featurize = functools.partial(neuro_atlas_features.interpolate, num_resulting_points=num_points_in_features)

        numeric_labels, feature_vectors, numeric_label_to_label, num_bad_tracks_in_read = \
            neuro_atlas_features.convert_to_features(labeled_tracks, featurize)

        numeric_label_to_closest_same_different = neuro_atlas_analysis.distance_investigate(
            numeric_labels, feature_vectors)  # neuro_atlas_analysis.SupervisedQuickBundles.minimum_average_direct_flip)

        for label, numeric in sorted([(value, key) for key, value in numeric_label_to_label.iteritems()]):
            furthest_same, closest_different = numeric_label_to_closest_same_different[numeric]
            print label, furthest_same, closest_different

    elif parsed_arguments.analysis == Analyses.cv_supervised_quick_bundles:

        featurize = functools.partial(neuro_atlas_features.interpolate, num_resulting_points=num_points_in_features)

        numeric_labels, feature_vectors, numeric_label_to_label, num_bad_tracks_in_read = \
            neuro_atlas_features.convert_to_features(labeled_tracks, featurize)

        estimator = neuro_atlas_analysis.SupervisedQuickBundles(
            unsupervised_thresh=parsed_arguments.quick_bundles_threshold)
        t0 = time.time()
        accuracies = cross_validation.cross_val_score(
            estimator, feature_vectors, numeric_labels, cv=cross_validation.StratifiedKFold(
                numeric_labels, n_folds=10, shuffle=True))

        print('Accuracies:')
        print(accuracies)
        mean_accuracy = numpy.mean(accuracies)
        print('Mean: {0}, Std: {1}'.format(mean_accuracy, numpy.std(accuracies)))
        print('Mean Error: {0}'.format(1 - mean_accuracy))
        print('Time(seconds): {0}'.format(time.time() - t0))

    elif parsed_arguments.analysis == Analyses.individual_supervised_quick_bundles:

        featurize = functools.partial(neuro_atlas_features.interpolate, num_resulting_points=num_points_in_features)

        numeric_labels, feature_vectors, numeric_label_to_label, num_bad_tracks_in_read = \
            neuro_atlas_features.convert_to_features(labeled_tracks, featurize)

        individual_labeled_tracks = neuro_atlas_io.read_all_tracks_with_labels(individual_path)

        individual_numeric_labels, individual_feature_vectors, numeric_label_to_label, num_bad_tracks_in_individual = \
            neuro_atlas_features.convert_to_features(individual_labeled_tracks, featurize, numeric_label_to_label)

        estimator = neuro_atlas_analysis.SupervisedQuickBundles(
            unsupervised_thresh=parsed_arguments.quick_bundles_threshold)
        t0 = time.time()
        estimator.fit(feature_vectors, numeric_labels)
        accuracy = estimator.score(individual_feature_vectors, individual_numeric_labels)

        print('Accuracy: {0}'.format(accuracy))
        print('Error: {0}'.format(1 - accuracy))
        print('Time(seconds): {0}'.format(time.time() - t0))

    elif parsed_arguments.analysis == Analyses.cv_knn_vs_num_points:

        if num_points_in_features is None:
            num_points_in_features = range(3, 21)

        dict_to_plot = dict()
        # make this reusable by initializing a list
        labeled_tracks = list(labeled_tracks)

        for current_num_points in num_points_in_features:

            featurize = functools.partial(
                neuro_atlas_features.extract_features_from_track, num_interpolated_points=current_num_points)

            numeric_labels, feature_vectors, numeric_label_to_label, num_bad_tracks_in_read = \
                neuro_atlas_features.convert_to_features(labeled_tracks, featurize)

            estimator = neighbors.KNeighborsClassifier(n_neighbors=1)
            accuracies = cross_validation.cross_val_score(
                estimator, feature_vectors, numeric_labels, cv=cross_validation.StratifiedKFold(
                    numeric_labels, n_folds=10, shuffle=True))

            mean_accuracy = numpy.mean(accuracies)
            print 'Num points: {0}, Mean: {1}, Std: {2} Mean Error: {3}, Bad Tracks: {4}'.format(
                current_num_points, mean_accuracy, numpy.std(accuracies), 1 - mean_accuracy, num_bad_tracks_in_read)

            dict_to_plot[current_num_points] = accuracies
            neuro_atlas_visualizations.plot_dict(dict_to_plot, output_path)

    elif parsed_arguments.analysis == Analyses.individual_knn_vs_num_points:

        if num_points_in_features is None:
            num_points_in_features = range(3, 21)

        dict_to_plot = dict()
        # make this reusable by initializing a list
        labeled_tracks = list(labeled_tracks)
        individual_labeled_tracks = list(neuro_atlas_io.read_all_tracks_with_labels(individual_path))

        for current_num_points in num_points_in_features:

            featurize = functools.partial(
                neuro_atlas_features.extract_features_from_track, num_interpolated_points=current_num_points)

            numeric_labels, feature_vectors, numeric_label_to_label, num_bad_tracks_in_read = \
                neuro_atlas_features.convert_to_features(labeled_tracks, featurize)

            (individual_numeric_labels, individual_feature_vectors,
             numeric_label_to_label, num_bad_tracks_in_individual) = \
                neuro_atlas_features.convert_to_features(individual_labeled_tracks, featurize, numeric_label_to_label)

            estimator = neighbors.KNeighborsClassifier(n_neighbors=1)
            t0 = time.time()
            estimator.fit(feature_vectors, numeric_labels)
            accuracy = estimator.score(individual_feature_vectors, individual_numeric_labels)

            print 'Num points: {0}, Accuracy: {1}, Error: {2}, Bad Tracks Avg: {3}, Bad Tracks Individual: {4}'.format(
                current_num_points, accuracy, 1 - accuracy, num_bad_tracks_in_read, num_bad_tracks_in_individual)

            dict_to_plot[current_num_points] = accuracy
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

    elif parsed_arguments.analysis == Analyses.make_atlas:

        neuro_atlas_analysis.make_atlas(labeled_tracks, output_path)

    else:

        # noinspection PyPep8
        parser.error('Unknown analysis: {0}'.format(parsed_arguments.analysis))
