import argparse
import os
import numpy
import neuro_atlas_analysis
import neuro_atlas_visualizations


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
                        choices=['cv_knn', 'cv_knn_vs_num_points', 'label_counts', 'find_duplicates'],
                        required=True)
    parser.add_argument('--num_points_in_features',
                        help='How many points to use to characterize a track. For cv_knn this should be an integer. '
                             'For cv_knn_vs_num_points this should be a comma-delimited list of integers. '
                             'If not provided, defaults to 3 for cv_knn, and range(3, 20) for cv_knn_vs_num_points')

    parsed_arguments = parser.parse_args()
    input_path = os.path.abspath(parsed_arguments.input_path)
    output_path = os.path.abspath(parsed_arguments.output_path)

    if parsed_arguments.analysis == 'cv_knn':

        num_points = getattr(parsed_arguments, 'num_points_in_features', None)
        if num_points is not None:
            try:
                num_points = int(num_points)
            except (TypeError, ValueError):
                parser.error('Bad argument for num_points_in_features {0}'.format(
                    parsed_arguments.num_points_in_features))
                exit(1)

        accuracies, num_bad_tracks = neuro_atlas_analysis.cv_knn(input_path, num_points_in_features=num_points)
        print('Accuracies:')
        print(accuracies)
        mean_accuracy = numpy.mean(accuracies)
        print('Mean: {0}, Std: {1}'.format(mean_accuracy, numpy.std(accuracies)))
        print('Mean Error: {0}'.format(1 - mean_accuracy))

    elif parsed_arguments.analysis == 'cv_knn_vs_num_points':

        num_points = getattr(parsed_arguments, 'num_points_in_features', None)
        if num_points is not None:
            try:
                num_points = [int(x.strip()) for x in num_points.split(',') if len(x.strip()) > 0]
            except (TypeError, ValueError):
                parser.error('Bad argument for num_points_in_features {0}'.format(
                    parsed_arguments.num_points_in_features))
                exit(1)

        num_points_to_result = neuro_atlas_analysis.cv_knn_vs_num_points(input_path, num_points_list=num_points)
        dict_to_plot = dict()
        for x, (accuracies, num_bad_tracks) in sorted(num_points_to_result.iteritems(), key=lambda (k, r): k):
            mean_accuracy = numpy.mean(accuracies)
            print 'Num points: {0}, Mean: {1}, Std: {2} Mean Error: {3}, Bad Tracks: {3}'.format(
                x, mean_accuracy, numpy.std(accuracies), 1 - mean_accuracy, num_bad_tracks)
            dict_to_plot[x] = accuracies
        neuro_atlas_visualizations.plot_dict(dict_to_plot, output_path)

    elif parsed_arguments.analysis == 'label_counts':

        label_counts = neuro_atlas_analysis.label_counts(input_path)
        sorted_counts = sorted(label_counts.iteritems(), key=lambda (s_l, s_c): s_c)
        total = sum([c for l, c in sorted_counts])
        with open(output_path, 'w') as output_file:
            for label, count in sorted_counts:
                output_file.write('{0}\t{1}\n'.format(label, count))
                print ('{0}: {1} ({2:.2%})'.format(label, count, float(count) / total))

    elif parsed_arguments.analysis == 'find_duplicates':

        num_duplicates = neuro_atlas_analysis.find_duplicates(input_path, output_path)
        print '{0} duplicates found'.format(num_duplicates)

    else:

        parser.error('Unknown analysis: {0}'.format(parsed_arguments.analysis))
