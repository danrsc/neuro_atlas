import os
import numpy
from neuro_atlas_util import status_iterate


def read_tracks(file_path):
    """
    A generator which yields each track as it is read
    :param file_path: A path to a text file containing track data
    :return: An Nx3 ndarray where each row is a set of (x, y, z) coordinates
    """

    with open(file_path, 'r') as track_file:
        for index_line, line in enumerate(track_file):

            coordinates = line.split()
            current_track = list()
            if len(coordinates) % 3 != 0:
                raise ValueError('Bad coordinate length: {0} ({1}, {2})'.format(
                    len(coordinates), file_path, index_line + 1))
            for index_point in range(len(coordinates) / 3):
                current_track.append(numpy.array([
                    float(coordinates[index_point * 3]),
                    float(coordinates[index_point * 3 + 1]),
                    float(coordinates[index_point * 3 + 2])
                ]))

            current_track = numpy.vstack(current_track)
            yield current_track


def write_track(output_file, track):

    for index_row in xrange(track.shape[0]):
        for index_coordinate in xrange(track.shape[1]):
            if index_coordinate > 0 or index_row > 0:
                output_file.write(' ')
            output_file.write('{0:.4f}'.format(track[index_row, index_coordinate]))
    output_file.write('\n')


def read_all_tracks_with_labels(input_dir):

    for file_name in status_iterate('{item} {fraction_complete:.2%} of files processed', os.listdir(input_dir)):
        label = os.path.splitext(file_name)[0]
        for index_track, track in enumerate(read_tracks(os.path.join(input_dir, file_name))):
            yield label, index_track, track
