import numpy


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
                raise ValueError('Bad coordinate length: {0}'.format(len(coordinates)))
            for index_point in range(len(coordinates) / 3):
                current_track.append(numpy.array([
                    float(coordinates[index_point * 3]),
                    float(coordinates[index_point * 3 + 1]),
                    float(coordinates[index_point * 3 + 2])
                ]))

            current_track = numpy.vstack(current_track)
            yield current_track
