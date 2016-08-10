import numpy
import datetime


# this is a neat trick from http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
def unique_rows(array, return_index=False, return_inverse=False, return_counts=False):
    """
    Similar to numpy.unique, but operating on the rows of a matrix
    :param array: Input array.
    :param return_index: If True, also return the indices of the array that result in the unique array
    :param return_inverse: If True, also return the indices of the unique array that can be used to reconstruct array
    :param return_counts: If True, also return the number of times each unique row occurs in array
    :return: The unique array
    """

    b = numpy.ascontiguousarray(array).view(numpy.dtype((numpy.void, array.dtype.itemsize * array.shape[1])))
    _, indices, inverses, counts = numpy.unique(b, return_index=True, return_inverse=True, return_counts=True)

    unique_array = array[indices]
    if not return_index and not return_inverse and not return_counts:
        return unique_array

    result = [unique_array]
    if return_index:
        result.append(indices)
    if return_inverse:
        result.append(inverses)
    if return_counts:
        result.append(counts)

    return tuple(result)


def __status_update(format_string, start_time, complete_count, item, item_count):
    """
    Helper function for status_iterate. Builds the keyword arguments for formatting
    :param format_string:
        If a string, then .format is called on the string with keyword arguments and the result is printed.
        Otherwise, treated as a callable and keyword args is passed to the callable. In this case the callable
        is responsible for printing (or whatever else it wants to do).
    :param start_time: The time at which iteration was started
    :param complete_count: How many items have been yielded
    :param item: The current item
    :param item_count: The total count of items, or None if the total count is not available.
    """

    duration = None
    avg_seconds = None
    average_time = None
    end_time = datetime.datetime.now()
    if start_time is not None:
        duration = end_time - start_time
        if complete_count is not None and complete_count > 0:
            avg_seconds = duration.total_seconds() / float(complete_count)
            average_time = datetime.timedelta(seconds=avg_seconds)
    remaining_time = None
    fraction_complete = None
    if item_count is not None:
        if item_count == 0:
            remaining_time = datetime.timedelta(seconds=0)
            fraction_complete = 1.0
        else:
            if avg_seconds is not None:
                remaining_time = datetime.timedelta(seconds=(avg_seconds * (item_count - complete_count)))
            fraction_complete = float(complete_count) / item_count

    kwargs = {
        'complete_count': complete_count,
        'item': item,
        'count': item_count,
        'fraction_complete': fraction_complete,
        'start_time': start_time,
        'end_time': end_time,
        'duration': duration,
        'average_time': average_time,
        'remaining_time': remaining_time
    }

    # allow format_string to be a callable
    if not isinstance(format_string, type('')):
        format_string(**kwargs)
        return

    print(format_string.format(**kwargs))


def status_iterate(format_string, iterable, status_modulus=1, item_count=None):
    """
    Yields each item of an iterable, and prints status updates
    :param format_string:
        If a string, then .format is called on the string with keyword arguments and the result is printed.
        Otherwise, treated as a callable and keyword args is passed to the callable. In this case the callable
        is responsible for printing (or whatever else it wants to do).
    :param iterable: The underlying iterable
    :param status_modulus: Status is printed every status_modulus items.
    :param item_count: The total number of items in the iterable, or None if the count is not available
    :return: Yields each item in the iterable
    """

    start_time = datetime.datetime.now()
    if item_count is None:
        try:
            item_count = len(iterable)
        except (TypeError, AttributeError):
            pass

    index = None
    item = None
    for index, item in enumerate(iterable):

        yield item

        if status_modulus == 1 or index % status_modulus == 0:
            __status_update(format_string, start_time, index + 1, item, item_count)

    # give a final update on completion, unless we just updated
    if index is None or index % status_modulus != 0:
        __status_update(format_string, start_time, index, item, item_count)


def upper_triangle_iterate(num_items, get_item=None, offset=0):
    """
    Walks the upper triangle of elements of a virtual pairwise matrix of items, yielding the pairs in this upper
    triangle
    :param num_items: The number of items in the iterable (not the number of pairs)
    :param get_item: Callable. Given index i, returns the item at index i
    :param offset: Which diagonal to start on. Defaults to the main diagonal
    :return: yields each pair of items in the upper triangle
    """
    for index_1 in range(num_items):
        item_1 = get_item(index_1) if get_item is not None else index_1
        for index_2 in range(index_1 + offset, num_items):
            item_2 = get_item(index_2) if get_item is not None else index_2
            yield item_1, item_2


def upper_triangle_count(num_items, offset=0):
    """
    Gets the number of items in the upper triangle
    :param num_items: The number of items in the iteration (not the number of pairs)
    :param offset: Which diagonal the upper triangle iteration will start on.
    :return: The number of items in the iteration
    """
    return (num_items - offset) * (num_items - offset + 1) / 2
