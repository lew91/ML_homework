from __future__ import print_function
import bisect

try:
    raw_input
except NameError:
    raw_input = input


def binary_search(sorted_collection, item):
    '''
    Pure implemention of binary search algorithm in Python

    Be careful collection must be ascending sorted, otherwise result will be unpredictable
    :param sorted_collection: some ascending sorted collection with comparable items
    :param item: item value to search
    :return: index of found item or None if time is not found
    '''
    left = 0
    right = len(sorted_collection) -1

    while left <=right:
        midpoint = left + (right -left) // 2
        current_item = sorted_collection[midpoint]
        if current_item == item:
            return midpoint
        else:
            if item < current_item:
                right = midpoint - 1
            else:
                left = midpoint + 1
    return None

def binary_search_std_lib(sorted_collection, item):
    """
    :param sorted_collection: some ascending sorted collection with comparable items
    :param item: item value to search
    :return: index of found item or None if item is not found
    """
    index = bisect.bisect_left(sorted_collection, item)
    if index != len(sorted_collection) and sorted_collection[index] == item:
        return index
    return None

def binary_search_by_recursion(sorted_collection, item,left, right):
    if (right < left):
        return None

    midpoint = left + (right - left) // 2

    if sorted_collection[midpoint] == item:
        return midpoint
    elif sorted_collection[midpoint] > item:
        return binary_search_by_recursion(sorted_collection, item, left, midpoint-1)
    else:
        return binary_search_by_recursion(sorted_collection, item, midpoint+1, right)

def __assert_sorted(collection):
    '''
    :param collection: collection
    :return: True if collection is ascending sorted
    :raise: py:class: 'ValueError' if collection is not ascending sorted
    '''
    if collection != sorted(collection):
        raise ValueError('Collection must be ascending sorted')
    return True

if __name__ == '__main__':
    import sys
    user_input = raw_input('Enter numbers separated by comma:\n').strip()
    collection = [ int(item) for item in user_input.split(',')]
    try:
        __assert_sorted(collection)
    except ValueError:
        sys.exit('Sequence must be ascending sorted to apply binary search')

    target_input = raw_input('Enter a single numbers to be found in the list:\n')
    target = int(target_input)
    result= binary_search(collection, target)
    if result is not None:
        print('{} found at positions: {}'.format(target, result))
    else:
        print('Not found')
        
