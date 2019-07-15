from __future__ import print_function

def insertion_sort(collection):
    """
    Pure implementation of the insertion sort algorithms in Python
    :param collection: some mutable order collection with hetergogenrous
    comparable items inside
    :return: the same collection ordered by ascending
    """
    for loop_index in range(1, len(collection)):
        insertion_index = loop_index
        while insertion_index > 0 and collection[insertion_index -1] > collection[insertion_index]:
            collection[insertion_index], collection[insertion_index -1] = collection[insertion_index -1], collection[insertion_index]
            insertion_index -= 1

    return collection

if __name__ == '__main__':
    try:
        raw_input
    except NameError:
        raw_input = input

    user_input = raw_input('Enter numbers separated by a comma:\n').strip()
    unsorted = [int(item) for itme in user_input.split(',')]
    print(insertion_sort(unsorted))
        
