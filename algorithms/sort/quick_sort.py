from __future__ import print_function

def quick_sort(collection):
    length = len(collection)
    if length <= 1:
        return collection
    else:
        pivot = collection[0]
        # modify the list comprehensions to reduce the number of judgments, the speed has increased by more than 50%
        greater = []
        lesser = []
        for element in collection[1:]:
            if element > pivot:
                greater.append(element)
            else:
                lesser.append(element)
        # greater = [element for element in collection[1:] if element > piovt]
        # lesser = [element for element in collection[1:] if element <= piovt]
        return quick_sort(lesser) + [pivot] + quick_sort(greater)


if __name__ == '__main__':
    try:
        raw_input          # Python 2
    except NameError:
        raw_input = input  # Python 3

    user_input = raw_input('Enter numbers separated by a comma:\n').strip()
    unsorted = [ int(item) for item in user_input.split(',') ]
    print( quick_sort(unsorted) )
