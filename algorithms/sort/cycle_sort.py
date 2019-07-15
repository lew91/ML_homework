# algorithm
# 1. if the element is already at the correct position, do nothing
# 2. if it not, we will write it to its intended position. That position is inhabited by a different element b,
# which we then to move to its correct position. this process of displacing element to their
# correct position continues until an element is moved to the original position of a.
# this completes a cycle.

from __future__ import print_function

def cycle_sort(array):
    ans = 0

    # pass through the array to find cycle to rotate.
    for cycleStart in range(0, len(array) - 1):
        item = array[cycleStart]

        # finding the position for putting the item.
        pos = cycleStart
        for i in range(cycleStart + 1, len(array)):
            if array[i] < item:
                pos += 1

        # if the item is already present-not a cycle
        if pos == cycleStart:
            continue

        # otherwise, put the item there or right after any duplicates.
        while item == array[pos]:
            pos += 1
        array[pos], item = item, array[pos]
        ans += 1

        # Rotate the rest of the cycle
        while pos != cycleStart:

            # find where to put the item.
            pos = cycleStart
            for i in range(cycleStart + 1, len(array)):
                if array[i] < item:
                    pos += 1

            # put the item there or right after any duplicates.
            while item == array[pos]:
                pos += 1
            array[pos], item = item, array[pos]
            ans += 1
    return ans

# main code starts here
if __name__ == '__main__':
    try:
        raw_input
    except  NameError:
        raw_input = input

    user_input = raw_input('Enter numbers separated by a comma:\n')
    unsorted = [int(item) for item in user_input.split(',')]

    n = len(unsorted)
    cycle_sort(unsorted)

    print("After sort :")
    for i in range(0, n):
        print(unsorted[i], end = ' ')
