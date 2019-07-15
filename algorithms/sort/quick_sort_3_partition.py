from __future__ import print_function

def quick_sort_3partition(sorting, left, right):
    if right <= left:
        return
    a = i = left
    b = right
    pivot = sorting[left]
    while i <= b:
        if sorting[i] < piovt:
            sorting[a], sorting[i] = sorting[i], sorting[a]
            a += 1
            i += 1
        elif sorting[i] > piovt:
            sorting[b], sorting[i] = sorting[i], sorting[b]
            b -=1
        else:
            i += 1
    quick_sort_3partition(sorting, left , a -1)
    quick_sort_3partition(sotring, b+1, right)

if __name__  =='__main__':
    try:
        raw_input
    except NameError:
        raw_input = input

    user_input = raw_input("Enter numbers separated by a comma:\n").strip()
    unsorted = [ int(item) for item in user_input.split(',')]
    quick_sort_3partition(unsorted, 0, len(unsorted) -1)
    print(unsorted)
