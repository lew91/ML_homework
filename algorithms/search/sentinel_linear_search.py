def sentinel_linear_search(sequence, target):
    index = 0
    while sequence[index] != target:
        index += 1

    sequence.pop()

    if index == len(sequence):
        return None

    return index

if __name__ == '__main__':
    try:
        raw_input
    except NameError:
        raw_input = input

    user_input = raw_input('Enter numbers separated by comma:\n').strip()
    sequence = [int(item) for item in user_input.split(',')]
    target_input = raw_input('Enter a single number to be found in the list:\n')
    target = int(target_input)
    result = sentinel_linear_search(sequence, target)
    if result is not None:
        print('{} found at positions: {}'.format(target, result))
    else:
        print('Not found')
