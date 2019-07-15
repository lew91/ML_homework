# 将所有待比较数值（正整数）统一为同样的数字长度，数字较短的数前面补零。然后，从最低位开始，依次进行一次排序。这样从最低位排序一直到最高位排序完成以后，数列就变成一个有序序列。

# 基数排序的方式可以采用LSD（Least significant digital）或MSD（Most significant digital），LSD的排序方式由键值的最右边开始，而MSD则相反，由键值的最左边开始。

def radix_sort(lst):
    RADIX = 10
    placement = 1

    # get the maximum number
    max_digit = max(lst)

    while placement < max_digit:
        # declare and initialize buckets
        buckets = [list() for _ in range( RADIX)]
       #  print(buckets)

        # split lst between lists
        for i in lst:
            tmp = int((i / placement) % RADIX)
            buckets[tmp].append(i)
           #  print(buckets)

        # empty lists into lst array
        a = 0
        for b in range(RADIX):
            buck = buckets[b]
           #  print(buck)
            for i in buck:
                lst[a] = i
               #  print(lst[a])
                a += 1
           #  print(lst)


        # move to next
        placement *= RADIX

    return lst

user_input =input('Enter numbers separated by a comma:').strip()
unsorted =[int(item) for item in user_input.split(',')]
print(radix_sort(unsorted))
