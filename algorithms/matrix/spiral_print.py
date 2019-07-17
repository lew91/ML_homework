"""
Matrix must satisfy below conditions
i) matrix should be only one or two dimensional
ii) column of all the row should be equal
"""


def checkMatrix(a):
    # must be
    if type(a) == list and len(a) > 0:
        if type(a[0]) == list:
            prevlen = 0
            for i in a:
                if prevlen == 0:
                    prevlen = len(i)
                    result = True
                elif prevlen == len(i):
                    result = True
                else:
                    result = False

        else:
            result = True

    else:
        result = False
    return result


def spiralPrint(a):
    if checkMatrix(a) and len(a) > 0:
        matRow = len(a)
        if type(a[0]) == list:
            matCol = len(a[0])
        else:
            for dat in a:
                print(dat),
            return

        # horizontal printing increasing
        for i in range(0, matCol):
            print(a[0][i])
        # vertical printing down
        for i in range(1, matRow):
            print(a[i][matCol - 1])
        # horizontal printing decreasing
        if matRow > 1:
            for i in range(matCol - 2, -1, -1):
                print(a[matRow - 1][i]),
        # vertical printing up
        for i in range(matRow - 2, 0, -1):
            print(a[i][0]),
        remainMat = [row[1:matCol - 1] for row in a[1:matRow - 1]]
        if len(remainMat) > 0:
            spiralPrint(remainMat)
        else:
            return
    else:
        print("Not a valid matrix")
        return


# driver code
a = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
spiralPrint(a)

