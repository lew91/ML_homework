import numpy as np
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


import regTrees



def reDraw(tolS, tolN):
    """
    绘制原始数据的散点图以及拟合数据的曲线图

    Parameters
    -----------
    tolS : 允许的误差下降值
    tolN : 诶分的最小样本值

    Returns
    ------------
    None
    """
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)

    if chkBtnVar.get():
        if tolN < 2:
            tolN = 2

        myTree = regTrees.createTree(reDraw.rawDat, regTrees.modelLeaf,
                                     regTrees.modelErr, (tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat,
                                       regTrees.modelTreeEval)

    else:
        myTree = regTrees.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = regTrees.createForeCast(myTree, reDraw.testDat)


    reDraw.a.scatter(reDraw.rawDat[:, 0].tolist(), reDraw.rawDat[:, 1].tolist(),
                     s=5)
    reDraw.a.plot(reDraw.testDat, yHat, 'b', linewidth=2.0)

    reDraw.canvas.show()


def getInputs():
    try:
        tolN = int(tolNentry.get())
    except:
        tolN = 10
        print("enter integer for tolN")
        tolNentry.delete(0, END)
        tolNentry.insert(0, '10')

    try:
        tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print("enter Folat for tolS")
        tolSentry.delete(0, END)
        tolSentry.insert(0, '1.0')

    return tolN, tolS 


def drawNewTree():
    tolN, tolS = getInputs()

    reDraw(tolS, tolN)


# 创建窗口
root = tk.Tk()
tk.Label(root, text="Plot place Holder").grid(row=0, columnspan=3)
tk.Label(root, text="tolN").grid(row=1, column=0)

tolNentry = tk.Entry(root)
tolNentry.grid(row=1, column=1)
tolNentry.insert(0, '10')

tk.Label(root, text="tolS").grid(row=2, column=0)
tolSentry = tk.Entry(root)
tolSentry.grid(row=2, column=1)
tolSentry.insert(0, '1.0')

tk.Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)

chkBtnVar = tk.IntVar()
chkBtn = tk.Checkbutton(root, text="Model Tree", variable=chkBtnVar)
chkBtn.grid(row=3, column=0, columnspan=2)

reDraw.rawDat = np.mat(regTrees.loadDataSet('sine.txt'))
reDraw.testDat = np.arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
reDraw(1.0, 10)

root.mainloop()





