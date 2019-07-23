import numpy as np
import math
import matplotlib.pyplot as plt
import random
from tensorboardX import SummaryWriter

K = 2
LAMBDA = 0.3

#生成数据
class0 = 50
class1 = 60
class2 = 55
class3 = 20
class4 = 10
class5 = 35
class6 = 40
class7 = 25

# color0='blue'
# color1='red'
# color2='green'
# color3='yellow'
# color4='cyan'
# color5='gray'
# color6='orange'
# color7='pink'

# color0='blue'
# color1='blueviolet'
# color2='brown'
# color3='burlywood'
# color4='cadetblue'
# color5='coral'
# color6='chocolate'
# color7='blanchedalmond'

# color0 = '#0000FF'
# color1 = '#8A2BE2'
# color2 = '#A52A2A'
# color3 = '#DEB887'
# color4 = '#5F9EA0'
# color5 = '#FF7F50'
# color6 = '#D2691E'
# color7 = '#FFEBCD'

color0 = '#0000FF'
color1 = '#00FFFF'
color2 = '#008000'
color3 = '#FFC0CB'
color4 = '#EE82EE'
color5 = '#FFA500'
color6 = '#FF0000'
color7 = '#FFFF00'
colorlist = [color0, color1, color2, color3, color4, color5, color6, color7]

class_num = 8

total_num = class0 + class1 + class2 + class3 + class4 + class5 + class6 + class7

dimension = 2  #数据特征维度

writer = SummaryWriter(comment='ColorAssignment GA', log_dir='../TensorBoard')

dataset = list()


class Dot:  #该类用来标识每个节点的类别
    def __init__(self, _x, _y):
        self.x = _x
        self.label = _y


data0 = []
data1 = []
data2 = []
data3 = []
data4 = []
data5 = []
data6 = []
data7 = []

#由于每一类数据都需要被单独控制，所以只好这样分开写
for i in range(class0):
    X = 1.8 * np.random.randn(1, dimension) - 10
    X[0, 0] = 3 * X[0, 0] + 40
    data0.append(X.squeeze())

for i in range(class1):
    X = 4 * np.random.randn(1, dimension) + 2
    X[0, 1] = 2 * X[0, 1]
    X[0, 0] = X[0, 0] + 15
    data1.append(X.squeeze())

for i in range(class2):
    X = 4.5 * np.random.randn(1, dimension) + 25
    X[0, 0] = X[0, 0] - 20
    X[0, 0] = 3 * X[0, 0]
    data2.append(X.squeeze())

for i in range(class3):
    X = 3 * np.random.randn(1, dimension) + 10
    X[0, 0] = 1.5 * X[0, 0] - 15
    data3.append(X.squeeze())

for i in range(class4):
    X = 1.5 * np.random.randn(1, dimension) + 3
    X[0, 0] = 3 * X[0, 0] + 20
    X[0, 1] = 2 * X[0, 1]
    data4.append(X.squeeze())

for i in range(class5):
    X = 5 * np.random.randn(1, dimension) - 5
    X[0, 0] = 2 * X[0, 0] - 4
    data5.append(X.squeeze())

for i in range(class6):
    X = 3 * np.random.randn(1, dimension) + 4
    X[0, 1] = 3 * X[0, 1] + 1
    data6.append(X.squeeze())

for i in range(class7):
    X = 2 * np.random.randn(1, dimension) - 9
    X[0, 0] = 1.4 * X[0, 0] - 7
    data7.append(X.squeeze())

data0 = np.array(data0)
data1 = np.array(data1)
data2 = np.array(data2)
data3 = np.array(data3)
data4 = np.array(data4)
data5 = np.array(data5)
data6 = np.array(data6)
data7 = np.array(data7)

plt.scatter(data0[:, 0], data0[:, 1], c=color0)
plt.scatter(data1[:, 0], data1[:, 1], c=color1)
plt.scatter(data2[:, 0], data2[:, 1], c=color2)
plt.scatter(data3[:, 0], data3[:, 1], c=color3)
plt.scatter(data4[:, 0], data4[:, 1], c=color4)
plt.scatter(data5[:, 0], data5[:, 1], c=color5)
plt.scatter(data6[:, 0], data6[:, 1], c=color6)
plt.scatter(data7[:, 0], data7[:, 1], c=color7)

plt.show()


def tag_list(data, tag):
    result = []
    for d in data:
        result.append(Dot(d, tag))

    return result


def LAB_distance(rgb_1, rgb_2):
    R_1 = int(rgb_1[1:3], 16)
    G_1 = int(rgb_1[3:5], 16)
    B_1 = int(rgb_1[5:], 16)
    R_2 = int(rgb_2[1:3], 16)
    G_2 = int(rgb_2[3:5], 16)
    B_2 = int(rgb_2[5:], 16)
    rmean = (R_1 + R_2) / 2
    R = R_1 - R_2
    G = G_1 - G_2
    B = B_1 - B_2
    return math.sqrt((2 + rmean / 256) * (R**2) + 4 * (G**2) +
                     (2 + (255 - rmean) / 256) * (B**2))


def LUM_distance(rgb_1, rgb_2):
    R_1 = int(rgb_1[1:3], 16)
    G_1 = int(rgb_1[3:5], 16)
    B_1 = int(rgb_1[5:], 16)
    R_2 = int(rgb_2[1:3], 16)
    G_2 = int(rgb_2[3:5], 16)
    B_2 = int(rgb_2[5:], 16)

    Y_1 = ((R_1 * 299) + (G_1 * 587) + (B_1 * 114)) / 1000
    Y_2 = ((R_2 * 299) + (G_2 * 587) + (B_2 * 114)) / 1000

    return Y_1 - Y_2


# test=LAB_distance('#FE4321','#123456')

#Optimization
dataset.extend(tag_list(data0, 0))
dataset.extend(tag_list(data1, 1))
dataset.extend(tag_list(data2, 2))
dataset.extend(tag_list(data3, 3))
dataset.extend(tag_list(data4, 4))
dataset.extend(tag_list(data5, 5))
dataset.extend(tag_list(data6, 6))
dataset.extend(tag_list(data7, 7))


def distance(x, y):
    return math.sqrt(sum(pow(x.x - y.x, 2)))


#找到下标即可
def initNeighbor():
    result = []
    for x in dataset:
        dis = []
        for point in dataset:
            if (point == x):
                dis.append(100000)
            else:
                dis.append(distance(x, point))

        dis = np.array(dis)
        result.append(np.argsort(dis)[:K])
    return result  #ATTENTION!!!!返回的是dataset中的下标


neighbors = initNeighbor()  #将邻居信息计算一遍后储存，避免重复计算


def findNeighbor(x, neighbors):
    return neighbors[x]


def alpha(x, neighbor, colorindex):  #neighbor里是下标
    s = 0
    for i in neighbor:
        s += LAB_distance(colorlist[colorindex[x.label]],
                          colorlist[colorindex[dataset[i].label]]) / distance(
                              x, dataset[i])

    return s / K


def beta(x, neighbor, colorindex):
    s = 0
    for i in neighbor:
        s += LUM_distance(colorlist[colorindex[x.label]],
                          '#FFFFFF') * (b(x, neighbor) - a(x, neighbor))

    return s / K


def a(x, neighbor):
    s = 0
    for i in neighbor:
        if (dataset[i].label == x.label):
            s += 1.0 / distance(x, dataset[i])

    return s / K


def b(x, neighbor):
    s = 0
    for i in neighbor:
        if (not dataset[i].label == x.label):
            s += 1.0 / distance(x, dataset[i])

    return s / K


def obj_fun(colorindex):
    obj_sum = 0
    for i, x in enumerate(dataset):
        neighbor = findNeighbor(i, neighbors)
        obj_sum += LAMBDA * alpha(x, neighbor, colorindex) + (
            1 - LAMBDA) * beta(x, neighbor, colorindex)

    return obj_sum


#GA
#target is colorindex
GA_INIT = 100
CROSSOVER = 0.6 
MUTATION = 0.05  #防止陷入局部最优解
EPOCH = 200
LEAVEBEST = 2

index_list = []
for i in range(GA_INIT):
    index_list.append(np.random.permutation(8))

# max=0
# final_index=[0,1,2,3,4,5,6,7]
# for index in index_list:
#     result=obj_fun(index)
#     if result>max:
#         max=result
#         final_index=index


def GA_select(genes):
    result = []
    obj_sum = 0
    wheel = []
    next_gen = []

    for index in genes:
        tmp = obj_fun(index)
        obj_sum += tmp
        result.append(tmp)

    tmp_list = np.argsort(result)[-LEAVEBEST:]  #argsort是从小到大排列
    for index in tmp_list:  #每次确保保留最好的两个
        next_gen.append(genes[index])

    for obj in result:
        wheel.append(obj / obj_sum)

    for i in range(GA_INIT - LEAVEBEST):
        rand = random.random()
        w_sum = 0
        for w in range(len(wheel)):
            w_sum += wheel[w]
            if w_sum >= rand:
                next_gen.append(genes[w])
                break

    return next_gen


def GA_crossover(gene1, gene2):
    rand1 = np.random.randint(0, 7)
    rand2 = np.random.randint(2, 6)  #限制长度为2-5

    gene1_sub = []
    gene2_sub = []
    for i in range(rand1, min([rand1 + rand2, 8])):
        gene1_sub.append(gene1[i])
        gene2_sub.append(gene2[i])

    sub_tag1 = 0
    sub_tag2 = 0
    for i in range(len(gene1)):
        if i >= rand1 and i < min([rand1 + rand2, 8]):
            continue

        if gene1[i] in gene2_sub:
            sub_count = 0
            for possible_sub1 in gene1_sub:
                if not possible_sub1 in gene2_sub:
                    sub_count += 1
                    if sub_count > sub_tag1:
                        gene1[i] = possible_sub1
                        sub_tag1 += 1
                        break

        if gene2[i] in gene1_sub:
            sub_count = 0
            for possible_sub2 in gene2_sub:
                if not possible_sub2 in gene1_sub:
                    sub_count += 1
                    if sub_count > sub_tag2:
                        gene2[i] = possible_sub2
                        sub_tag2 += 1
                        break

    for i in range(rand1, min([rand1 + rand2, 8])):
        gene1[i] = gene2_sub[i - rand1]
        gene2[i] = gene1_sub[i - rand1]


def GA_mutation(gene):
    rand1 = np.random.randint(0, 8)
    rand2 = np.random.randint(0, 8)
    if rand1 == rand2:  #避免随机数一样
        if rand2 == 0:
            rand2 = 7
        else:
            rand2 -= 1

    gene[rand1], gene[rand2] = gene[rand2], gene[rand1]


init_obj = obj_fun([0, 1, 2, 3, 4, 5, 6, 7])
print("-1 :", init_obj)  #起始得分，先打印一下
writer.add_scalar('E(tau)', init_obj, 0)

for epoch in range(EPOCH):
    index_list = GA_select(index_list)
    max_obj = obj_fun(index_list[LEAVEBEST - 1])
    print(epoch, ":", max_obj)  #根据GA_select放在第二个的一定是最好的

    writer.add_scalar('E(tau)', max_obj, epoch+1)

    tag = 0
    pre = []
    tmp_index_list = []
    for index in index_list:
        if random.random() < CROSSOVER:
            if tag == 0:
                tag = 1
                pre = index
            else:
                tag = 0
                tmp_index_list.append(
                    pre.copy())  #接下来要对这两个基因型杂交，所以在那之前先保存好原基因型
                tmp_index_list.append(index.copy())
                GA_crossover(pre, index)

    for index in tmp_index_list:
        index_list.append(index)

    for index in index_list:
        if random.random() < MUTATION:  #该index需要mutation
            GA_mutation(index)

max = 0
final_index = [0, 1, 2, 3, 4, 5, 6, 7]
for index in index_list:
    result = obj_fun(index)
    if result > max:
        max = result
        final_index = index

#replot
plt.scatter(data0[:, 0], data0[:, 1], c=colorlist[final_index[0]])
plt.scatter(data1[:, 0], data1[:, 1], c=colorlist[final_index[1]])
plt.scatter(data2[:, 0], data2[:, 1], c=colorlist[final_index[2]])
plt.scatter(data3[:, 0], data3[:, 1], c=colorlist[final_index[3]])
plt.scatter(data4[:, 0], data4[:, 1], c=colorlist[final_index[4]])
plt.scatter(data5[:, 0], data5[:, 1], c=colorlist[final_index[5]])
plt.scatter(data6[:, 0], data6[:, 1], c=colorlist[final_index[6]])
plt.scatter(data7[:, 0], data7[:, 1], c=colorlist[final_index[7]])

plt.show()
