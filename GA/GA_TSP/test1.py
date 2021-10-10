import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import random
matplotlib.rcParams['font.family'] = 'STSong'

# 载入数据
city_name = []
city_condition = []
with open('data.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split('\n')[0]
        line = line.split(',')
        city_name.append(line[0])
        city_condition.append([float(line[1]), float(line[2])])
city_condition = np.array(city_condition)

# 展示地图
plt.scatter(city_condition[:, 0], city_condition[:, 1])
plt.show()

# 距离矩阵
city_count = len(city_name)
Distance = np.zeros([city_count, city_count])
for i in range(city_count):
    for j in range(city_count):
        if i > j:
            Distance[i, j] = Distance[j, i]
            continue
        d = city_condition[i]-city_condition[j]
        Distance[i, j] = math.sqrt(sum(d**2))

# 种群数
count = 30

# 改良次数
improve_count = 10000

# 进化次数
itter_time = 300

# 设置强者的定义概率，即种群前30%为强者
retain_rate = 0.3

# 设置弱者的存活概率
random_select_rate = 0.5

# 变异率
mutation_rate = 0.1

# 设置起点，从上海出发
origin = 2
index = list(range(city_count))
index.remove(origin)


# 根据输入的x的序列中每个元素的大小产生一个序列
def get_path(x):
    graded = [[x[i], index[i]] for i in range(len(x))]
    # index是城市去掉起始点的索引
    graded_index = [t[1] for t in sorted(graded)]
    return graded_index


# 总距离
def get_total_distance(x):
    graded_index = get_path(x)
    distance = 0
    distance += Distance[origin][graded_index[0]]
    for i in range(len(graded_index)):
        if i == len(graded_index)-1:
            distance += Distance[origin][graded_index[i]]
        else:
            distance += Distance[graded_index[i]][graded_index[i+1]]
    return distance


# 通过交换序列中两数比较交换前后总距离大小进行序列的改良
def improve(x):
    distance = get_total_distance(x)
    for _ in range(improve_count):
        # randint [a,b]生成a到b之间包括a，b的整数
        u = random.randint(0, len(x)-1)
        v = random.randint(0, len(x)-1)
        if u != v:
            new_x = x.copy()
            new_x[u], new_x[v] = new_x[v], new_x[u]
            new_distance = get_total_distance(new_x)
            if new_distance < distance:
                distance = new_distance
                x = new_x.copy()
    return x


# 自然选择
def selection(population):
    """
    选择
    先对适应度从大到小排序，选出存活的染色体
    再进行随机选择，选出适应度虽然小，但是幸存下来的个体
    """
    # 对总距离从小到大进行排序
    graded = [[get_total_distance(x), x] for x in population]
    graded = [t[1] for t in sorted(graded)]
    # 选出适应性强的染色体
    retain_length = int(len(graded) * retain_rate)
    parents = graded[:retain_length]
    # 选出适应性不强，但是幸存的染色体
    for chromosome in graded[retain_length:]:
        if random.random() < random_select_rate:
            parents.append(chromosome)
    return parents


# 交叉繁殖
def crossover(parents):
    # 生成子代的个数,以此保证种群数量稳定
    target_count = count-len(parents)
    # 孩子列表
    children = []
    while len(children) < target_count:
        male_index = random.randint(0, len(parents) - 1)
        female_index = random.randint(0, len(parents) - 1)
        if male_index != female_index:
            male = parents[male_index]
            female = parents[female_index]
            mask = random.randint(1, len(male)-1)
            child1 = male[:mask]+female[mask:]
            child2 = female[:mask]+male[mask:]
            children.append(child1)
            children.append(child2)
    return children


# 变异
def mutation(children):
    new_children = []
    for child in children:
        if random.random() < mutation_rate:
            child = [t+random.random() for t in child]
        new_children.append(child)
    return new_children


# 得到最佳输出结果
def get_result(population):
    graded = [[get_total_distance(x), x] for x in population]
    graded = sorted(graded)
    return graded[0][0], get_path(graded[0][1])


# 使用改良圈算法初始化种群
population = []
for i in range(count):
    # 随机生成个体
    x = [random.random() for _ in range(city_count-1)]
    x = improve(x)
    population.append(x)

# 开始迭代
register = []
distance, result_path = get_result(population)
for _ in range(itter_time):
    # 选择繁殖个体群
    parents = selection(population)
    # 交叉繁殖
    children = crossover(parents)
    # 变异操作
    children = mutation(children)
    # 更新种群
    population = parents+children

    distance, result_path = get_result(population)
    register.append(distance)
print(distance)
print(result_path)

# 画图，可视化
result_path = [origin]+result_path+[origin]
X = []
Y = []
for index in result_path:
    X.append(city_condition[index, 0])
    Y.append(city_condition[index, 1])

plt.plot(X, Y, '-o')
plt.show()

plt.plot(list(range(len(register))), register)
plt.show()
