import numpy as np
import random
import time
# 多策略随机选择版本

class GeneticAlgTSP:
    def __init__(self, filename, pop_size=100, mutation_rate=0.5):
        """
        构造函数：读取TSP数据集并初始化种群
        :param filename: TSP数据集文件名，例如 "dj38.tsp"
        """
        # 读取TSP文件并存储城市坐标
        self.cities = self._load_tsp_data(filename)
        self.num_cities = len(self.cities)

        # 初始化遗传算法参数
        self.pop_size = pop_size  # 种群大小
        self.mutation_rate = mutation_rate  # 变异概率
        self.elite_size = int(0.1 * self.pop_size)  # 精英个体数量

        # 初始化种群：随机生成pop_size条路径
        self.population = [self._create_random_path() for _ in range(self.pop_size)]
        self.best_solution = None
        self.best_distance = float("inf")

        # 定义可选的操作函数列表
        self.selection_methods = [
            self._tournament_selection,
            self._roulette_selection,
            self._sus_selection,
        ]
        self.crossover_methods = [
            self._ox_crossover,
            self._pmx_crossover,
            self._cx_crossover,
        ]
        self.mutation_methods = [
            self._swap_mutation,
            self._insertion_mutation,
            self._inversion_mutation,
        ]

    def _load_tsp_data(self, filename):
        """读取TSP文件中的城市坐标"""
        cities = []
        with open(filename, "r") as f:
            lines = f.readlines()
            start_reading = False
            for line in lines:
                if line.startswith("NODE_COORD_SECTION"):
                    start_reading = True
                    continue
                if line.startswith("EOF"):
                    break
                if start_reading:
                    _, x, y = line.strip().split()
                    cities.append([float(x), float(y)])
        return np.array(cities)

    def _create_random_path(self):
        """生成一条随机路径"""
        path = list(range(1, self.num_cities + 1))
        random.shuffle(path)
        return path

    def _calculate_distance(self, path):
        """计算路径总距离"""
        total_dist = 0
        for i in range(len(path)):
            city1 = self.cities[path[i] - 1]
            city2 = self.cities[path[(i + 1) % self.num_cities] - 1]
            total_dist += np.linalg.norm(city1 - city2)
        return total_dist

    # 选择操作
    def _tournament_selection(self):
        """锦标赛选择"""
        tournament_size = 3
        selected = []
        for _ in range(self.pop_size - self.elite_size):
            contenders = random.sample(self.population, tournament_size)
            winner = min(contenders, key=self._calculate_distance)
            selected.append(winner[:])
        return selected

    def _roulette_selection(self):
        """轮盘赌选择"""
        distances = [self._calculate_distance(path) for path in self.population]
        fitness = [1 / d for d in distances]
        total_fitness = sum(fitness)
        probabilities = [f / total_fitness for f in fitness]

        selected = []
        for _ in range(self.pop_size - self.elite_size):
            pick = random.random()
            cumulative = 0
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if pick <= cumulative:
                    selected.append(self.population[i][:])
                    break
        return selected

    def _sus_selection(self):
        """随机均匀选择"""
        distances = [self._calculate_distance(path) for path in self.population]
        fitness = [1 / d for d in distances]
        total_fitness = sum(fitness)
        step = total_fitness / (self.pop_size - self.elite_size)

        selected = []
        r = random.uniform(0, step)
        cumulative = 0
        i = 0
        for _ in range(self.pop_size - self.elite_size):
            while cumulative < r and i < len(self.population):
                cumulative += fitness[i]
                i += 1
            selected.append(self.population[i - 1][:])
            r += step
        return selected

    # 交叉操作
    def _ox_crossover(self, parent1, parent2):
        """顺序交叉"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end] = parent1[start:end]

        remaining = [x for x in parent2 if x not in child[start:end]]
        j = 0
        for i in range(size):
            if child[i] is None:
                child[i] = remaining[j]
                j += 1
        return child

    def _pmx_crossover(self, parent1, parent2):
        """部分映射交叉"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size
        child[start:end] = parent1[start:end]

        mapping = {parent1[i]: parent2[i] for i in range(start, end)}
        for i in range(size):
            if child[i] is None:
                val = parent2[i]
                while val in child[start:end]:
                    val = mapping.get(val, val)
                child[i] = val
        return child

    def _cx_crossover(self, parent1, parent2):
        """循环交叉"""
        size = len(parent1)
        child = [None] * size
        visited = set()
        start = 0

        while None in child:
            cycle_val = parent1[start]
            child[start] = cycle_val
            visited.add(start)
            next_val = parent2[start]

            while next_val != cycle_val:
                idx = parent1.index(next_val)
                child[idx] = next_val
                visited.add(idx)
                next_val = parent2[idx]

            start = min([i for i in range(size) if i not in visited], default=0)
            if start == 0 and None in child:
                for i in range(size):
                    if child[i] is None:
                        child[i] = parent2[i]
        return child

    # 变异操作
    def _swap_mutation(self, path):
        """交换变异"""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(path)), 2)
            path[i], path[j] = path[j], path[i]
        return path

    def _insertion_mutation(self, path):
        """插入变异"""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(path)), 2)
            city = path.pop(i)
            path.insert(j, city)
        return path

    def _inversion_mutation(self, path):
        """逆序变异"""
        if random.random() < self.mutation_rate:
            i, j = sorted(random.sample(range(len(path)), 2))
            path[i : j + 1] = path[i : j + 1][::-1]
        return path

    def iterate(self, num_iterations):
        """
        迭代求解TSP，每次随机选择一种选择、交叉和变异操作
        :param num_iterations: 迭代次数
        :return: 当前较优解（城市编号列表）
        """
        for _ in range(num_iterations):
            print(f"doing {_} of {num_iterations}", end=" \t")
            # 计算当前种群中每个个体的适应度
            distances = [self._calculate_distance(path) for path in self.population]
            min_dist_idx = np.argmin(distances)
            if distances[min_dist_idx] < self.best_distance:
                self.best_distance = distances[min_dist_idx]
                self.best_solution = self.population[min_dist_idx][:]

            # 保留精英个体
            elite = sorted(range(len(distances)), key=lambda k: distances[k])[
                : self.elite_size
            ]
            new_population = [self.population[i][:] for i in elite]

            # 随机选择一种选择操作
            selection_method = random.choice(self.selection_methods)
            selected = selection_method()

            # 随机选择一种交叉操作
            crossover_method = random.choice(self.crossover_methods)
            while len(new_population) < self.pop_size:
                parent1, parent2 = random.sample(selected, 2)
                child = crossover_method(parent1, parent2)
                new_population.append(child)

            # 随机选择一种变异操作
            mutation_method = random.choice(self.mutation_methods)
            for i in range(self.elite_size, self.pop_size):
                new_population[i] = mutation_method(new_population[i])

            self.population = new_population
            print(f"Best distance: {self.best_distance}")

        return self.best_solution
