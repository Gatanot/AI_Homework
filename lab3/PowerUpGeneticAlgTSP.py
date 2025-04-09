import numpy as np
import random
from multiprocessing import Pool
from functools import partial
import time


class GeneticAlgTSP:
    def __init__(self, filename):
        """构造函数：读取TSP数据集并初始化种群"""
        self.cities = self._load_tsp_data(filename)
        self.num_cities = len(self.cities)

        # 预计算距离矩阵
        self.dist_matrix = self._compute_distance_matrix()

        # 初始化参数
        self.pop_size = 100
        self.mutation_rate = 0.5
        self.elite_size = int(0.1 * self.pop_size)

        # 初始化种群
        self.population = [self._create_random_path() for _ in range(self.pop_size)]
        self.best_solution = None
        self.best_distance = float("inf")

        # 操作函数列表
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

    def _compute_distance_matrix(self):
        """预计算城市间距离矩阵"""
        n = self.num_cities
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist[i, j] = dist[j, i] = np.linalg.norm(
                    self.cities[i] - self.cities[j]
                )
        return dist

    def _create_random_path(self):
        """生成随机路径"""
        path = np.arange(1, self.num_cities + 1)
        np.random.shuffle(path)
        return path.tolist()

    def _calculate_distance(self, path):
        """使用距离矩阵计算路径总距离"""
        total_dist = 0
        for i in range(len(path)):
            city1 = path[i] - 1  # 转换为0-based索引
            city2 = path[(i + 1) % self.num_cities] - 1
            total_dist += self.dist_matrix[city1, city2]
        return total_dist

    # 选择操作
    def _tournament_selection(self):
        tournament_size = 3
        selected = []
        for _ in range(self.pop_size - self.elite_size):
            contenders = random.sample(self.population, tournament_size)
            winner = min(contenders, key=self._calculate_distance)
            selected.append(winner)
        return selected

    def _roulette_selection(self):
        distances = [self._calculate_distance(path) for path in self.population]
        fitness = [1 / d for d in distances]
        total_fitness = sum(fitness)
        probabilities = [f / total_fitness for f in fitness]
        selected = random.choices(
            self.population, weights=probabilities, k=self.pop_size - self.elite_size
        )
        return [p[:] for p in selected]

    def _sus_selection(self):
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
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(path)), 2)
            path[i], path[j] = path[j], path[i]
        return path

    def _insertion_mutation(self, path):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(path)), 2)
            city = path.pop(i)
            path.insert(j, city)
        return path

    def _inversion_mutation(self, path):
        if random.random() < self.mutation_rate:
            i, j = sorted(random.sample(range(len(path)), 2))
            path[i : j + 1] = path[i : j + 1][::-1]
        return path

    def _process_individual(self, individual_data):
        """并行处理个体：交叉和变异"""
        parent1, parent2, crossover_method, mutation_method = individual_data
        child = crossover_method(parent1, parent2)
        return mutation_method(child)

    def iterate(self, num_iterations):
        """迭代求解TSP，使用并行优化"""
        with Pool() as pool:
            for _ in range(num_iterations):
                print(f"doing {_} of {num_iterations}", end=" ")
                # 并行计算距离
                distances = pool.map(self._calculate_distance, self.population)
                min_dist_idx = np.argmin(distances)
                if distances[min_dist_idx] < self.best_distance:
                    self.best_distance = distances[min_dist_idx]
                    self.best_solution = self.population[min_dist_idx][:]

                # 保留精英
                elite = sorted(range(len(distances)), key=lambda k: distances[k])[
                    : self.elite_size
                ]
                new_population = [self.population[i][:] for i in elite]

                # 随机选择操作
                selection_method = random.choice(self.selection_methods)
                crossover_method = random.choice(self.crossover_methods)
                mutation_method = random.choice(self.mutation_methods)

                # 选择
                selected = selection_method()

                # 并行交叉和变异
                num_to_generate = self.pop_size - self.elite_size
                parents = [random.sample(selected, 2) for _ in range(num_to_generate)]
                individual_data = [
                    (p[0], p[1], crossover_method, mutation_method) for p in parents
                ]
                new_individuals = pool.map(self._process_individual, individual_data)
                new_population.extend(new_individuals)

                self.population = new_population
                print(f"current best distance: {self.best_distance}")

        return self.best_solution


# 示例使用
if __name__ == "__main__":
    start_time = time.time()
    ga = GeneticAlgTSP("uy734.tsp")
    best_path = ga.iterate(1000)  # 迭代10000次
    print(f"Best path: {best_path}")
    print(f"Best distance: {ga.best_distance}")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"程序执行时间: {execution_time:.4f} 秒")
