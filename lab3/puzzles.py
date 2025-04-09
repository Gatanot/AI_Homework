from heapq import heappush, heappop
from typing import List, Tuple, Set
import copy

# 计算曼哈顿距离的辅助函数
def manhattan_distance(puzzle: List[List[int]]) -> int:
    size = len(puzzle)
    distance = 0
    for i in range(size):
        for j in range(size):
            if puzzle[i][j] != 0:
                target_x = (puzzle[i][j] - 1) // size
                target_y = (puzzle[i][j] - 1) % size
                distance += abs(target_x - i) + abs(target_y - j)
    return distance

# 找到0的位置
def find_zero(puzzle: List[List[int]]) -> Tuple[int, int]:
    for i in range(len(puzzle)):
        for j in range(len(puzzle)):
            if puzzle[i][j] == 0:
                return i, j
    return -1, -1

# 检查是否是目标状态
def is_goal(puzzle: List[List[int]]) -> bool:
    size = len(puzzle)
    target = [[i * size + j + 1 for j in range(size)] for i in range(size)]
    target[size-1][size-1] = 0
    return puzzle == target

# 获取可能的移动方向
def get_moves(x: int, y: int, size: int) -> List[Tuple[int, int]]:
    moves = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上，下，左，右
    for dx, dy in directions:
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < size and 0 <= new_y < size:
            moves.append((new_x, new_y))
    return moves

# A*算法实现
def A_star(puzzle: List[List[int]]) -> List[int]:
    size = len(puzzle)
    start = copy.deepcopy(puzzle)
    open_list = []
    closed = set()
    
    # (f_score, g_score, puzzle, path)
    heappush(open_list, (manhattan_distance(start), 0, start, []))
    while open_list:
        f_score, g_score, current, path = heappop(open_list)
        
        if is_goal(current):
            return path
            
        state_tuple = tuple(map(tuple, current))
        if state_tuple in closed:
            continue
        closed.add(state_tuple)
        
        x, y = find_zero(current)
        for new_x, new_y in get_moves(x, y, size):
            next_puzzle = copy.deepcopy(current)
            # 移动方块
            next_puzzle[x][y], next_puzzle[new_x][new_y] = next_puzzle[new_x][new_y], next_puzzle[x][y]
            next_tuple = tuple(map(tuple, next_puzzle))
            
            if next_tuple in closed:
                continue
                
            new_g = g_score + 1
            new_h = manhattan_distance(next_puzzle)
            new_f = new_g + new_h
            new_path = path + [next_puzzle[x][y]]
            
            heappush(open_list, (new_f, new_g, next_puzzle, new_path))
    
    return []  # 无解

# IDA*算法实现
def IDA_star(puzzle: List[List[int]]) -> List[int]:
    size = len(puzzle)
    start = copy.deepcopy(puzzle)
    
    def search(puzzle: List[List[int]], g: int, threshold: int, path: List[int], visited: Set) -> Tuple[bool, int, List[int]]:
        f = g + manhattan_distance(puzzle)
        if f > threshold:
            return False, f, path
        if is_goal(puzzle):
            return True, f, path
            
        state_tuple = tuple(map(tuple, puzzle))
        if state_tuple in visited:
            return False, f, path
        visited.add(state_tuple)
        
        min_cost = float('inf')
        result_path = path
        x, y = find_zero(puzzle)
        
        for new_x, new_y in get_moves(x, y, size):
            next_puzzle = copy.deepcopy(puzzle)
            next_puzzle[x][y], next_puzzle[new_x][new_y] = next_puzzle[new_x][new_y], next_puzzle[x][y]
            
            found, cost, new_path = search(next_puzzle, g + 1, threshold, 
                                        path + [next_puzzle[x][y]], visited)
            if found:
                return True, cost, new_path
            min_cost = min(min_cost, cost)
        
        return False, min_cost, result_path
    
    threshold = manhattan_distance(start)
    while True:
        found, new_threshold, path = search(start, 0, threshold, [], set())
        if found:
            return path
        if new_threshold == float('inf'):
            return []  # 无解
        threshold = new_threshold

# 测试代码
if __name__ == "__main__":
    puzzle = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[0,13,14,15]]
    print("A* result:", A_star(puzzle))
    print("IDA* result:", IDA_star(puzzle))