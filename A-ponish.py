import heapq
import matplotlib.pyplot as plt
import numpy as np
import random

# 必須是奇數，確保迷宮有牆有路
SIZE = 31 

def heuristic(a, b):
    # 曼哈頓距離，提供 A* 演算法的方向感
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def generate_maze(w, h):
    # 建立一個全黑的矩陣（全是牆壁，代價為 inf）
    maze = np.full((h, w), float('inf'))
    
    def walk(x, y):
        maze[y, x] = 1 # 設為可通行的道路
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx * 2, y + dy * 2
            if 0 <= nx < w and 0 <= ny < h and maze[ny, nx] == float('inf'):
                maze[y + dy, x + dx] = 1 # 打通中間的牆
                walk(nx, ny)
                
    walk(1, 1) # 從 (1,1) 開始挖迷宮
    return maze

def solve_a_star(grid, start, goal):
    # 優先隊列儲存: (f_score, x, y)
    pq = [(heuristic(start, goal), start[0], start[1])]
    # 儲存從起點到該點的實際成本
    g_score = {start: 0}
    came_from = {}
    
    while pq:
        f, x, y = heapq.heappop(pq)
        curr = (x, y)
        
        if curr == goal:
            # 重建路徑
            path = []
            while curr in came_from:
                path.append(curr)
                curr = came_from[curr]
            path.append(start)
            return path[::-1], g_score[goal]
        
        # 嚴格四方向移動（禁絕斜向，避免穿牆）
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < SIZE and 0 <= ny < SIZE:
                weight = grid[ny, nx] # 取得地圖權重
                
                # 鋼鐵碰撞判斷：只要是牆壁 (inf)，絕對不考慮
                if weight == float('inf'):
                    continue
                
                tentative_g = g_score[(x, y)] + weight
                next_node = (nx, ny)
                
                if tentative_g < g_score.get(next_node, float('inf')):
                    came_from[next_node] = (x, y)
                    g_score[next_node] = tentative_g
                    f_score = tentative_g + heuristic(next_node, goal)
                    heapq.heappush(pq, (f_score, nx, ny))
                    
    return [], 0

# --- 1. 初始化地圖 ---
maze_grid = generate_maze(SIZE, SIZE)

# 隨機加入岩漿區域（增加路徑複雜度）
for _ in range(40):
    rx, ry = random.randint(1, SIZE-2), random.randint(1, SIZE-2)
    if maze_grid[ry, rx] == 1:
        maze_grid[ry, rx] = 20 # 設為高成本區域

# 確保起終點通暢
start_pos = (1, 1)
goal_pos = (29, 29)
maze_grid[start_pos[1], start_pos[0]] = 1
maze_grid[goal_pos[1], goal_pos[0]] = 1

# --- 2. 執行尋路 ---
path, total_cost = solve_a_star(maze_grid, start_pos, goal_pos)

# --- 3. 視覺化呈現 ---
fig, ax = plt.subplots(figsize=(10, 10), facecolor='#121212')

# 讓顯示顏色更好看
d_plot = np.copy(maze_grid)
d_plot[np.isinf(d_plot)] = 50 # 讓牆壁顯示為深綠色

# 核心修正：extent 讓座標整數點剛好落在格子的中心
ax.imshow(d_plot, cmap='YlGn', origin='lower', extent=(-0.5, SIZE-0.5, -0.5, SIZE-0.5))

if path:
    # 提取路徑座標
    px, py = zip(*path)
    # 線條會完美對齊格子中心
    ax.plot(px, py, color='#e74c3c', linewidth=4, zorder=5, solid_capstyle='round')

# 標記起點與終點
ax.scatter(start_pos[0], start_pos[1], c='lime', s=200, edgecolors='white', zorder=10, label='Start')
ax.scatter(goal_pos[0], goal_pos[1], c='red', s=200, edgecolors='white', zorder=10, label='Goal')

ax.set_title(f"A* Pathfinding (Cost: {total_cost})", color='white', fontsize=16)
ax.axis('off')
plt.show()