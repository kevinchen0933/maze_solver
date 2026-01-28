import heapq
import matplotlib.pyplot as plt
import numpy as np
import random

# 必須是奇數
SIZE = 31 
STAIR_COST = 5

def pos_to_index(x, y):
    # 確保座標與陣列索引完全對應：(x,y) -> [y,x]
    return int(y), int(x)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def generate_maze(w, h):
    maze = np.full((h, w), float('inf'))
    def walk(x, y):
        maze[y, x] = 1
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx*2, y + dy*2
            if 0 <= nx < w and 0 <= ny < h and maze[ny, nx] == float('inf'):
                maze[y+dy, x+dx] = 1
                walk(nx, ny)
    walk(1, 1)
    return maze

def create_world():
    world = np.ones((2, SIZE, SIZE))
    world[0] = generate_maze(SIZE, SIZE)
    world[1] = 4 # 地下層
    # 關鍵：強制鎖死起點、終點周邊，避免生成時被堵死
    for l, x, y in [(0,1,1), (0,29,29), (0,1,29), (1,15,15)]:
        world[l, y, x] = 1
    return world

def solve_a_star(world, start, goal, item, stairs):
    start_state = (start[0], start[1], start[2], 0)
    goal_xy = (goal[1], goal[2])
    g_score = {start_state: 0}
    f_score = {start_state: heuristic((start[1], start[2]), goal_xy)}
    pq = [(f_score[start_state], start_state)]
    came_from = {}
    
    while pq:
        _, (l, x, y, s) = heapq.heappop(pq)
        if (x, y) == goal_xy:
            return reconstruct(came_from, start_state, (l, x, y, s)), g_score[(l, x, y, s)]

        # --- 鄰居搜尋：絕對禁行牆壁 ---
        neighbors = []
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]: # 嚴格四方向
            nx, ny = x + dx, y + dy
            if 0 <= nx < SIZE and 0 <= ny < SIZE:
                # 直接讀取陣列，只要是 inf 就絕對不考慮
                if world[l, ny, nx] != float('inf'):
                    neighbors.append((l, nx, ny, world[l, ny, nx]))
        
        # 樓梯跨層
        if (x, y) in stairs:
            nl = 1 - l
            if world[nl, y, x] != float('inf'):
                neighbors.append((nl, x, y, STAIR_COST))

        for nl, nx, ny, base_cost in neighbors:
            ns = 1 if (nl, nx, ny) == item or s == 1 else 0
            # 裝備抵消：只在地面層生效
            cost = 1 if (s == 1 and nl == 0 and base_cost > 1 and base_cost != STAIR_COST) else base_cost
            
            tentative_g = g_score[(l, x, y, s)] + cost
            st = (nl, nx, ny, ns)
            
            if tentative_g < g_score.get(st, float('inf')):
                came_from[st] = (l, x, y, s)
                g_score[st] = tentative_g
                f_score[st] = tentative_g + heuristic((nx, ny), goal_xy)
                heapq.heappush(pq, (f_score[st], st))
    return [], 0

def reconstruct(cf, s, e):
    path = [e]
    while path[-1] != s: path.append(cf[path[-1]])
    return path[::-1]

# --- 繪圖區塊 ---
world_data = create_world()
stair_list = [(15, 15), (5, 5), (25, 25)]
path_res, _ = solve_a_star(world_data, (0, 1, 1), (0, 29, 29), (0, 1, 29), stair_list)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 7), facecolor='#121212')

for ax, l, title, cm in zip([ax0, ax1], [0, 1], ["Surface Maze", "Underground Cavern"], [plt.cm.YlGn, plt.cm.bone]):
    ax.axis('off')
    ax.set_title(title, color='white', fontsize=16)
    d_p = np.copy(world_data[l])
    d_p[np.isinf(d_p)] = 150
    # 關鍵設定：extent 定義了邊界，讓整數座標剛好落在方格中心
    ax.imshow(d_p, cmap=cm, origin='lower', extent=(-0.5, SIZE-0.5, -0.5, SIZE-0.5))
    for sx, sy in stair_list:
        ax.scatter(sx, sy, c='#3498db', marker='D', s=100, edgecolors='white', zorder=10)

if path_res:
    for i in range(len(path_res)-1):
        p1, p2 = path_res[i], path_res[i+1]
        if p1[0] == p2[0]:
            ax = ax0 if p1[0] == 0 else ax1
            # 這裡不需要加 0.5，因為 extent 已經幫你校正了中心點
            ax.plot([p1[1], p2[1]], [p1[2], p2[2]], 
                    color='#2ecc71' if p1[3]==1 else '#e74c3c', linewidth=4, zorder=20)

ax0.scatter(1, 1, c='lime', s=200, zorder=30)
ax0.scatter(29, 29, c='red', s=200, zorder=30)
ax0.scatter(1, 29, c='gold', marker='*', s=400, edgecolors='black', zorder=30)
plt.show()