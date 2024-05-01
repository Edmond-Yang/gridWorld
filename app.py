from flask import Flask, render_template, request, jsonify
import numpy as np
import random
from tqdm import tqdm

app = Flask(__name__)

N = 20

class GridWorld:
    def __init__(self, n):
        self.n = n  # grid size
        self.grid = np.zeros((n, n))
        self.start = None
        self.end = None
        self.obstacles = []
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.2  # Exploration rate

    def set_start(self, row, col):
        self.start = (row, col)

    def set_end(self, row, col):
        self.end = (row, col)

    def set_obstacle(self, row, col):
        self.obstacles.append((row, col))

    def is_reachable(self, state):
        row, col = state
        return 0 <= row < self.n and 0 <= col < self.n and state not in self.obstacles

    def initialize_q_table(self):
        """Initialize the Q-table for each state-action pair to zero."""
        for i in range(self.n):
            for j in range(self.n):
                if (i, j) not in self.obstacles and (i, j) != self.end:
                    self.q_table[(i, j)] = {'up': 0, 'down': 0, 'left': 0, 'right': 0}
                # Ensure the end state is also in the Q-table to avoid KeyError
                if (i, j) == self.end:
                    self.q_table[(i, j)] = {'up': 0, 'down': 0, 'left': 0, 'right': 0}

    def choose_action(self, state):
        """Choose an action using the epsilon-greedy policy."""
        if random.random() < self.epsilon:  # Explore with probability epsilon
            return random.choice(['up', 'down', 'left', 'right'])
        else:  # Exploit best action
            return max(self.q_table[state], key=self.q_table[state].get)

    def update_q_value(self, state, action, reward, next_state):
        """Update the Q-value for the given state and action."""
        if next_state in self.q_table:
            max_future_q = max(self.q_table[next_state].values())
            current_q = self.q_table[state][action]
            new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (
                        reward + self.discount_factor * max_future_q)
            self.q_table[state][action] = new_q
        else:
            # Handle the case where next_state is not in the q_table (like end state or non-reachable state)
            self.q_table[state][action] = (1 - self.learning_rate) * self.q_table[state][
                action] + self.learning_rate * reward

    def train(self, episodes=2000):
        """Train the agent using the Q-learning algorithm and record paths every 5 episodes."""
        self.initialize_q_table()
        path_log = []  # List to hold paths taken every 5 episodes

        for episode in range(episodes):
            state = self.start
            next_state = self.end

            current_path = []  # List to store the current path

            while state != self.end and state not in current_path and state != next_state:
                action = self.choose_action(state)
                next_state = self.get_next_state(state, action)
                reward = self.get_reward(next_state)
                self.update_q_value(state, action, reward, next_state)

                print(state)
                print(next_state)
                print('-'*30)

                if episode % N == N-1:  # Add state to the path if we're recording this episode
                    current_path.append(state)

                state = next_state

            if episode % N == N-1:  # Add the final state to the path and save it to the log
                if state == self.end:
                    current_path.append(self.end)
                path_log.append(current_path)

        return path_log  # Return the path log for analysis

    def get_next_state(self, state, action):
        """Get the next state from current state and action."""
        row, col = state
        if action == 'up':
            return (row - 1, col) if self.is_reachable((row - 1, col)) else state
        elif action == 'down':
            return (row + 1, col) if self.is_reachable((row + 1, col)) else state
        elif action == 'left':
            return (row, col - 1) if self.is_reachable((row, col - 1)) else state
        elif action == 'right':
            return (row, col + 1) if self.is_reachable((row, col + 1)) else state
        return state

    def get_reward(self, state):
        """Return the reward for reaching the given state."""
        if state == self.end:
            return 10  # reward for reaching the end
        if state in self.obstacles:
            return -1  # penalty for hitting an obstacle
        return -0.01  # small penalty for each move

    def get_optimal_path_from_q_table(self):
        path = []
        state = self.start
        while state != self.end:

            action = max(self.q_table[state], key=self.q_table[state].get)  # 選擇當前狀態下最大 Q 值的動作
            path.append(state)
            next_state = self.get_next_state(state, action)  # 獲取下一狀態
            if next_state in path:
                return []
            state = next_state

        path.append(self.end)
        return path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate_grid', methods=['POST'])
def generate_grid():
    n = int(request.form['n'])
    if n < 3 or n > 7:
        return "Error: n should be between 3 and 7"
    return render_template('index.html', n=n)


@app.route('/evaluate_policy', methods=['POST'])
def evaluate_policy():
    global N
    points = request.json['points']
    grid_size = int(request.json['n'])

    if grid_size in [3, 4]:
        N = 20
    else:
        N = 40

    grid_world = GridWorld(grid_size)

    for point in points:
        row, col, cell_type = point['row'], point['col'], point['type']
        if cell_type == 'start':
            grid_world.set_start(row, col)
        elif cell_type == 'end':
            grid_world.set_end(row, col)
        elif cell_type == 'obstacle':
            grid_world.set_obstacle(row, col)

    grid_world.initialize_q_table()
    action_log = grid_world.train(episodes=1000)  # 训练模型使用 Q-learning 算法

    optimal_path = grid_world.get_optimal_path_from_q_table()  # 从 Q 表获取最佳路径

    new_optimal_path = ''
    for point in optimal_path:
        new_optimal_path += '[' + str(point[0]) + ', ' + str(point[1]) + '] → '

    new_optimal_path = new_optimal_path[:-3]

    new_action_log = []

    if not new_optimal_path:
        new_optimal_path = 'Optimal Path Not Found'
    else:
        if action_log:
            for action_list in action_log:
                for actions in action_list:
                    new_action_log.append({'state': actions})

    print('ANS:')
    print(new_optimal_path)

    return jsonify({
        'message': new_optimal_path,
        'optimal_path': optimal_path,
        'action_log': new_action_log
    })


if __name__ == '__main__':
    app.run(port=9000, debug=True)