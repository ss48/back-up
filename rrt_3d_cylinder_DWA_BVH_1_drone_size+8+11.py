import numpy as np
import time
from rtree import index
import sys
import uuid
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import random
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.append('/home/dell/rrt-algorithms')

from rrt_algorithms.rrt.rrt2 import RRT
from rrt_algorithms.search_space.search_space3 import SearchSpace
from rrt_algorithms.utilities.plotting2 import Plot
from rrt_algorithms.dwa_algorithm.DWA4 import DWA, FuzzyController
from rrt_algorithms.dqn_algorithm.DQN0 import DQNAgent
from rrt_algorithms.dqn_algorithm.DQN0 import DroneEnv
from rrt_algorithms.utilities.obstacle_generation3 import generate_random_cylindrical_obstacles 
import time  # Import for time tracking

# Initialize cube times and add placeholders for text annotations
cube_times = {
    'cube1': {'start': time.time(), 'end': None},
    'cube2': {'start': time.time(), 'end': None},
    'cube3': {'start': time.time(), 'end': None}
}


# Define the search space dimensions
X_dimensions = np.array([(-200, 200), (-200, 200), (-150, 150)])
path = None

# Initial and goal positions
x_init = (150, 50, -82)
x_intermediate = (-180, -150, -150)
x_second_goal = (200, 160, -130)
x_third_goal = (-120, 160, -10)
x_final_goal = (150,-150, -100)

# Generate random obstacles
n = 4 # Number of random obstacles


# Define static obstacles
obstacles = [
    (100, 0, -150, 85, 25),  #(center coordinate, height,diameter)
    (-100, -100, -150, 60, 19),   
    (100, 150, 0, 190, 15),   
    (-80, -180, 50, 70, 22), 
    (0, 50, -130, 80, 12),
]

cube_size = 5
#cube_position = np.array(x_init)

# Set the initial positions for each cube (use goal positions or starting positions as required)
cube1_position = np.array(x_init)           # Initial position for Cube 1
cube2_position = np.array(x_intermediate)    # Initial position for Cube 2
cube3_position = np.array(x_second_goal)     # Initial position for Cube 3
final_goal = np.array(x_final_goal)          # Shared final goal for all cubes


# Initialize battery levels for each cube
battery_levels = [100.0, 100.0, 100.0]  # Starting with 100% battery for each cube
battery_depletion_rate = 0.05  # Define battery depletion rate per distance unit


# Store all initial positions and assign each one to a cube's start position

# Define paths for each cube from their start position to the final goal
 
charging_station = np.array([0, -150, 0])   
# Function to create vertices for the cube
def create_cube_vertices(center, size):
    d = size / 2.0
    return np.array([
        center + np.array([-d, -d, -d]),
        center + np.array([-d, -d,  d]),
        center + np.array([-d,  d, -d]),
        center + np.array([-d,  d,  d]),
        center + np.array([ d, -d, -d]),
        center + np.array([ d, -d,  d]),
        center + np.array([ d,  d, -d]),
        center + np.array([ d,  d,  d]),
    ])

# Plot each cube with the initial vertices
def plot_cube(ax, vertices, color='orange'):
    pairs = [
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)
    ]
    lines = []
    for p1, p2 in pairs:
        line, = ax.plot([vertices[p1][0], vertices[p2][0]],
                        [vertices[p1][1], vertices[p2][1]],
                        [vertices[p1][2], vertices[p2][2]],
                        color=color)
        lines.append(line)
    return lines
    

# Combine static and randomly generated obstacles into one list
X = SearchSpace(X_dimensions)
# Initialize the SearchSpace with all obstacles
Obstacles = generate_random_cylindrical_obstacles(X, (-150,-150,-150), (150,150,0), n)

all_obstacles = obstacles + Obstacles
X = SearchSpace(X_dimensions, all_obstacles)
# RRT parameters
q = 300
r = 1
max_samples = 5024
prc = 0.1
def add_vertex(self, tree, v):
    if isinstance(v, tuple) and len(v) == 3:
        v_composite = (v[0], v[1], v[2], v[2])  # Adding z as a 2D bounding pair
    else:
        v_composite = v
    self.trees[tree].V.insert(0, v_composite, v)
# DWA parameters
dwa_params = {
    'max_speed': 3.0,
    'min_speed': 0.0,
    'max_yaw_rate': 40.0 * np.pi / 180.0,
    'max_accel': 0.3,  
    'v_reso': 0.05,  
    'yaw_rate_reso': 0.2 * np.pi / 180.0,
    'dt': 0.1,
    'predict_time': 2.0,
    'to_goal_cost_gain': 1.5,
    'speed_cost_gain': 0.5,
    'obstacle_cost_gain': 1.0,
}
def generate_collision_free_rrt(start, goal, search_space, obstacles, dynamic_paths=None, max_attempts=20):
    dynamic_paths = dynamic_paths or []  # Default to an empty list if None is provided
    for _ in range(max_attempts):
        rrt = RRT(search_space, q=30, x_init=start, x_goal=goal, max_samples=2024, r=1, prc=0.1)
        path = rrt.rrt_search()

        # Check for collisions
        if path and is_collision_free_path(path, obstacles, dynamic_paths):
            return path  # Return only if entire path is obstacle-free

    return None



def generate_full_rrt_path(start, goals, search_space, obstacles, dynamic_paths=None, max_attempts=10):
    dynamic_paths = dynamic_paths or []  # Ensure it defaults to an empty list if None
    complete_path = []
    current_start = start

    for goal in goals:
        path_segment = generate_collision_free_rrt(current_start, goal, search_space, obstacles, dynamic_paths, max_attempts)
        if path_segment is None:
            print(f"Failed to find path to goal: {goal}")
            return None
        complete_path.extend(path_segment if not complete_path else path_segment[1:])
        current_start = goal

    return complete_path


def compute_energy_usage(path, velocity):
    energy = 0.0
    for i in range(1, len(path)):
        distance = np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))
        energy += distance * velocity
    return energy
    
def min_obstacle_clearance(path, obstacles):
    min_clearance = float('inf')
    for point in path:
        for obs in obstacles:
            clearance = np.linalg.norm(np.array(point[:2]) - np.array(obs[:2])) - obs[4]
            if clearance < min_clearance:
                min_clearance = clearance
    return min_clearance   
         
def path_length(path):
    return sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) for i in range(len(path) - 1))

def path_smoothness(path):
    total_curvature = 0.0
    for i in range(1, len(path) - 1):
        vec1 = np.array(path[i]) - np.array(path[i-1])
        vec2 = np.array(path[i+1]) - np.array(path[i])
        angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        total_curvature += angle
    return total_curvature
    

# Adaptive Large Neighborhood Search (ALNS) Functions
def alns_optimize_path(path, X, all_obstacles, dynamic_paths=None, max_iterations=100):
    neighborhoods = [segment_change, detour_addition, direct_connection]
    neighborhood_scores = {segment_change: 1.0, detour_addition: 1.0, direct_connection: 1.0}
    current_path = path.copy()

    for _ in range(max_iterations):
        neighborhood = random.choices(neighborhoods, weights=neighborhood_scores.values())[0]
        new_path = neighborhood(current_path, X, all_obstacles)

        if is_collision_free_path(new_path, all_obstacles, dynamic_paths):
            if path_length(new_path) < path_length(current_path):
                current_path = new_path
                neighborhood_scores[neighborhood] *= 1.1  # Reward successful neighborhood
            else:
                neighborhood_scores[neighborhood] *= 0.9  # Penalize unsuccessful neighborhood

    return current_path


def segment_change(path, X, all_obstacles, r=1):
    if len(path) < 3:
        return path

    i = random.randint(0, len(path) - 3)
    j = random.randint(i + 2, len(path) - 1)

    x_a = path[i]
    x_b = path[j]

    spatial_x_a = np.array(x_a[:3])
    spatial_x_b = np.array(x_b[:3])

    new_point_spatial = spatial_x_a + (spatial_x_b - spatial_x_a) / 2
    new_point = list(new_point_spatial) + list(x_a[3:])

    if X.collision_free(spatial_x_a, new_point_spatial, r) and X.collision_free(new_point_spatial, spatial_x_b, r):
        new_path = path[:i + 1] + [new_point] + path[j:]
        return new_path

    return path

def detour_addition(path, X, all_obstacles, r=1):
    if len(path) < 2:
        return path

    i = random.randint(0, len(path) - 2)
    x_a = path[i]
    x_b = path[i + 1]

    spatial_x_a = np.array(x_a[:3])
    spatial_x_b = np.array(x_b[:3])

    detour_point_3d = spatial_x_a + (np.random.rand(3) - 0.5) * 2 * r
    detour_point = list(detour_point_3d) + list(x_a[3:])

    if X.collision_free(spatial_x_a, detour_point_3d, r) and X.collision_free(detour_point_3d, spatial_x_b, r):
        new_path = path[:i + 1] + [detour_point] + path[i + 1:]
        return new_path

    return path
def is_collision_free_path(path, obstacles, dynamic_paths=None, safe_distance=10.0):
    # Ensure dynamic_paths is a list if None is provided
    dynamic_paths = dynamic_paths or []

    for i in range(len(path) - 1):
        # Ensure segment_start and segment_end have only (x, y, z) coordinates
        segment_start = path[i][:3]
        segment_end = path[i + 1][:3]

        # Check collision with static obstacles
        if not X.collision_free(segment_start, segment_end, safe_distance):
            return False

        # Check dynamic paths if provided
        for other_path in dynamic_paths:
            if other_path == path:
                continue
            for j in range(len(other_path) - 1):
                if segments_intersect(segment_start, segment_end, other_path[j][:3], other_path[j + 1][:3]):
                    return False

    return True




def segments_intersect(p1, p2, q1, q2):
    """Checks if two line segments (p1, p2) and (q1, q2) intersect."""
    return np.cross(q2 - q1, p1 - q1) * np.cross(q2 - q1, p2 - q1) < 0 and \
           np.cross(p2 - p1, q1 - p1) * np.cross(p2 - p1, q2 - p1) < 0


def direct_connection(path, X, all_obstacles, r=1):
    if len(path) < 3:
        return path  # Can't directly connect if there aren't enough points

    new_path = path.copy()
    i = random.randint(0, len(path) - 3)  # Select a random starting point
    j = random.randint(i + 2, len(path) - 1)  # Select a random ending point

    x_a = path[i][:3]  # Extract only the spatial coordinates (x, y, z)
    x_b = path[j][:3]  # Extract only the spatial coordinates (x, y, z)

    if X.collision_free(x_a, x_b, r):
        new_path = new_path[:i + 1] + new_path[j:]  # Remove the points between i and j

    return new_path

def find_closest_point(path, goal):
    distances = [np.linalg.norm(np.array(p[:3]) - np.array(goal)) for p in path]
    return np.argmin(distances)

# RRT pathfinding to the first intermediate goal
print("RRT to first intermediate goal...")
start_time = time.time()
rrt = RRT(X, q, x_init, x_intermediate, max_samples, r, prc)
path1 = rrt.rrt_search()
rrt_time = time.time() - start_time

if path1 is not None:
    print("RRT to 1st to 2nd goal...")
    rrt = RRT(X, q, x_intermediate, x_second_goal, max_samples, r, prc)
    path2 = rrt.rrt_search()

    if path2 is not None:
        print("RRT to 2nd to 3rd goal...")
        rrt = RRT(X, q, x_second_goal, x_third_goal, max_samples, r, prc)
        path3 = rrt.rrt_search()
            
        if path3 is not None:
            print("RRT to final goal...")
            rrt = RRT(X, q, x_third_goal, x_final_goal, max_samples, r, prc)
            path_final = rrt.rrt_search()

        # Combine all paths
            if path_final is not None:
                path = path1 + path2[1:] + path3[1:] + path_final[1:] 
            else:
                path = path1 + path2[1:] + path3[1:] 
        else:
            path = path1
    else:
        path = None




# Apply DWA for local optimization along the RRT path
if path is not None:
    dwa = DWA(dwa_params)
    optimized_path = []

    start_time = time.time()
    for i in range(len(path) - 1):
        start_point = path[i] + (0.0, 0.0)  # Initialize v and w to 0, using tuple
        end_point = path[i + 1]

        # Enhanced DWA path planning with high-resolution collision checking
        local_path = dwa.plan(start_point, end_point, X, all_obstacles)  # Increased steps for finer checks
        optimized_path.extend(local_path)
    dwa_time = time.time() - start_time

    # Apply ALNS optimization to each segment of the path
    print("Applying ALNS optimization...")
    alns_optimized_path = []
    initial_positions = [cube1_position, cube2_position, cube3_position]



    # Find the closest points to the goals
    index_intermediate = find_closest_point(optimized_path, x_intermediate)
    index_second_goal = find_closest_point(optimized_path, x_second_goal)
    index_third_goal = find_closest_point(optimized_path, x_third_goal)
    # Segment 1: Start to Intermediate
    segment1 = optimized_path[:index_intermediate + 1]
    alns_optimized_segment1 = alns_optimize_path(segment1, X, all_obstacles, max_iterations=100)
    
    # Segment 2: Intermediate to Second Goal
    segment2 = optimized_path[index_intermediate:index_second_goal + 1]
    alns_optimized_segment2 = alns_optimize_path(segment2, X, all_obstacles, max_iterations=100)
    
    # Segment 3: Second Goal to Final Goal
    segment3 = optimized_path[index_second_goal:]
    alns_optimized_segment3 = alns_optimize_path(segment3, X, all_obstacles, max_iterations=100)
    
        # Segment 3: Second Goal to Final Goal
    segment4 = optimized_path[index_third_goal:]
    alns_optimized_segment4 = alns_optimize_path(segment4, X, all_obstacles, max_iterations=100)
   
    
    # Combine the ALNS-optimized segments
    alns_optimized_path = alns_optimized_segment1 + alns_optimized_segment2[1:] + alns_optimized_segment3[1:] + alns_optimized_segment4 + alns_optimized_segment1

    # Flatten nested structures if necessary
    alns_optimized_path = [point[:3] for point in alns_optimized_path]  # Keep only (x, y, z) coordinates

    goals = [x_intermediate, x_second_goal, x_third_goal, x_final_goal]  # Adjust based on desired goals
    
    paths = []
    
    initial_positions = [cube1_position, cube2_position, cube3_position]
    goals = [x_intermediate, x_second_goal, x_third_goal, x_final_goal]

    for start_pos in initial_positions:
        # Generate the full path for each cube with ALNS applied
        path_segment = generate_full_rrt_path(start_pos, goals, X, all_obstacles, max_attempts=20)
    
        if path_segment:
            # Optimize the path using ALNS
            optimized_path_segment = alns_optimize_path(path_segment, X, all_obstacles, max_iterations=100)
            paths.append(optimized_path_segment)
        else:
            print(f"Warning: Failed to find a path from {start_pos} to final goal.")




    # Ensure all paths are correctly formatted and flatten nested structures

def validate_and_correct_path(path, label="Path"):
    """
    Ensures that the path only contains (x, y, z) points and handles nested structures.
    Returns a cleaned path if valid; raises a warning and skips if not.
    """
    corrected_path = []
    for idx, point in enumerate(path):
        if isinstance(point, np.ndarray):
            flat_point = point.flatten().tolist()
        elif isinstance(point, (list, tuple)):
            flat_point = point
        else:
            print(f"Warning: {label} at index {idx} is not an array or tuple. Skipping.")
            continue
        
        # Ensure (x, y, z) format and check for NaN values
        if len(flat_point) >= 3 and not any(np.isnan(flat_point[:3])):
            corrected_path.append(flat_point[:3])
        else:
            print(f"Warning: {label} at index {idx} is incorrectly formatted or contains NaNs. Skipping.")
    return np.array(corrected_path) if corrected_path else None
env = DroneEnv(X, x_init, x_final_goal, all_obstacles, dwa_params)
# --- Generate ML Path with Validation ---
# Assuming the environment `env` and the DQN agent are defined elsewhere in the code
ml_path = []
state = env.reset()
done = False

while not done:
    action = env.agent.act(state)
    next_state, _, done = env.step(action)
    ml_path.append(next_state[0][:3])  # Only keep (x, y, z) coordinates
    state = next_state

# Validate the ML path after generation
ml_path = validate_and_correct_path(ml_path, label="ML Path")

if ml_path is not None:
    print("ML path successfully generated and validated.")
else:
    print("Warning: ML path could not be validated and will not be plotted.")



    # Validate and correct paths
optimized_path = validate_and_correct_path(optimized_path)
alns_optimized_path = validate_and_correct_path(alns_optimized_path)

# Metrics calculation
# Initialize default metrics in case paths are not generated
rrt_path_length = dwa_path_length = alns_path_length = 0
rrt_smoothness = dwa_smoothness = alns_smoothness = 0
rrt_clearance = dwa_clearance = alns_clearance = float('inf')
rrt_energy = dwa_energy = alns_energy = 0

# Check if paths are generated before calculating metrics
# Check if paths are generated before calculating metrics
if path is not None:
    rrt_path_length = path_length(path)
    rrt_smoothness = path_smoothness(path)
    rrt_clearance = min_obstacle_clearance(path, all_obstacles)
    rrt_energy = compute_energy_usage(path, dwa_params['max_speed'])

if optimized_path is not None and optimized_path.size > 0:
    dwa_path_length = path_length(optimized_path)
    dwa_smoothness = path_smoothness(optimized_path)
    dwa_clearance = min_obstacle_clearance(optimized_path, all_obstacles)
    dwa_energy = compute_energy_usage(optimized_path, dwa_params['max_speed'])

if alns_optimized_path is not None and alns_optimized_path.size > 0:
    alns_path_length = path_length(alns_optimized_path)
    alns_smoothness = path_smoothness(alns_optimized_path)
    alns_clearance = min_obstacle_clearance(alns_optimized_path, all_obstacles)
    alns_energy = compute_energy_usage(alns_optimized_path, dwa_params['max_speed'])

# Print the metrics
print(f"RRT Path Length: {rrt_path_length}")
print(f"DWA Optimized Path Length: {dwa_path_length}")
print(f"ALNS Optimized Path Length: {alns_path_length}")
print(f"RRT Path Smoothness: {rrt_smoothness}")
print(f"DWA Optimized Path Smoothness: {dwa_smoothness}")
print(f"ALNS Optimized Path Smoothness: {alns_smoothness}")
print(f"RRT Path Minimum Clearance: {rrt_clearance}")
print(f"DWA Optimized Path Minimum Clearance: {dwa_clearance}")
print(f"ALNS Optimized Path Minimum Clearance: {alns_clearance}")
print(f"RRT Energy Usage: {rrt_energy}")
print(f"DWA Energy Usage: {dwa_energy}")
print(f"ALNS Energy Usage: {alns_energy}")



def intersects_obstacle(self, point, obstacle):
    # Ensure point has only (x, y, z) coordinates
    point = point[:3]

    try:
        x, y, z = point
    except ValueError:
        raise ValueError(f"Expected point to have three values for (x, y, z), got: {point}")

    # Debug log for validation
    print(f"Intersects check for point {point} with obstacle {obstacle}")

    # Assuming obstacle is (center_x, center_y, center_z, height, radius)
    center_x, center_y, center_z, height, radius = obstacle
    distance_xy = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    within_radius = distance_xy <= radius
    within_height = center_z <= z <= (center_z + height)

    return within_radius and within_height



average_velocity = dwa_params['max_speed']
rrt_energy = compute_energy_usage(path, average_velocity)
dwa_energy = compute_energy_usage(optimized_path, average_velocity)
alns_energy = compute_energy_usage(alns_optimized_path, average_velocity)

# Assuming each cube starts moving at different points in the script
if cube_times['cube1']['start'] is None:
    cube_times['cube1']['start'] = time.time()  # Record start time for Cube 1

if cube_times['cube2']['start'] is None:
    cube_times['cube2']['start'] = time.time()  # Record start time for Cube 2

if cube_times['cube3']['start'] is None:
    cube_times['cube3']['start'] = time.time()  # Record start time for Cube 3

# Training the DQN

# Training parameters
num_episodes = 5
batch_size = 24

# Train the DQN model
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.agent.act(state)
        next_state, reward, done = env.step(action)
        env.agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(env.agent.memory) > batch_size:
            env.agent.replay(batch_size)

    print(f"Episode: {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {env.agent.epsilon:.4f}")

# Save the trained model
env.agent.model.save("dqn_model_optimized.keras")
print("Optimized DQN model saved as dqn_model_optimized.keras")



while not done:
    action = env.agent.act(state)  # Use the trained policy to act
    next_state, _, done = env.step(action)
    ml_path.append(next_state)
    state = next_state

# Convert ml_path to a numpy array for easy handling
ml_path = np.array(ml_path)

# Check if ml_path has valid data
if ml_path.size == 0 or len(ml_path) < 3:
    print("Warning: ML path is not available or has insufficient points.")
else:
    print("ML path successfully generated.")


# Ensure the final paths are properly flattened and structured
def flatten_path_points(path):
    cleaned_path = []
    for idx, point in enumerate(path):
        if isinstance(point, np.ndarray):
            flat_point = point.flatten().tolist()
            if len(flat_point) >= 3:
                cleaned_path.append(flat_point[:3])  # Take only the first three elements (x, y, z)
            else:
                raise ValueError(f"Point {idx} in optimized_path is incorrectly formatted: {flat_point}")
        elif isinstance(point, (list, tuple)) and len(point) >= 3:
            cleaned_path.append(point[:3])  # Handle lists or tuples directly
        else:
            raise ValueError(f"Point {idx} in optimized_path is not correctly formatted: {point}")
    return np.array(cleaned_path)

# Apply the flattening function to optimized_path before plotting
try:
    optimized_path = flatten_path_points(optimized_path)
    ml_path = flatten_path_points(ml_path)
except ValueError as e:
    print(f"Error encountered during flattening: {e}")



# First figure: RRT, DWA, ALNS paths with ML path
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
print(type(ax))  
ax.set_ylim(X_dimensions[0])
ax.set_xlim(X_dimensions[1])
ax.set_zlim(X_dimensions[2])

# Initialize time display annotations
time_texts = [
    ax.text2D(0.0, 0.98 - i * 0.05, "", transform=ax.transAxes)
    for i in range(3)
]

# Speed annotations for each cube
speed_texts = [
    ax.text2D(0.09, 0.98 - i * 0.05, "", transform=ax.transAxes)
    for i in range(3)
]

# Distance annotations for each cube
distance_texts = [
    ax.text2D(0.2, 0.98 - i * 0.05, "", transform=ax.transAxes)
    for i in range(3)
]

# Battery level annotations for each cube
battery_texts = [
    ax.text2D(0.32, 0.98 - i * 0.05, "", transform=ax.transAxes)
    for i in range(3)
]


goals = {
    "Start": x_init,
    "2nd Goal": x_intermediate,
    "3rd Goal": x_second_goal,
    "4th Goal": x_third_goal,
    "Final Goal": x_final_goal
}

for label, coord in goals.items():
    ax.scatter(*coord, label=label, s=50)
    ax.text(coord[0], coord[1], coord[2], label, fontsize=9, ha='right')

# Generate optimized path (Assuming `alns_optimized_path` exists and is a list of points)
# This should be calculated before this section
alns_optimized_path = [x_init, x_intermediate, x_second_goal, x_third_goal, x_final_goal]  # Placeholder for testing



# Initial position for the moving cube
cube_vertices = create_cube_vertices(cube1_position, cube_size)
#cube = plot_cube(ax, cube_vertices)

# Set up the speed of the cube
speed = 0.02 # distance per frame


ax.set_xlabel('X Axis (m)')
ax.set_ylabel('Y Axis (m)')
ax.set_zlabel('Z Axis (m)')




# Ensure the ALNS, RRT, and DWA paths have valid coordinates
if path is not None and len(path) > 1:
    ax.plot(*zip(*path), linestyle='--', color='red', label="RRT Path")
else:
    print("Warning: RRT path is not available or has insufficient points.")

if optimized_path is not None and len(optimized_path) > 1:
    ax.plot(*zip(*optimized_path), linestyle='--', color='blue', label="DWA Optimized Path")
else:
    print("Warning: DWA optimized path is not available or has insufficient points.")

if alns_optimized_path is not None and len(alns_optimized_path) > 1:
    ax.plot(*zip(*alns_optimized_path), linestyle='--', color='green', label="ALNS Optimized Path")
else:
    print("Warning: ALNS optimized path is not available or has insufficient points.")

# Ensure mlpath has valid coordinates if it is plotted
#if ml_path.size > 0 and len(ml_path) >= 2:
#    ax.plot(ml_path[:, 0], ml_path[:, 1], ml_path[:, 2], color='black', label="ML Path")
#    print(f"ml_path length: {len(ml_path)}")
#    print(f"ml_path data:\n{ml_path}")
#else:
#    print("Warning: ML path is not available or has insufficient points.")



# Plot paths if they are validated
if ml_path is not None:
    ax.plot(ml_path[:, 0], ml_path[:, 1], ml_path[:, 2], color='black', label="ML Path")



# Plot static obstacles
for obs in all_obstacles:
    x_center, y_center, z_min, z_max, radius = obs
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(z_min, z_max, 100)
    x = radius * np.outer(np.cos(u), np.ones(len(v))) + x_center
    y = radius * np.outer(np.sin(u), np.ones(len(v))) + y_center
    z = np.outer(np.ones(len(u)), v)
    ax.plot_surface(x, y, z, color='red', alpha=0.3)


# Initial positions of the cubes
cube_start_positions = [alns_optimized_path[0], alns_optimized_path[0], alns_optimized_path[0]]  # All start at the beginning of the ALNS path


# Distances covered by each cube
distances_covered = [0.0, 0.0, 0.0]



# Function to get position along ALNS path
# Check that cube_positions (ALNS path) is not empty
if not alns_optimized_path:
    print("Error: ALNS optimized path is empty. Ensure ALNS path is correctly computed.")
else:
    # Proceed if alns_optimized_path has valid points
    cube_positions = [alns_optimized_path[0:], alns_optimized_path[1:], alns_optimized_path[2:]]

    # Function to get position along ALNS path with validation
# Helper function to find the interpolated position on the path based on distance
def get_position_along_path(path, distance):
    # Calculate the cumulative distances along the path
    cumulative_distances = [0]
    for i in range(1, len(path)):
        cumulative_distances.append(
            cumulative_distances[i-1] + np.linalg.norm(np.array(path[i]) - np.array(path[i-1]))
        )

    # Find the two points between which the current distance falls
    for i in range(1, len(cumulative_distances)):
        if cumulative_distances[i] >= distance:
            # Interpolate between points i-1 and i
            segment_distance = cumulative_distances[i] - cumulative_distances[i-1]
            t = (distance - cumulative_distances[i-1]) / segment_distance  # Interpolation factor
            interpolated_position = (1 - t) * np.array(path[i-1]) + t * np.array(path[i])
            return interpolated_position
    return path[-1]  # Return the last point if distance exceeds path length


#collision_radius = np.sqrt(3 * (cube_size / 2) ** 2)
collision_radius = cube_size * np.sqrt(3) * 5 / 2  # Radius for cube-to-cube collision detection

cube_plots = [
    plot_cube(ax, create_cube_vertices(cube_start_positions[i], cube_size), color=color)
    for i, color in enumerate(['blue','orange', 'green'])
]
# Define `offset_applied` globally if it should persist across frames
offset_applied = [False] * len(cube_plots)  # Initialize offset tracking for each cube

# Enhanced function to check collision-free paths that includes the cube's collision radius
def collision_free(self, start, end, safe_distance=1.0):
    """
    Checks if the path from start to end is collision-free.
    """
    step_vector = np.array(end[:3]) - np.array(start[:3])  # Ensure start and end are (x, y, z)
    distance = np.linalg.norm(step_vector)

    if distance == 0:
        return True  # No need to check if start and end are the same

    steps = int(distance / safe_distance) + 1  # Increase step granularity
    for step in range(steps + 1):
        intermediate_point = np.array(start[:3]) + step * step_vector / steps
        # Ensure intermediate_point has only (x, y, z) coordinates before checking
        intermediate_point = intermediate_point[:3]
        if any(self.intersects_obstacle(intermediate_point, obs) for obs in self.obstacles):
            return False

    return True




# Flag and start time for return delay
waiting_at_goal = [False] * len(cube_plots)  # Track if each cube is waiting at the goal
return_start_time = [None] * len(cube_plots)  # Start time for the return wait


# Define global safe distance and offset distance for the return trip
return_safe_distance = 30.0  # Safe distance when returning
return_offset_distance = 25.0  # Offset distance for collision avoidance on the return path


# Create the nibble (direction indicator) for each cube
nibble_plots = []
for color in ['blue', 'orange', 'green']:
    nibble_line, = ax.plot([], [], [], color=color, linewidth=2)  # Creates the nibble line
    nibble_plots.append(nibble_line)

cube_speeds = [random.uniform(1.5, 5.5) for _ in range(len(cube_plots))]
cube_yaws = [random.uniform(-40,40) for _ in range(len(cube_plots))]


#cube_speeds = [random.uniform(dwa_params['min_speeds'], dwa_params['max_speed']) for _ in range(len(cube_plots))]
#cube_yaws = [random.uniform(-dwa_params['max_yaw_rate'], dwa_params['max_yaw_rate']) for _ in range(len(cube_plots))]
cube_priorities = [0] * len(cube_plots)  # Start with equal priority

# Dynamically update cube priorities based on their speed




nib_length = 5.0

# Create an array to store the nib plot elements
nib_plots = [ax.plot([], [], [], color=color, linewidth=2)[0] for color in ['black', 'black', 'black']]

# Define dynamic priority update function
def update_priority(cube_speeds, battery_levels, distances_to_goal):
    """
    Dynamically updates priorities based on cube speeds, battery levels, and proximity to goal.
    """
    # Calculate priority based on conditions (you may customize this as needed)
    priority_scores = [speed + (100 - battery) + (1 / (dist_to_goal + 1)) 
                       for speed, battery, dist_to_goal in zip(cube_speeds, battery_levels, distances_to_goal)]
    
    # Assign rank based on scores
    sorted_indices = np.argsort(priority_scores)[::-1]  # Higher score gets higher priority
    return {idx: rank + 1 for rank, idx in enumerate(sorted_indices)}


yaw_angles = [0.0] * len(cube_plots)  # Start with 0 yaw for all cubes

# Length of nibble to indicate cube direction
nibble_length = 5.0  # Adjust as needed

def update_yaw_and_direction(cube_index, new_position, previous_position):
    # Convert positions to numpy arrays for element-wise operations
    new_position = np.array(new_position)
    previous_position = np.array(previous_position)
    
    # Calculate direction vector in 2D (ignoring the z-axis for yaw)
    direction_vector = new_position[:2] - previous_position[:2]
    
    # Calculate yaw as the angle of the direction vector
    yaw = np.arctan2(direction_vector[1], direction_vector[0])
    
    # Update any other state if necessary, e.g., cube heading or orientation
    # This function could return yaw or directly update cube data structures as needed
    return yaw


def calculate_nib_position(cube_index, position, yaw):
    """
    Calculates the nib position based on the cube's yaw angle for the specified cube.
    Ensures the nib appears directly in front of the cube.
    """
    nib_start = np.array(position)
    # Calculate the forward direction based on yaw
    forward_direction = np.array([np.cos(yaw), np.sin(yaw), 0])  # Forward vector in the XY plane
    nib_end = nib_start + nibble_length * forward_direction  # Offset nib in the forward direction

    return nib_start, nib_end


# Initialize path lines for each cube
cube_path_lines = [ax.plot([], [], [], color=color, linestyle='-', linewidth=1)[0]
                   for color in ['blue', 'orange', 'green']]
cube_paths = [[] for _ in range(len(cube_plots))]  # Store the path points for each cube




AVOIDANCE_FACTOR = 8.0  # Offset distance when avoiding

# Adjust constants for avoidance behavior


STOP_DISTANCE = 8.0   # Decrease from 10.0       # Distance at which cubes temporarily stop to avoid collision
MAX_PAUSE_DURATION = 2.0    # Maximum duration in seconds for which a cube can pause
ORIGINAL_SPEEDS = cube_speeds.copy()  # Store original speeds for resetting after avoidance
# Add a condition in the main loop where you call `adjust_speed`


# Modify collision avoidance logic to prevent indefinite pausing
def collision_avoidance(i, new_position, priorities, distances_covered):
    offset_applied = False  # Track if any offset was applied
    for j, other_path in enumerate(cube_positions):
        if i != j:
            other_position = np.array(get_position_along_path(other_path, distances_covered[j]), dtype=np.float64)
            distance_to_other = np.linalg.norm(new_position - other_position)

            # Adjust speed based on proximity
            if distance_to_other < SAFE_DISTANCE:
                # Apply lateral offset instead of stopping
                direction_vector = new_position - other_position
                perpendicular_offset = np.cross(direction_vector, [0, 0, 1])
                if np.linalg.norm(perpendicular_offset) > 0:
                    perpendicular_offset = (SAFE_DISTANCE / np.linalg.norm(perpendicular_offset)) * perpendicular_offset
                    new_position += perpendicular_offset
                    offset_applied = True
                    print(f"Cube {i} applied lateral offset to avoid Cube {j}.")
    
    # Ensure at least some speed is maintained to avoid deadlock
    if not offset_applied:
        cube_speeds[i] = max(cube_speeds[i] - PAUSE_SPEED_REDUCTION_RATE, 0.05)
    else:
        cube_speeds[i] = original_speeds[i]
    
    return offset_applied




# Define constants for the gradual pause mechanism
SAFE_DISTANCE = 15.0  # Minimum safe distance between cubes

STOP_DISTANCE = 10.0  # Distance at which cubes stop
PAUSE_SPEED_REDUCTION_RATE = 0.1  # Rate at which speed decreases
RECOVERY_SPEED_INCREMENT = 0.1  # Rate at which speed recovers after the conflict

PAUSE_DURATION = 2.0  # Duration in seconds for a temporary pause
RECOVERY_OFFSET_FACTOR = 1.2  # Slight offset adjustment factor for recovery

# Define the number of cubes
num_cubes = len(cube_plots)

# Initialize lists with the same length as num_cubes
pause_status = [False] * num_cubes
pause_timer = [0.0] * num_cubes
avoidance_attempts = [0] * num_cubes
is_returning = [False] * num_cubes
temporary_offset = [False] * num_cubes
paused_at_goal = [False] * num_cubes
return_start_time = [None] * num_cubes

# Initialize cube_speeds and copy to ORIGINAL_SPEEDS
cube_speeds = [3.0] * num_cubes  # Example: uniform speed for all cubes
ORIGINAL_SPEEDS = cube_speeds.copy()

# Other parameters
MAX_AVOIDANCE_ATTEMPTS = 5
SLOW_DOWN_DISTANCE = 20  # Adjust as needed



# Usage example inside a loop



OFFSET_DISTANCE = 15.0  # Offset distance for parallel path detour
GOAL_THRESHOLD = 1.0  # Threshold distance for reaching the goal
RETURN_PAUSE_DURATION = 1  # Pause duration at the goal before returning
PARALLEL_OFFSET_FACTOR = 1.0  # Factor to control offset scaling

# Flags and tracking variables for each cube
is_returning = [False] * len(cube_plots)  # Track if each cube is returning to start
temporary_offset = [False] * len(cube_plots)  # Track if each cube is in a temporary offset path
paused_at_goal = [False] * len(cube_plots)  # Track if each cube is paused at the goal
return_start_time = [None] * len(cube_plots)  # Track pause time at goal before return
AVOIDANCE_OFFSET_DISTANCE = 30.0  # Offset distance for collision avoidance

# Variables to track original speeds
original_speeds = cube_speeds.copy()  # Store each cube's original speed


# Initial battery levels for each cube


# Battery consumption parameters

speed_battery_drain_factor = 0.05  # Additional drain factor based on speed




def adjust_speed(i, new_position, distances_covered, path, stuck=False):
    # Decrease speed or allow a small backward movement if the cube is stuck
    if stuck:
        return -ORIGINAL_SPEEDS[i] * 0.1  # Small backward step

    # Adjust speed near the goal
    goal_distance = np.linalg.norm(new_position[:3] - np.array(path[-1]))
    speed = ORIGINAL_SPEEDS[i] * 0.3 if goal_distance < 50 else ORIGINAL_SPEEDS[i]

    # Reduce speed if another cube is nearby
    for j, other_path in enumerate(cube_positions):
        if i != j:
            other_position = np.array(get_position_along_path(other_path, distances_covered[j]))
            if np.linalg.norm(new_position - other_position) < SLOW_DOWN_DISTANCE:
                speed *= 0.5  # Slow down if too close to another cube

    return speed

# Safe check within the loop
#for i in range(num_cubes):  # Loop only within the range of initialized lists
#    stuck = pause_status[i] and avoidance_attempts[i] >= MAX_AVOIDANCE_ATTEMPTS
#    speed = adjust_speed(i, new_position, distances_covered, path, stuck=stuck)



# Define battery consumption parameters
base_battery_drain = 0.001  # Base drain per frame when moving at low speed
high_speed_drain_factor = 0.01  # Additional drain per frame based on high speed
detour_drain_factor = 0.02  # Extra drain during detours
idle_battery_drain = 0.0001  # Minimal drain when paused or at low speeds
recovery_rate = 0.001  # Small recovery when paused or moving very slowly

def optimized_battery_drain(cube_index, speed, is_detouring, is_paused):
    """
    Calculate the optimized battery drain based on speed, detouring status, and pause state.
    """
    if is_paused:
        # Minimal drain when the cube is paused
        return idle_battery_drain

    drain = base_battery_drain  # Start with base drain

    # Add drain based on speed (diminishing return for higher speeds)
    if speed > 1.0:
        drain += high_speed_drain_factor * (speed ** 0.7)  # Apply diminishing returns at higher speeds

    # Add extra drain if detouring (collision avoidance)
    if is_detouring:
        drain += detour_drain_factor

    # Apply slight recovery if moving very slowly
    if speed < 0.5:
        drain -= recovery_rate  # Recover a small amount to simulate reduced drain

    # Ensure the battery drain is non-negative
    return max(drain, 0)

# Update the `update_cubes` function with optimized battery drainage

def collision_free_expanded(search_space, start, end, collision_radius):
    """
    Checks if the path from start to end is collision-free, considering an expanded collision radius.
    """
    # Define the vector from start to end
    step_vector = np.array(end[:3]) - np.array(start[:3])
    distance = np.linalg.norm(step_vector)

    if distance == 0:
        return True  # No need to check if start and end are the same

    # Define the number of steps based on the distance and a safe distance step size
    steps = int(distance / collision_radius) + 1  # Increase granularity based on collision radius
    for step in range(steps + 1):
        intermediate_point = np.array(start[:3]) + step * step_vector / steps

        # Check each intermediate point for collisions with obstacles
        if any(search_space.intersects_obstacle(intermediate_point, obs) for obs in search_space.obstacles):
            return False  # Collision detected

    return True  # Path is collision-free

def check_battery_and_redirect(cube_index, battery_level, current_position, num, charging_station_position):
    if battery_level < 50:
        print(f"Cube {cube_index + 1} battery low ({battery_level}%), redirecting to charging station.")
        
        # Generate a path to the charging station
        path_to_charging_station = generate_collision_free_rrt(
            current_position, charging_station_position, X, all_obstacles
        )

        if path_to_charging_station:
            print(f"Cube {cube_index + 1} path to charging station generated successfully.")
            # Flash red color every 5 frames
            for line in cube_plots[cube_index]:
                line.set_color('red' if num % 10 < 5 else 'none')
            
            return True, path_to_charging_station  # Return the path for redirection
        else:
            print(f"Cube {cube_index + 1} failed to generate path to charging station.")
            return True, None  # Flash but don't redirect if path generation fails

    # Set cube color back to normal if battery level is sufficient
    for line in cube_plots[cube_index]:
        line.set_color(['blue', 'orange', 'green'][cube_index])  # Reset to original color based on index

    return False, None  # No flashing, continue on current path


def on_key_press(event):
    global cube1_to_charging_station, cube1_rrt_path_to_charging_station, cube1_paused_for_path
    if event.key == '1':
        print("Key '1' detected - initiating path generation for Cube 1.")
        
        # Set the flag to pause Cube 1 and prepare to generate the path
        cube1_paused_for_path = True
        cube1_to_charging_station = False  # Reset movement flag until path is ready
        cube1_rrt_path_to_charging_station = None  # Clear any existing path

# Connect the key press event to the plot
fig.canvas.mpl_connect('key_press_event', on_key_press)


# Define this at the top of your script
max_avoidance_attempts = 5
charging_station_position = np.array([0, -150, 0])  # Coordinates for the charging station
cube1_paused_for_path = False  # Pauses Cube 1 for path generation
cube1_to_charging_station = False  # Indicates Cube 1 should follow the charging station path
cube1_rrt_path_to_charging_station = None  # Stores the generated RRT path to the charging station

cube1_waiting_at_charging = False  # To track if Cube 1 is waiting at the charging station
wait_start_time = None  # Track the start time for Cube 1’s wait at the charging station
# Other parameters


# Update the `update_cubes` function
def update_cubes(num):
    global distances_covered, cube_times, waiting_at_goal, return_start_time
    global cube_speeds, pause_status, battery_levels, cube_positions
    global cube1_to_charging_station, cube1_rrt_path_to_charging_station
    global cube1_paused_for_path, cube1_waiting_at_charging, wait_start_time

    # Define constants for avoidance and movement thresholds
    max_avoidance_attempts = 5
    min_safe_distance = 30.0
    GOAL_THRESHOLD = 5.0
    RETURN_PAUSE_DURATION = 2.0

    current_time = time.time()
    for i, (cube, path, speed) in enumerate(zip(cube_plots, cube_positions, cube_speeds)):
        current_position = get_position_along_path(path, distances_covered[i])
        cube_name = f'cube{i+1}'
        # Check if Cube 1 is set to generate a path to the charging station
        if i == 0 and cube1_paused_for_path:
            print("Attempting to generate path for Cube 1 to the charging station...")
            cube1_rrt_path_to_charging_station = generate_collision_free_rrt(current_position[:3], charging_station_position, X, all_obstacles)
            
            if cube1_rrt_path_to_charging_station:
                print("Path to charging station generated successfully.")
                cube1_to_charging_station = True
                cube1_paused_for_path = False  # Clear the paused flag to allow movement
                distances_covered[i] = 0  # Reset distance to start moving along the new path
                cube_positions[i] = cube1_rrt_path_to_charging_station
            else:
                print("Failed to generate path to charging station.")
                cube1_paused_for_path = False  # Unpause if path fails
                cube1_to_charging_station = False

        # If Cube 1 is following the path to the charging station
        elif i == 0 and cube1_to_charging_station and cube1_rrt_path_to_charging_station:
            distances_covered[i] += speed
            new_position = get_position_along_path(cube1_rrt_path_to_charging_station, distances_covered[i])

            if np.linalg.norm(new_position[:3] - charging_station_position) < GOAL_THRESHOLD:
                print("Cube 1 reached the charging station.")
                cube1_to_charging_station = False
                cube1_rrt_path_to_charging_station = None
                cube1_waiting_at_charging = True
                wait_start_time = time.time()

        # Handle waiting at the charging station
            elif i == 0 and cube1_waiting_at_charging:
                if time.time() - wait_start_time >= WAIT_DURATION:
                    print("Cube 1 finished waiting. Returning to start position.")
                    cube1_waiting_at_charging = False
                    cube_positions[i] = path[::-1]  # Reverse path for return
                    distances_covered[i] = 0  # Reset distance for return journey
                cube1_to_charging_station = False  # Reset to avoid re-triggering

        # Standard movement logic for all cubes if no special conditions are met
        else:
            distances_covered[i] += speed
            new_position = get_position_along_path(path, distances_covered[i])


        # Step 4: Movement logic for other cubes and standard movement for Cube 1
        previous_position = get_position_along_path(path, max(0, distances_covered[i] - speed))
        distances_covered[i] += speed
        new_position = get_position_along_path(path, distances_covered[i])

        # Collision avoidance
        if not collision_free_expanded(X, previous_position, new_position, collision_radius):
            perpendicular_offset = np.cross(new_position - previous_position, [0, 0, 1])
            if np.linalg.norm(perpendicular_offset) > 0:
                perpendicular_offset = (AVOIDANCE_OFFSET_DISTANCE / np.linalg.norm(perpendicular_offset)) * perpendicular_offset
                new_position += perpendicular_offset
                print(f"Cube {i} applied lateral offset to avoid collision.")

        # Ensure new_position fallback if None
        if new_position is None:
            new_position = path[-1]
        if previous_position is None:
            previous_position = path[0]

        # Update battery level and apply battery consumption
        battery_drain = optimized_battery_drain(i, cube_speeds[i], pause_status[i], False)
        battery_levels[i] = max(battery_levels[i] - battery_drain, 0)

        # Display time, speed, and distance information
        elapsed_time = (
            cube_times[cube_name]['end'] - cube_times[cube_name]['start']
            if cube_times[cube_name]['end'] is not None
            else current_time - cube_times[cube_name]['start']
        )
        time_texts[i].set_text(f"Cube {i + 1}-Time:{elapsed_time:.2f}s;")
        speed_texts[i].set_text(f"Cube {i + 1}-Speed:{cube_speeds[i]:.2f}m/s; ")
        distance_texts[i].set_text(f"Cube {i + 1}-Dis Trav:{distances_covered[i]:.2f}m; ")
        battery_texts[i].set_text(f"Cube {i + 1}-Battery:{battery_levels[i]:.2f}%")

        # Update cube’s path in the plot
        cube_paths[i].append(new_position)
        if len(cube_paths[i]) > 1:
            x_vals, y_vals, z_vals = zip(*cube_paths[i])
            cube_path_lines[i].set_data(x_vals, y_vals)
            cube_path_lines[i].set_3d_properties(z_vals)

        # Yaw and nib position updates
        yaw = update_yaw_and_direction(i, new_position, previous_position)
        nib_start, nib_end = calculate_nib_position(i, new_position, yaw)
        nib_plots[i].set_data([nib_start[0], nib_end[0]], [nib_start[1], nib_end[1]])
        nib_plots[i].set_3d_properties([nib_start[2], nib_end[2]])

        # Update cube vertices based on new position
        cube_vertices = create_cube_vertices(new_position, cube_size)
        for line, (p1, p2) in zip(cube, [
            (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
            (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)
        ]):
            line.set_data([cube_vertices[p1][0], cube_vertices[p2][0]],
                          [cube_vertices[p1][1], cube_vertices[p2][1]])
            line.set_3d_properties([cube_vertices[p1][2], cube_vertices[p2][2]])

    return [line for cube in cube_plots for line in cube] + time_texts + speed_texts + distance_texts + battery_texts + nib_plots + cube_path_lines


# --- Start Animation with Validated Paths ---
if paths:
    longest_path_length = max(len(path) for path in paths if path is not None)
    ani = animation.FuncAnimation(fig, update_cubes, frames=longest_path_length, interval=100, blit=True)
else:
    print("Error: No valid paths generated. Exiting animation setup.")


# Define the charging station position


# Plot the charging station as a red marker with a label
ax.scatter(*charging_station, color='red', s=80, label="Charging Station")
ax.text(charging_station[0], charging_station[1], charging_station[2], "Charging Station", color='red', fontsize=12, ha='center')



# Plot paths, cubes, and other markers as needed...

# Add legend and show plot
ax.legend()
plt.show()



