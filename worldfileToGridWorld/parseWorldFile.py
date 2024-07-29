import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

def parse_sdf(sdf_file):
    tree = ET.parse(sdf_file)
    root = tree.getroot()
    return root

def extract_objects(root):
    objects = []
    for model in root.findall('.//model'):
        model_name = model.get('name')
        pose = model.find('pose')
        if pose is not None:
            position = pose.text.split()
            position = [float(x) for x in position[:2]]  # Only consider x, y positions
        else:
            position = [0, 0]  # Default position if pose is missing
        
        for link in model.findall('link'):
            for collision in link.findall('collision'):
                for geometry in collision.findall('geometry'):
                    for box in geometry.findall('box'):
                        size = box.find('size').text.split()
                        size = [float(x) for x in size]
                        objects.append((model_name, position, size))
    return objects

def create_2d_grid(objects, grid_size, world_size):
    grid_world = np.zeros((grid_size, grid_size))
    scale = grid_size / world_size
    
    for obj in objects:
        model_name, position, size = obj
        grid_pos = (int(position[0] * scale), int(position[1] * scale))
        grid_dimensions = (int(size[0] * scale), int(size[1] * scale))
        
        x_min = max(0, grid_pos[0] - grid_dimensions[0] // 2)
        x_max = min(grid_size - 1, grid_pos[0] + grid_dimensions[0] // 2)
        y_min = max(0, grid_pos[1] - grid_dimensions[1] // 2)
        y_max = min(grid_size - 1, grid_pos[1] + grid_dimensions[1] // 2)
        
        grid_world[y_min:y_max+1, x_min:x_max+1] = 1  # Mark as occupied
    
    return grid_world

def plot_grid_world(grid_world):
    plt.figure(figsize=(8, 8))
    plt.imshow(grid_world, cmap='gray_r', origin='upper')
    plt.title("2D Grid Representation of the SDF World")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()

def main():
    sdf_file = 'crazysim_mod.sdf'  # Path to your SDF file
    grid_size = 40  # Define the size of the grid
    world_size = 10.0  # Define the real-world size of the environment (e.g., 10x10 meters)

    root = parse_sdf(sdf_file)
    objects = extract_objects(root)
    grid_world = create_2d_grid(objects, grid_size, world_size)
    
    # Plot the grid world
    plot_grid_world(grid_world)

if __name__ == "__main__":
    main()
