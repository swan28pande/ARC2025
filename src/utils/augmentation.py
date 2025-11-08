import numpy as np

# --------------------------
# 1. Rotation
# --------------------------
def rotate_grid(grid, angle):
    """
    Rotate a grid (numpy array) by 90, 180, or 270 degrees.
    Args:
        grid (np.ndarray): Input grid.
        angle (int): Rotation angle in degrees. Must be one of [90, 180, 270].
    Returns:
        np.ndarray: Rotated grid.
    """
    if angle not in [90, 180, 270]:
        raise ValueError("Angle must be 90, 180, or 270")
    k = angle // 90
    return np.rot90(grid, k=k)  # rot90 rotates counter-clockwise

# --------------------------
# 2. Color Change
# --------------------------
def change_color(grid, color_map=None):
    """
    Change colors/symbols in the grid consistently.
    Args:
        grid (np.ndarray): Input grid.
        color_map (dict): Optional dict mapping old_color -> new_color.
                          If None, random mapping is created.
    Returns:
        np.ndarray: Grid with colors changed.
    """
    grid = np.array(grid)
    unique_colors = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    if color_map is None:
        shuffled_colors = np.random.permutation(unique_colors)
        color_map = {old: new for old, new in zip(unique_colors, shuffled_colors)}
    
    new_grid = grid.copy()
    for old_color, new_color in color_map.items():
        new_grid[grid == old_color] = new_color
    return new_grid, color_map

# --------------------------
# 3. Horizontal Flip
# --------------------------
def flip_horizontal(grid):
    """
    Flip grid horizontally.
    Args:
        grid (np.ndarray): Input grid.
    Returns:
        np.ndarray: Horizontally flipped grid.
    """
    return np.fliplr(grid)

# --------------------------
# 4. Vertical Flip
# --------------------------
def flip_vertical(grid):
    """
    Flip grid vertically.
    Args:
        grid (np.ndarray): Input grid.
    Returns:
        np.ndarray: Vertically flipped grid.
    """
    return np.flipud(grid)
