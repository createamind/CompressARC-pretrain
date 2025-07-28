import numpy as np
from typing import Dict, Any, List, Tuple, DefaultDict
from collections import defaultdict

def determine_background_color(
    task_data: Dict[str, Any],
    pixel_threshold_pct: int = 40,
    debug: bool = False,
) -> int:
    """Analyze all training grids and return the dominant background color."""
    if debug:
        print(f"Determining background color with threshold: {pixel_threshold_pct}%")

    all_grids: List[List[List[int]]] = []
    for example in task_data.get("train", []):
        all_grids.append(example.get("input"))
        all_grids.append(example.get("output"))

    color_total_percentages: Dict[int, float] = defaultdict(float)
    color_appearance_count: Dict[int, int] = defaultdict(int)

    for grid in all_grids:
        if not grid:
            continue
        total_pixels = len(grid) * len(grid[0])
        if total_pixels == 0:
            continue

        color_counts: Dict[int, int] = defaultdict(int)
        for row in grid:
            for cell in row:
                color_counts[cell] += 1

        for color, count in color_counts.items():
            percentage = count / total_pixels * 100
            color_total_percentages[color] += percentage
            color_appearance_count[color] += 1

    color_avg_percentages: Dict[int, float] = {
        color: color_total_percentages[color] / color_appearance_count[color]
        for color in color_total_percentages
    }

    sorted_colors = sorted(
        color_avg_percentages.items(), key=lambda x: x[1], reverse=True
    )

    if debug:
        print("Color distribution across training data:")
        for col, pct in sorted_colors:
            print(f"  color {col}: {pct:.2f}%")

    if sorted_colors:
        max_color, max_percentage = sorted_colors[0]
        if max_percentage >= pixel_threshold_pct:
            if debug:
                print(f"确定全局背景色: {max_color} (占比: {max_percentage:.2f}%)")
            return max_color
    return 0  # Default to 0 if no background color is determined

def nonbg_pixels(grid, bg_color):
    """Return list of non-background pixels as (row, col) coordinates."""
    result = []
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell != bg_color:
                result.append((i, j))
    return result

def count_non_background_pixels(task_data: Dict[str, Any], pixel_threshold_pct: int) -> int:
    """Return the total number of non-background pixels in all train grids."""
    bg_color = determine_background_color(task_data, pixel_threshold_pct=pixel_threshold_pct, debug=False)
    count = 0
    for pair in task_data.get("train", []):
        for kind in ("input", "output"):
            grid = pair.get(kind, [])
            count += len(nonbg_pixels(grid, bg_color))
    return count

def compute_task_complexity(task_data, bg_threshold=40):
    """Compute task complexity based on non-background pixels, colors, and grid size"""
    # Count non-background pixels
    non_bg_count = count_non_background_pixels(task_data, bg_threshold)
    
    # Count unique colors
    all_colors = set()
    for example in task_data.get("train", []):
        for kind in ("input", "output"):
            grid = example.get(kind, [])
            for row in grid:
                for cell in row:
                    all_colors.add(cell)
    
    # Get max grid dimensions
    max_h, max_w = 0, 0
    for example in task_data.get("train", []):
        for kind in ("input", "output"):
            grid = example.get(kind, [])
            if grid:
                max_h = max(max_h, len(grid))
                max_w = max(max_w, len(grid[0]) if grid else 0)
    
    # Calculate complexity score
    grid_size_factor = np.log(max(1, max_h * max_w))
    color_factor = len(all_colors)
    pixel_factor = non_bg_count
    
    return pixel_factor * color_factor * grid_size_factor

def grid_augmentation(grid, method=None):
    """Apply data augmentation to a grid"""
    if method is None:
        # Randomly choose an augmentation method
        method = np.random.choice(['rotate', 'flip', 'color_permute', 'none'], p=[0.2, 0.2, 0.2, 0.4])
    
    if method == 'rotate':
        k = np.random.randint(1, 4)  # Rotate 90, 180, or 270 degrees
        return np.rot90(grid, k=k)
    elif method == 'flip':
        axis = np.random.randint(0, 2)
        return np.flip(grid, axis=axis)
    elif method == 'color_permute':
        colors = np.unique(grid)
        if len(colors) > 1:
            permutation = np.random.permutation(colors)
            color_map = {old: new for old, new in zip(colors, permutation)}
            new_grid = np.zeros_like(grid)
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    new_grid[i, j] = color_map[grid[i, j]]
            return new_grid
    
    # 'none' or default case
    return grid.copy()