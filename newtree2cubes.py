import os
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d

def farthest_point_sampling(points, n_samples):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    down_pcd = pcd.farthest_point_down_sample(n_samples)
    return np.asarray(down_pcd.points)

def cut_point_cloud(
    point_cloud,
    outpath,
    cube_sizes,           # list of 4 floats specifying cube sizes
    min_points=500,       # minimum points to save a cube
    max_points=2048,      # max points allowed in a cube before downsampling
    target_points=2048,   # number of points after downsampling
    offsets=None,         # list of 4 np.array(3,) for grid offsets
    sampling_method='fps',# sampling method: 'fps' or 'random'
    progress_bar=None,    # optional progress bar object with .progress(float)
    status_text=None      # optional text display object with .text(str)
):
    assert isinstance(point_cloud, np.ndarray) and point_cloud.shape[1] == 3, "point_cloud must be Nx3 numpy array"
    assert len(cube_sizes) == 4, "cube_sizes must contain exactly 4 elements"

    if offsets is None:
        offsets = [
            np.array([0, 0, 0]),
            np.array([cube_sizes[1]/2]*3),
            np.array([-cube_sizes[2]/2]*3),
            np.array([0, 0, -0.3])
        ]
    else:
        assert len(offsets) == 4, "offsets must contain exactly 4 elements"

    os.makedirs(outpath, exist_ok=True)

    min_coords_global = np.min(point_cloud, axis=0)
    max_coords_global = np.max(point_cloud, axis=0)

    total_cubes = 0
    for cube_size in cube_sizes:
        bbox = max_coords_global - min_coords_global
        bbox_aligned = np.ceil(bbox / cube_size) * cube_size
        n_x, n_y, n_z = (bbox_aligned / cube_size).astype(int)
        total_cubes += n_x * n_y * n_z

    processed_cubes = 0
    saved_cubes = 0

    tree = cKDTree(point_cloud)

    for grid_idx, cube_size in enumerate(cube_sizes):
        offset = offsets[grid_idx]
        min_coords = min_coords_global + offset
        bbox = max_coords_global - min_coords_global
        bbox_aligned = np.ceil(bbox / cube_size) * cube_size
        n_x, n_y, n_z = (bbox_aligned / cube_size).astype(int)

        grid_folder = os.path.join(outpath, f"grid_{grid_idx+1}_size_{cube_size}")
        os.makedirs(grid_folder, exist_ok=True)

        for i in range(n_x):
            for j in range(n_y):
                for k in range(n_z):
                    cube_min = min_coords + np.array([i, j, k]) * cube_size
                    cube_max = cube_min + cube_size

                    center = (cube_min + cube_max) / 2
                    radius = cube_size * (3**0.5) / 2

                    idx_in_cube = tree.query_ball_point(center, radius)
                    points_in_cube = point_cloud[idx_in_cube]

                    mask = np.all((points_in_cube >= cube_min) & (points_in_cube <= cube_max), axis=1)
                    points_in_cube = points_in_cube[mask]

                    if len(points_in_cube) < min_points:
                        processed_cubes += 1
                        continue

                    if len(points_in_cube) > max_points:
                        if sampling_method == 'fps':
                            points_in_cube = farthest_point_sampling(points_in_cube, target_points)
                        else:
                            choice = np.random.choice(len(points_in_cube), target_points, replace=False)
                            points_in_cube = points_in_cube[choice]

                    filename = f"cube_grid{grid_idx+1}_size{cube_size}_i{i}_j{j}_k{k}.txt"
                    filepath = os.path.join(grid_folder, filename)
                    np.savetxt(filepath, points_in_cube, fmt='%.16f')

                    saved_cubes += 1
                    processed_cubes += 1

                    if progress_bar and processed_cubes % 10 == 0:
                        progress_bar.progress(min(1.0, processed_cubes / total_cubes))
                    if status_text and processed_cubes % 10 == 0:
                        percent = int((processed_cubes / total_cubes) * 100)
                        status_text.text(f"Processing cubes... {percent}%")

    if status_text:
        status_text.text(f"✅ Done. {saved_cubes} cubes saved in {len(cube_sizes)} grids.")
