# The new version of tree2cubes, the library for converting tree point clouds to cubes before the inference for completion.
# This version uses farthest point sampling (FPS) to downsample the point clouds in each cube.
# The FPS method is more efficient and effective than the previous method of random sampling.

import numpy as np
import os
import open3d as o3d

def farthest_point_sampling(points, n_samples):
    """
    Downsample a point cloud using Farthest Point Sampling (FPS).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    down_pcd = pcd.farthest_point_down_sample(n_samples)
    return np.asarray(down_pcd.points)

def scale_adaptive_cut_with_fps(
    point_cloud,
    outpath,
    scale_fraction=0.05,
    min_points=500,
    max_points=2048,
    target_points=2048,
    progress_bar=None,
    status_text=None
):
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # Bounding box
    min_coords = np.min(point_cloud, axis=0)
    max_coords = np.max(point_cloud, axis=0)
    bbox = max_coords - min_coords

    max_bbox_dim = np.max(bbox)  # Use the largest dimension to make cubes
    cube_size = max_bbox_dim * scale_fraction  # ✅ Move this up here

    # Align the bounding box so it's a multiple of cube_size
    bbox_aligned = np.ceil(bbox / cube_size) * cube_size
    max_coords_aligned = min_coords + bbox_aligned


    # Number of cubes in each direction
    num_cubes_x = int(bbox_aligned[0] / cube_size) # Number of cubes in x direction
    num_cubes_y = int(bbox_aligned[1] / cube_size)
    num_cubes_z = int(bbox_aligned[2] / cube_size)


    # Prepare iteration
    cube_indices = [(i, j, k) for i in range(num_cubes_x)
                              for j in range(num_cubes_y)
                              for k in range(num_cubes_z)]
    total = len(cube_indices)
    saved_count = 0

    # Iterate over each cube
    for idx, (i, j, k) in enumerate(cube_indices):
        cube_min = min_coords + np.array([i, j, k]) * cube_size
        cube_max = cube_min + cube_size

        mask = np.all((point_cloud >= cube_min) & (point_cloud < cube_max), axis=1)
        points_in_cube = point_cloud[mask]

        if len(points_in_cube) < min_points:
            continue

        if len(points_in_cube) > max_points:
            points_in_cube = farthest_point_sampling(points_in_cube, target_points)

        filename = f"/cube_{i}_{j}_{k}.txt"
        np.savetxt(outpath + filename, points_in_cube[:, :3], fmt='%.16f', delimiter=' ')
        saved_count += 1

        # Update progress bar
        if progress_bar is not None:
            percent = (idx + 1) / total
            progress_bar.progress(min(1.0, percent))
        if status_text is not None:
            status_text.text(f"Splitting tree into cubes... {int(percent * 100)}%")

    if status_text is not None:
        status_text.text(f"✅ Done. {saved_count} cubes saved.")
