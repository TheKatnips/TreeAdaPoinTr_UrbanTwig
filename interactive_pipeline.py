import streamlit as st
import os
import subprocess
import glob
from pathlib import Path
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pyntcloud import PyntCloud
import plotly.express as px
import torch
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from datetime import datetime
import time
from tqdm import tqdm
import re

# Load the custom modules
import newtree2cubes

# Update (or create) a CSV file that logs all test attempts

def update_summary_table(
    model_name: str,
    tree_name: str,
    chunk_params: str,
    nb_chunks: int,
    cut_time: float,
    pts_before: int,
    pts_after_cut: int,
    pts_after_inf: int,
    inference_time: float,
    chamfer_l1: float,
    chamfer_l2: float,
    chamfer_l1_with_flips: float,
    chamfer_l2_with_flips: float,
    chamfer_time: float,
    summary_path: str = "summary_tests.csv"
):

    now = datetime.now()
    new_row = {
        "model": model_name,
        "tree": tree_name,
        "chunk_params": chunk_params,
        "nb_chunks": nb_chunks,
        "cut_time": cut_time,
        "pts_before_cut": pts_before,
        "pts_after_cut": pts_after_cut,
        "pts_after_inference": pts_after_inf,
        "inference_time": inference_time,
        "chamfer_l1": chamfer_l1,
        "chamfer_l2": chamfer_l2,
        "chamfer_l1_with_flips": chamfer_l1_with_flips,
        "chamfer_l2_with_flips": chamfer_l2_with_flips,
        "chamfer_time": chamfer_time,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S")
    }

    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)

        # Check if a row with same model, tree, and chunk_params exists
        match = (
            (df["model"] == model_name) &
            (df["tree"] == tree_name) &
            (df["chunk_params"] == chunk_params)
        )

        if match.any():
            # Update the first matching row
            st.warning(f"⚠️ Results for this configuration already exist. Skipping update.")
        else:
            # Append as a new row
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(summary_path, index=False)
    print(f"✅ Summary updated at {summary_path}")


# Function to split the tree into chunks
def cut_point_cloud(point_cloud, outpath, scale_fraction):
    import time
    import glob
    import numpy as np

    start_cut_time = time.time()
    pts_before = point_cloud.shape[0]

    # Create Streamlit progress UI
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Run the chunking with FPS and progress
    newtree2cubes.scale_adaptive_cut_with_fps(
        point_cloud=point_cloud,
        outpath=str(outpath),
        scale_fraction=scale_fraction,
        min_points=500,
        max_points=2048,
        target_points=2048,
        progress_bar=progress_bar,
        status_text=status_text
    )

    # Count points after cutting
    chunk_files = glob.glob(str(outpath) + "/cube*.txt")
    pts_after_cut = sum(np.loadtxt(f).shape[0] for f in chunk_files)

    cut_time = time.time() - start_cut_time

    # Clear the progress bar and status text
    progress_bar.empty()
    status_text.empty()

    # ✅ Display results
    st.write(f"Total points before cutting: {pts_before}")
    st.write(f"Total points after cutting: {pts_after_cut}")
    st.write(f"Cutting time: {cut_time:.2f} seconds")

    return pts_before, pts_after_cut, cut_time

# Function to generate flipped versions of the chunks
def generate_flipped_chunks(outpath):
    for files in glob.glob(str(outpath) + "/*.txt"):
        data = np.loadtxt(files)
        filename = os.path.basename(files)
        flipfile = np.column_stack((data[:, 2], data[:, 1], data[:, 0]))
        flipped_file_path = outpath / (filename.replace(".txt", "_flip.txt"))
        np.savetxt(flipped_file_path, flipfile)

# Function to run inference
def run_inference(data_path, outpath, cfg_file, ckpt_file):
    # Record start time for inference
    start_inference_time = time.time()

    if not os.path.exists(outpath):
        os.makedirs(outpath)
        print(f"Folder {outpath} created successfully.")
    
    script_path = project_root / "tools" / "inference.py"
    command = [
        sys.executable, str(script_path), str(cfg_file), str(ckpt_file),
        "--pc_root", str(data_path),
        "--out_pc_root", str(outpath),
        "--save_xyz"
    ]
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(project_root))
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error while executing the command: {e}")
        print(f"Standard output: {e.stdout}")
        print(f"Error: {e.stderr}")
    
    # Record end time for inference
    end_inference_time = time.time()
    inference_time = end_inference_time - start_inference_time  # Duration of inference operation

    return inference_time

def load_xyz(file_path):
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            st.error(f"❌ File not found: {file_path}")
            return None  # Return None if the file is not found

        # Check if the first line contains a header
        with open(file_path, 'r') as f:
            first_line = f.readline()
            has_header = any(char.isalpha() for char in first_line)

        # Load the data, skipping the header if it exists
        data = np.loadtxt(file_path, delimiter=" ", skiprows=1 if has_header else 0, usecols=(0, 1, 2))

        # Create DataFrame
        df = pd.DataFrame(data, columns=["x", "y", "z"])
        return df  # Return the dataframe for use in other functions

    except Exception as e:
        st.error(f"❌ Failed to open file: {file_path}")
        st.exception(e)  # Capture the full exception message

# Visualization of point clouds
def visualize_xyz(file_path, title):
    try:
        df = load_xyz(file_path)
        if df is None:
            return  # Return if the file opening failed

        # Sample 100,000 points or fewer
        df = df.sample(n=min(len(df), 100_000))

        # Plot the 3D scatter plot
        fig = px.scatter_3d(df, x='x', y='y', z='z', opacity=0.6, title=title)
        fig.update_traces(marker=dict(size=1))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error("❌ Failed to visualize point cloud.")
        st.exception(e)  # Capture the full exception message

# Function to visualize the chopped chunks, ignoring 'flip' in the filename
def visualize_chunks(outpath_chunks):
    # Get all chunk files that do not have 'flip' in their name
    chunk_files = glob.glob(str(outpath_chunks) + "/cube*.txt")
    chunk_files = [file for file in chunk_files if 'flip' not in file]
    
    if not chunk_files:
        st.warning(f"❌ No chunks found in {outpath_chunks} (excluding 'flip' files)")
        return

    # Initialize an empty DataFrame to store the points
    all_chunks_points = np.empty((0, 3))
    
    # Load each chunk and append it to the DataFrame
    for chunk_file in chunk_files:
        chunk_data = np.loadtxt(chunk_file, delimiter=" ")
        all_chunks_points = np.vstack((all_chunks_points, chunk_data))
    
    # Create a DataFrame from the chunk points
    chunk_df = pd.DataFrame(all_chunks_points, columns=["x", "y", "z"])
    
    # Create the visualization
    fig = px.scatter_3d(chunk_df, x="x", y="y", z="z", opacity=0.6, title="Chopped Chunks (Excluding 'flip')")
    fig.update_traces(marker=dict(size=1))
    st.plotly_chart(fig, use_container_width=True)



######################################### Streamlit UI ###########################################################

st.title("treePoinTr v2 - Point Cloud Completion")

# Model and dataset selection
model_name = st.selectbox("Select a model", ["TheGrove80realTLS20", "supermodel"])
tree_name = st.selectbox("Select a dataset", ["tls_labo_1_ultra_high", "tls_labo_2", "tls_BLK360_T05_Fagus", "tls_BLK360_T06_Fagus", "tls_BLK360_T17_Fagus", "tls_BLK360_T12_Quercus", "tls_real_rennes_Qru_01", "colmap_real_rennes_Qru_01"])

# Chunking parameters
scale_fraction = st.slider(
    "Relative cube size (% of tree size)", 
    min_value=0.5, 
    max_value=100.0, 
    value=5.0, 
    step=1.0
) / 100.0

# Define paths
project_root = Path(__file__).resolve().parent
model_path = project_root / "models_for_inference" / model_name
tree_dir = project_root / "data_clean" / tree_name
inference_dir = project_root / "inferences"

infile = tree_dir / f"{tree_name}.xyz"
chunk_params = f"scale_{int(scale_fraction * 100)}"
outpath_chunks = tree_dir / f"chunks_{chunk_params}"
pred_path = inference_dir / model_name / tree_name / f"inf_chunks_{chunk_params}"

st.write("🌳 Input tree:", infile)
st.write("📦 Chunk directory:", outpath_chunks)
st.write("📤 Inference output directory:", pred_path)

# Initialize the point cloud
if 'point_cloud' not in st.session_state:
    st.session_state.point_cloud = None

# Load the point cloud
if st.button("Load point cloud"):
    try:
        # Check if the first line contains a header
        with open(infile, 'r') as f:
            first_line = f.readline()
            has_header = any(char.isalpha() for char in first_line)

        # Load the data, skipping the header if it exists
        data = np.loadtxt(infile, skiprows=1 if has_header else 0, usecols=(0, 1, 2), delimiter=" ")

        st.session_state.point_cloud = data
        st.success("✅ Point cloud loaded successfully.")
        st.write(f"Shape: {data.shape}")
    except Exception as e:
        st.error("❌ Failed to load point cloud.")
        st.exception(e)

# Split the point cloud into cubes
if st.button("Split tree into cubes"):
    if st.session_state.point_cloud is not None:
        # Check if chunks already exist
        if outpath_chunks.exists() and any(outpath_chunks.glob("*.txt")):
            st.warning(f"⚠️ The chunks already exist in {outpath_chunks}. Step skipped.")
        else:
            # Split the point cloud into chunks
            pts_before, pts_after_cut, cut_time = cut_point_cloud(st.session_state.point_cloud, outpath_chunks, scale_fraction) # Cut the point cloud and count the points
            st.session_state.cut_time = cut_time # Store the cutting time
            st.session_state.pts_before_cut = pts_before # Store the number of points before cutting
            st.session_state.pts_after_cut = pts_after_cut # Store the number of points after cutting
            # Generate flipped versions of the chunks
            generate_flipped_chunks(outpath_chunks) # Generate flipped versions
            st.success(f"✅ Chunks successfully created and flipped versions generated.")
    else:
        st.error("❌ Please load the point cloud first.")


# Run inference with the selected model
if st.button("Run inference"):
    if st.session_state.point_cloud is not None:
        if pred_path.exists() and any(pred_path.glob("*.xyz")):
            st.warning(f"⚠️ Inferences already exist in {pred_path}. Step skipped.")
        else:
            model_cfg_dir = project_root / "cfgs" / f"{model_name}_models"
            adapointr_cfg = model_cfg_dir / "AdaPoinTr.yaml"
            pointr_cfg = model_cfg_dir / "PoinTr.yaml"

            if adapointr_cfg.exists():
                cfg_file = adapointr_cfg
            elif pointr_cfg.exists():
                cfg_file = pointr_cfg
            else:
                st.error(f"❌ No config file found in {model_cfg_dir}")
                st.stop()

            ckpt_file = model_path / "ckpt-best.pth"

            # Run inference and store values
            inference_time = run_inference(outpath_chunks, pred_path, cfg_file, ckpt_file)
            st.session_state.inference_time = inference_time # Store the inference time
            
            st.success(f"✅ Inference successfully launched. Results in {pred_path}")
            st.write(f"Inference time: {inference_time:.2f} seconds")
    else:
        st.error("❌ Please load the point cloud first.")

# Display the terminal command
if st.button("Show terminal command"):
    cfg_file = model_path / "cfgs" / f"{model_name}_models" / "PoinTr.yaml"  # or AdaPoinTr.yaml
    ckpt_file = model_path / "ckpt-best.pth"
    command = f"""python tools/inference.py "{cfg_file}" "{ckpt_file}" --pc_root "{outpath_chunks}" --out_pc_root "{pred_path}" --save_xyz"""
    st.code(command, language='bash')
    st.write("Copy and paste this command into your terminal to run the inference manually.")

# Assemble the normal and flipped results
if st.button("Assemble predicted point clouds"):
    files_found = glob.glob(str(pred_path) + "/*.xyz") 
    if not files_found:
        st.warning("No .xyz files found in the inference folder.")
    elif (pred_path / "completion.xyz").exists() or (pred_path / "completion_withflips.xyz").exists():
        st.warning("⚠️ Merged files already present. Step skipped to avoid overwriting.")
        st.code(str(pred_path / "completion.xyz"))
        st.code(str(pred_path / "completion_withflips.xyz"))
    else:
        full_pred = np.empty((0, 3)) 
        pred1 = np.empty((0, 3))
        predflip = np.empty((0, 3))

        for file in files_found:
            data = np.loadtxt(file, delimiter=" ") 
            filename = os.path.basename(file)
            newfile = np.column_stack((data[:, 2], data[:, 1], data[:, 0]))  # swap x <-> z if necessary

            if "flip" in filename:
                predflip = np.concatenate((predflip, newfile), 0)   
            else:
                newfile = data  # keep original coordinates
                pred1 = np.concatenate((pred1, newfile), 0)

        # Safe writing only if the files do not already exist
        np.savetxt(pred_path / "completion.xyz", pred1, fmt="%.16f")
        np.savetxt(pred_path / "completion_withflips.xyz", predflip, fmt="%.16f")

        st.success("✅ Prediction merging complete.")
        st.write("📂 Saved files:")
        st.code(str(pred_path / "completion.xyz"))
        st.code(str(pred_path / "completion_withflips.xyz"))


st.header("Visualization of Point Clouds")

# Original point cloud
if st.checkbox("Show original point cloud"):
    visualize_xyz(infile, "Original point cloud")

# Chunks
if st.checkbox("Show point cloud after cutting (chunks)"):
    visualize_chunks(outpath_chunks)

# Normal completion
if st.checkbox("Show normal prediction (without flips)"):
    visualize_xyz(pred_path / "completion.xyz", "Prediction without flips")

# Completion with flips
if st.checkbox("Show prediction with flips"):
    visualize_xyz(pred_path / "completion_withflips.xyz", "Prediction with flips")


# Compute Chamfer Distance
st.header("Chamfer Distance Evaluation")

if st.checkbox("Compute Chamfer Distance"):
    try:
        # Record start time for Chamfer Distance computation
        start_chamfer_time = time.time()

        # Load GT (ground truth) and predicted point clouds
        gt_points_df = load_xyz(infile)  
        gt_array = gt_points_df[["x", "y", "z"]].values  
        gt_points = torch.tensor(gt_array, dtype=torch.float32).unsqueeze(0).cuda()
 
        pred1_points = torch.tensor(np.loadtxt(pred_path / "completion.xyz", delimiter=" "), dtype=torch.float32).unsqueeze(0).cuda()
        predflip_points = torch.tensor(np.loadtxt(pred_path / "completion_withflips.xyz", delimiter=" "), dtype=torch.float32).unsqueeze(0).cuda()

        # Initialize Chamfer distance modules
        chamfer_l1 = ChamferDistanceL1()
        chamfer_l2 = ChamferDistanceL2()

        # Compute distances
        loss_l1_pred1 = chamfer_l1(pred1_points, gt_points).item()
        loss_l2_pred1 = chamfer_l2(pred1_points, gt_points).item()
        loss_l1_flip = chamfer_l1(predflip_points, gt_points).item()
        loss_l2_flip = chamfer_l2(predflip_points, gt_points).item()

        # Record end time for Chamfer Distance computation
        end_chamfer_time = time.time()
        chamfer_time = end_chamfer_time - start_chamfer_time

        # Store results in session state
        st.session_state.loss_l1_pred1 = loss_l1_pred1
        st.session_state.loss_l2_pred1 = loss_l2_pred1
        st.session_state.loss_l1_flip = loss_l1_flip
        st.session_state.loss_l2_flip = loss_l2_flip
        st.session_state.chamfer_time = chamfer_time
        
        # Show results
        st.success("Chamfer Distance computed ✅")
        st.write(f"**Chamfer L1 (no flip)**: {loss_l1_pred1:.6f}")
        st.write(f"**Chamfer L2 (no flip)**: {loss_l2_pred1:.6f}")
        st.write(f"**Chamfer L1 (flip)**: {loss_l1_flip:.6f}")
        st.write(f"**Chamfer L2 (flip)**: {loss_l2_flip:.6f}")
        st.write(f"**Chamfer computation time**: {chamfer_time:.2f} seconds")

    except Exception as e:
        st.error("❌ Error computing Chamfer Distance")
        st.exception(e)



# Compute Chamfer Distance for all chunks
if st.checkbox("Compute Chamfer Distance for all chunks"):
    try:
        # Load ground truth files from the outpath_chunks
        gt_files = glob.glob(str(outpath_chunks) + "/cube*.txt")
        
        if not gt_files:
            st.warning(f"No ground truth chunk files found in {outpath_chunks}")
            st.stop()  # Arrêt de l'exécution de Streamlit ici

        # Load prediction files from the pred_path
        pred_files = glob.glob(str(pred_path) + "/cube*.xyz")
        
        if not pred_files:
            st.warning(f"No prediction files found in {pred_path}")
            st.stop()  # Arrêt de l'exécution de Streamlit ici

        # Initialize Chamfer distance modules
        chamfer_l1 = ChamferDistanceL1()
        chamfer_l2 = ChamferDistanceL2()

        # Store results
        results = []
        
        # Variables to calculate averages
        total_l1_no_flip = 0
        total_l2_no_flip = 0
        total_l1_flip = 0
        total_l2_flip = 0
        count_no_flip = 0
        count_flip = 0

        # File name for saving results
        csv_filename = f"{model_name}_{tree_name}_{chunk_params}_chamfer.csv"
        chamfer_csv_path = pred_path / csv_filename

        # Check if the results CSV already exists
        if chamfer_csv_path.exists():
            if st.button("🔄 Recompute Chamfer Distance"):
                st.warning(f"⚠️ Recalculating Chamfer Distance and overwriting {csv_filename}...")

                # Proceed with recalculation if the button is pressed
                # Loop through each ground truth file
                for gt_file in gt_files:
                    # Load the corresponding ground truth points
                    gt_points = torch.tensor(np.loadtxt(gt_file, delimiter=" "), dtype=torch.float32).unsqueeze(0).cuda()

                    # Get corresponding prediction file (with the same name)
                    pred_file = None
                    for pf in pred_files:
                        if os.path.basename(pf).startswith(os.path.basename(gt_file).split('.')[0]):
                            pred_file = pf
                            break
                    
                    if pred_file is None:
                        st.warning(f"No corresponding prediction file found for {os.path.basename(gt_file)}")
                        continue

                    # Load the predicted points
                    pred_points = torch.tensor(np.loadtxt(pred_file, delimiter=" "), dtype=torch.float32).unsqueeze(0).cuda()

                    # Initialize variables for both flip and no-flip cases
                    loss_l1_pred = chamfer_l1(pred_points, gt_points).item()
                    loss_l2_pred = chamfer_l2(pred_points, gt_points).item()

                    # Check if "flip" is in the prediction filename
                    if "flip" in os.path.basename(pred_file):
                        results.append({
                            "File": os.path.basename(pred_file),
                            "Flip": True,
                            "Chamfer_L1": loss_l1_pred,
                            "Chamfer_L2": loss_l2_pred
                        })
                        total_l1_flip += loss_l1_pred
                        total_l2_flip += loss_l2_pred
                        count_flip += 1
                    else:
                        results.append({
                            "File": os.path.basename(pred_file),
                            "Flip": False,
                            "Chamfer_L1": loss_l1_pred,
                            "Chamfer_L2": loss_l2_pred
                        })
                        total_l1_no_flip += loss_l1_pred
                        total_l2_no_flip += loss_l2_pred
                        count_no_flip += 1

                # Calculate averages
                avg_l1_no_flip = total_l1_no_flip / count_no_flip if count_no_flip > 0 else 0
                avg_l2_no_flip = total_l2_no_flip / count_no_flip if count_no_flip > 0 else 0
                avg_l1_flip = total_l1_flip / count_flip if count_flip > 0 else 0
                avg_l2_flip = total_l2_flip / count_flip if count_flip > 0 else 0

                # Create results DataFrame
                df_results = pd.DataFrame(results)

                # Save to CSV
                df_results.to_csv(chamfer_csv_path, index=False)
                st.success(f"Chamfer Distance results saved to {chamfer_csv_path}")

                # Display results
                st.write(df_results)
                st.write(f"**Average Chamfer L1 (no flip)**: {avg_l1_no_flip:.6f}")
                st.write(f"**Average Chamfer L2 (no flip)**: {avg_l2_no_flip:.6f}")
                st.write(f"**Average Chamfer L1 (flip)**: {avg_l1_flip:.6f}")
                st.write(f"**Average Chamfer L2 (flip)**: {avg_l2_flip:.6f}")

            else:
                st.info(f"📄 The Chamfer Distance results already exist in {csv_filename}")
                # Load and display the existing CSV
                df_results = pd.read_csv(chamfer_csv_path)
                st.dataframe(df_results)

                # Calculate averages
                avg_l1_no_flip = df_results[df_results["Flip"] == False]["Chamfer_L1"].mean()
                avg_l2_no_flip = df_results[df_results["Flip"] == False]["Chamfer_L2"].mean()
                avg_l1_flip = df_results[df_results["Flip"] == True]["Chamfer_L1"].mean()
                avg_l2_flip = df_results[df_results["Flip"] == True]["Chamfer_L2"].mean()

                st.write(f"**💡 Average Chamfer L1 (no flip)**: {avg_l1_no_flip:.6f}")
                st.write(f"**💡 Average Chamfer L2 (no flip)**: {avg_l2_no_flip:.6f}")
                st.write(f"**💡 Average Chamfer L1 (flip)**: {avg_l1_flip:.6f}")
                st.write(f"**💡 Average Chamfer L2 (flip)**: {avg_l2_flip:.6f}")
                st.stop()  # Stop further execution if we are showing the existing file

        else:
            # Proceed with calculation if the file does not exist
            # Loop through each ground truth file
            for gt_file in gt_files:
                # Load the corresponding ground truth points
                gt_points = torch.tensor(np.loadtxt(gt_file, delimiter=" "), dtype=torch.float32).unsqueeze(0).cuda()

                # Get corresponding prediction file (with the same name)
                pred_file = None
                for pf in pred_files:
                    if os.path.basename(pf).startswith(os.path.basename(gt_file).split('.')[0]):
                        pred_file = pf
                        break
                
                if pred_file is None:
                    st.warning(f"No corresponding prediction file found for {os.path.basename(gt_file)}")
                    continue

                # Load the predicted points
                pred_points = torch.tensor(np.loadtxt(pred_file, delimiter=" "), dtype=torch.float32).unsqueeze(0).cuda()

                # Initialize variables for both flip and no-flip cases
                loss_l1_pred = chamfer_l1(pred_points, gt_points).item()
                loss_l2_pred = chamfer_l2(pred_points, gt_points).item()

                # Check if "flip" is in the prediction filename
                if "flip" in os.path.basename(pred_file):
                    results.append({
                        "File": os.path.basename(pred_file),
                        "Flip": True,
                        "Chamfer_L1": loss_l1_pred,
                        "Chamfer_L2": loss_l2_pred
                    })
                    total_l1_flip += loss_l1_pred
                    total_l2_flip += loss_l2_pred
                    count_flip += 1
                else:
                    results.append({
                        "File": os.path.basename(pred_file),
                        "Flip": False,
                        "Chamfer_L1": loss_l1_pred,
                        "Chamfer_L2": loss_l2_pred
                    })
                    total_l1_no_flip += loss_l1_pred
                    total_l2_no_flip += loss_l2_pred
                    count_no_flip += 1

            # Calculate averages
            avg_l1_no_flip = total_l1_no_flip / count_no_flip if count_no_flip > 0 else 0
            avg_l2_no_flip = total_l2_no_flip / count_no_flip if count_no_flip > 0 else 0
            avg_l1_flip = total_l1_flip / count_flip if count_flip > 0 else 0
            avg_l2_flip = total_l2_flip / count_flip if count_flip > 0 else 0

            # Create results DataFrame
            df_results = pd.DataFrame(results)

            # Save to CSV
            df_results.to_csv(chamfer_csv_path, index=False)
            st.success(f"Chamfer Distance results saved to {chamfer_csv_path}")

            # Display results
            st.write(df_results)
            st.write(f"**Average Chamfer L1 (no flip)**: {avg_l1_no_flip:.6f}")
            st.write(f"**Average Chamfer L2 (no flip)**: {avg_l2_no_flip:.6f}")
            st.write(f"**Average Chamfer L1 (flip)**: {avg_l1_flip:.6f}")
            st.write(f"**Average Chamfer L2 (flip)**: {avg_l2_flip:.6f}")

    except Exception as e:
        st.error("❌ Error computing Chamfer Distance")
        st.exception(e)



#### Summary table ####
st.header("Summary")

# Button to Show the Summary Table
if st.button("Show Summary Table"):
    summary_path = "summary_tests.csv"
    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
                                                                         
        # Rename columns just for visualization (not saving)
        df_summary_display = df_summary.rename(columns={
            "model_name": "Model",
            "tree_name": "Tree",
            "chunk_params": "Chunk Params",
            "nb_chunks": "Nb. of Chunks",
            "cut_time": "Cut Time (s)",
            "pts_before": "Points Before Cut",
            "pts_after_cut": "Points After Cut",
            "pts_after_inf": "Points After Inference",
            "inference_time": "Inference Time (s)",
            "chamfer_l1": "Chamfer L1",
            "chamfer_l2": "Chamfer L2",
            "chamfer_l1_with_flips": "Chamfer L1 (Flips)",
            "chamfer_l2_with_flips": "Chamfer L2 (Flips)",
            "chamfer_time": "Chamfer Time (s)",
        })
        st.dataframe(df_summary.style.format(precision=4), use_container_width=True)
        st.success("✅ Summary table displayed.")
    else:
        st.warning("⚠️ Summary file not found.")

# Button to Update the Summary Table
if st.button("Update Summary Table"):
    # Count the number of chunks (excluding flipped files)
    chunk_files = glob.glob(str(outpath_chunks / "cube*.txt"))
    chunk_files = [f for f in chunk_files if "flip" not in f]
    nb_chunks = len(chunk_files)

    pts_after_inference = load_xyz(pred_path / "completion.xyz")
    if pts_after_inference is not None:
        pts_after_inf = int(pts_after_inference.shape[0])
    else:
        st.error("❌ Failed to load the predicted point cloud for summary.")
        st.stop()

    # Load the values from session state
    cut_time = st.session_state.get('cut_time', None)
    pts_before = st.session_state.get('pts_before_cut', None)
    pts_after_cut = st.session_state.get('pts_after_cut', None)
    inference_time = st.session_state.get('inference_time', None)
    loss_l1_pred1 = st.session_state.get('loss_l1_pred1', None)
    loss_l2_pred1 = st.session_state.get('loss_l2_pred1', None)
    loss_l1_flip = st.session_state.get('loss_l1_flip', None)
    loss_l2_flip = st.session_state.get('loss_l2_flip', None)
    chamfer_time = st.session_state.get('chamfer_time', None)
    
    # Check if all required values are available
    if cut_time is None or pts_before is None or pts_after_cut is None or inference_time is None:
        st.error("❌ Missing values for summary table.")
        st.stop()
    if loss_l1_pred1 is None or loss_l2_pred1 is None or loss_l1_flip is None or loss_l2_flip is None:
        st.error("❌ Missing Chamfer Distance values for summary table.")
        st.stop()

    # Update the summary table
    update_summary_table(
        model_name=model_name,
        tree_name=tree_name,
        chunk_params=chunk_params,
        nb_chunks=nb_chunks,
        cut_time=cut_time,
        pts_before=pts_before,
        pts_after_cut=pts_after_cut,
        pts_after_inf=pts_after_inf,
        inference_time=inference_time,
        chamfer_l1=loss_l1_pred1,
        chamfer_l2=loss_l2_pred1,
        chamfer_l1_with_flips=loss_l1_flip,
        chamfer_l2_with_flips=loss_l2_flip,
        chamfer_time=chamfer_time,
    )
    st.success("✅ Summary table updated.")


# Chamfer Comparison Plot

# Function to extract number from chunk_param (e.g., 'scale_50' -> 50)
def extract_chunk_number(chunk_label):
    match = re.search(r'\d+', str(chunk_label))
    return int(match.group()) if match else -1

# Helper to darken a color
def darken_color(color, amount=1.0):
    return tuple(max(min(c * amount, 1.0), 0) for c in color)

# Button to Show Chamfer Comparison Plot
if st.button("Show Chamfer Comparison Plot"):
    summary_path = "summary_tests.csv"
    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)

        # Check required columns
        required_cols = {'model', 'tree', 'chunk_params', 'chamfer_l1', 'chamfer_l2'}
        if not required_cols.issubset(df_summary.columns):
            st.error("❌ Missing required columns in summary_tests.csv.")
        else:
            # Extract and sort chunk_params numerically
            df_summary['chunk_number'] = df_summary['chunk_params'].apply(extract_chunk_number)
            df_summary.sort_values('chunk_number', inplace=True)

            # Unique models and vibrant palette
            models = df_summary['model'].unique()
            model_palette = sns.color_palette("tab10", n_colors=len(models))
            model_colors = {model: model_palette[i] for i, model in enumerate(models)}

            # Tree index mapping
            unique_trees = df_summary['tree'].unique()
            tree_mapping = {tree: idx for idx, tree in enumerate(unique_trees)}

            # Set up subplots
            fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

            # Plot each model/tree combo
            for (model, tree), group in df_summary.groupby(['model', 'tree']):
                base_color = model_colors[model]
                tree_index = tree_mapping[tree] % 10
                shade = darken_color(base_color, amount=1 + 0.15 * tree_index)

                label = f"{model} / {tree}"
                axes[0].plot(group['chunk_number'], group['chamfer_l1'], marker='o', label=label, color=shade)
                axes[1].plot(group['chunk_number'], group['chamfer_l2'], marker='x', label=label, color=shade)

            # Customize plots
            axes[0].set_title('Chamfer L1 vs. Number of Chunks')
            axes[0].set_ylabel('Chamfer L1')
            axes[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
            axes[0].grid(True)

            axes[1].set_title('Chamfer L2 vs. Number of Chunks')
            axes[1].set_xlabel('Number of Chunks')
            axes[1].set_ylabel('Chamfer L2')
            axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
            axes[1].grid(True)

            st.pyplot(fig)
    else:
        st.warning("⚠️ Summary file not found.")