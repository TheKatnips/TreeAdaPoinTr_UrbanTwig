# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:25:14 2023

"""
import numpy as np
import open3d as o3d
import random
#import laspy 
#import pptk
import os
# import glob
# import re
import json
import shutil

def bounding_box(points, min_x=-np.inf, max_x=np.inf, min_y=-np.inf,
                        max_y=np.inf, min_z=-np.inf, max_z=np.inf):
    """ Compute a bounding_box filter on the given points
    from: https://stackoverflow.com/questions/42352622/finding-points-within-a-bounding-box-with-numpy    
    Parameters
    ----------                        
    points: (n,3) array
        The array containing all the points's coordinates. Expected format:
            array([
                [x1,y1,z1],
                ...,
                [xn,yn,zn]])

    min_i, max_i: float
        The bounding box limits for each coordinate. If some limits are missing,
        the default values are -infinite for the min_i and infinite for the max_i.

    Returns
    -------
    bb_filter : boolean array
        The boolean mask indicating wherever a point should be kept or not.
        The size of the boolean mask will be the same as the number of given points.

    """

    bound_x = np.logical_and(points[:, 0] > min_x, points[:, 0] < max_x)
    bound_y = np.logical_and(points[:, 1] > min_y, points[:, 1] < max_y)
    bound_z = np.logical_and(points[:, 2] > min_z, points[:, 2] < max_z)

    bb_filter = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    return bb_filter





def mksamples_sim(fulltree_path, sim_path, outpath, start_count=0, stop_count=100, boxsize=0.5):

    n_points = 8192
     
    file = os.listdir(fulltree_path)
    
    # make data output directory 
    if not os.path.exists(outpath+"/train/"):
        os.makedirs(outpath+"/train/")
    outpath = outpath+"/train/"
        
    for item in file:

         treename = os.path.splitext(os.path.basename(item))[0]
         
         # load point cloud and store as numpy array
         if item.endswith('.npy'):
             pc = np.load(fulltree_path+item)
             
         if item.endswith('.xyz'):
             pc = np.loadtxt(fulltree_path+item, skiprows=1, delimiter=",")
             
         if item.endswith('.txt'):
            pc = np.loadtxt(fulltree_path+item, skiprows=1, delimiter=",") 
             
         if item.endswith('.ply'):
             ply_cloud = o3d.io.read_point_cloud(fulltree_path+item) 
             pc = np.asarray(ply_cloud.points)

        

        # generate point cloud samples of bb size s for random locations in tree
         count = start_count
            # number of samples:
         n = stop_count
            
         while count in range(n):
                # select a random point (row)
                number_of_rows = pc.shape[0]
                random_indices = np.random.choice(number_of_rows, size=1, replace=False)
                # display random rows
                #print("\nRandom row:")
                row = pc[random_indices, :]
                #print(row)
                
                
                # bounding box selection around point
                # box size?
                s = boxsize
                
                min_x = row[0,0]-s
                max_x = row[0,0]+s
                min_y = row[0,1]-s
                max_y = row[0,1]+s
                min_z = row[0,2]-s
                max_z = row[0,2]+s
                
                pc_bb_filter = bounding_box(pc, min_x, max_x, min_y, max_y, min_z, max_z)
                pc_sample = pc[pc_bb_filter,:]
    
                
                if len(pc_sample)>n_points:
                
                    # # translate sample point cloud back into original
                    # sample[:, 0] = sample[:, 0]-100
                    # sample[:, 1] = sample[:, 1]-100
                
                    # downsample complete point cloud to n_points points
                    number_of_rows = pc_sample.shape[0]

                    # random sampling :
                    random_indices = np.random.choice(number_of_rows, size=n_points, replace=False)
                    pc_sample = pc_sample[random_indices,:]
                    
                    # make data output directory for each tree
                    if not os.path.exists(outpath+"complete/"+treename):
                       os.makedirs(outpath+"complete/"+treename)
                    
                    
                    # Directory containing files (including subdirectories)
                    directory = sim_path+"/"+treename+".obj"
                    
                    # List to store the loaded arrays
                    arrays = {}
                    
                    # Read the .xyz files and store the arrays in a dictionary with original file names as keys
                    # Traverse the directory tree recursively
                    for root, _, files in os.walk(directory):
                        for file in files:
                            if file.endswith(".xyz"):
                                file_path = os.path.join(root, file)
                                ppc = np.loadtxt(file_path, delimiter=' ')
                                ppc_bb_filter = bounding_box(ppc, min_x, max_x, min_y, max_y, min_z, max_z)
                                ppc_sample = ppc[ppc_bb_filter,:]                
                                arrays[file] = ppc_sample                     
    
                    
                        # Generate combinations of arrays
                        combinations = set()
                        for file1, array1 in arrays.items():
                            for file2, array2 in arrays.items():
                                if file1 != file2:
                                    combination_key = frozenset([file1, file2])
                                    if combination_key not in combinations:
                                        combined = np.concatenate((array1, array2), axis=0)
                                        combinations.add(combination_key)
                                        
                                        # only save partial cloud if enough points in partial cloud sample (more than 1/3 of complete cloud)
                                        if n_points/3 < len(combined):
                                            
                                            # make data output directory for each tree
                                            outdir = outpath+"partial/"+treename+'/'+treename+'_'+str(count)+'_size'+str(s*2)
                                            if not os.path.exists(outdir):
                                               os.makedirs(outdir)
    
                                            # Save as .npy file
                                            npy_file = treename+'_'+str(count)+'_size'+str(s*2)+'_'+f"{file1[:-4]}_and_{file2[:-4]}.npy"
                                            np.save(outdir+"/"+npy_file, combined[:,0:3])
                                            #print(f"Saved {npy_file}")
            
                                            # # Save as .xyz file
                                            # xyz_file = treename+'_'+str(count)+'_size'+str(s*2)+'_'+f"{file1[:-4]}_combination_{file2[:-4]}.xyz"
                                            # np.savetxt(outdir+"/"+xyz_file, combined, fmt="%0.6f", delimiter="\t")
                                            # print(f"Saved {xyz_file}")
    
                        # also save partial clouds of individual legs
                        for file1, array1 in arrays.items():
                            # only save partial cloud if enough points in partial cloud sample (more than 1/3 of complete cloud)
                            if n_points/3 < len(array1):
                                
                                # make data output directory for each tree
                                outdir = outpath+"partial/"+treename+'/'+treename+'_'+str(count)+'_size'+str(s*2)
                                if not os.path.exists(outdir):
                                    os.makedirs(outdir)
    
                                # Save as .npy file
                                npy_file = treename+'_'+str(count)+'_size'+str(s*2)+'_'+f"{file1[:-4]}.npy"
                                np.save(outdir+"/"+npy_file, array1[:,0:3])
                                #print(f"Saved {npy_file}")
    
                        
                        # downsample number of points if over 2/3 of complete
                        # if len(ppc_sample) > (n_points/3)*2:
                        #     random_indices = np.random.choice(len(ppc_sample), size=round((n_points/3)*2), replace=False)
                        #     ppc_sample = ppc_sample[random_indices,:]
                        
                    
                    
                    if os.path.exists(outpath+"partial/"+treename+'/'+treename+'_'+str(count)+'_size'+str(s*2)):
                        # save complete point cloud sample
                        outname = outpath+"complete/"+treename+'/'+treename+'_'+str(count)+'_size'+str(s*2)
                        np.save(outname, pc_sample[:,0:3])
                        
                        # only count up if sample gets saved
                        count += 1






def traintest_json(complete_dir_train, output_path, dataset_name):

    dict_lst = [] 
       
    banrs = os.listdir(complete_dir_train) # taxonomy id
    for banr in banrs: 
        filenames = [os.path.splitext(f)[0] for f in os.listdir(complete_dir_train+"/"+banr)]
        #filenames_test = [os.path.splitext(f)[0] for f in os.listdir(complete_dir_test+"/"+banr)]
        test = random.sample(filenames, int(len(filenames) * 0.2))
        train = np.setdiff1d(filenames,test).tolist()
        
        dictionary = {
            "taxonomy_id": str(banr),
            "taxonomy_name": str(banr),
            "test": test,
            "train": train,
            "val": []
            }
        
        dict_lst.append(dictionary)
        
        # move the test files to the test folder structure
        for file in test:
     
            # Create the full paths for source and destination
            source_path = complete_dir_train + "/"+ banr + "/" + file +".npy"
            destination_directory_complete = output_path + "test/complete/" + banr + "/"
            # Create the destination directory if it doesn't exist
            if not os.path.exists(destination_directory_complete):
                os.makedirs(destination_directory_complete)
            # Construct the destination path
            destination_path = destination_directory_complete + file +".npy"
            # Move the file to the destination
            shutil.move(source_path, destination_path)
        
            # Create the full paths for source and destination
            source_path = output_path + "train/partial/"+ banr + "/" + file
            destination_directory_partial = output_path + "test/partial/" + banr +"/"
            # Create the destination directory if it doesn't exist
            if not os.path.exists(destination_directory_partial):
                os.makedirs(destination_directory_partial)
            # Construct the destination path
            destination_path = destination_directory_partial + file 
            # Move the file to the destination
            shutil.move(source_path, destination_path)
        

    json_object = json.dumps(dict_lst, indent=4)
    with open(output_path+dataset_name+".json", "w") as outfile:
        outfile.write(json_object)









# write json for train/test set definitions

# import json
# import os

# output_path = "F:/definitive_data_PoinTr/treeSim_real/10thousand/"
# partial_dir_train = "F:/definitive_data_PoinTr/treeSim_real/real/"+"train/partial"
# partial_dir_test = "F:/definitive_data_PoinTr/treeSim_real/real/"+"test/partial"
# complete_dir_train = output_path+"train/complete"
# complete_dir_test = output_path+"test/complete"

# dict_lst = [] 
   
# banrs = os.listdir(complete_dir_train) # taxonomy id
# for banr in banrs:
#     fn_train_complete = [os.path.splitext(f)[0] for f in os.listdir(complete_dir_train+"/"+banr)]
#     fn_train_partial = os.listdir(partial_dir_train+"/"+banr)
#     fn_train = list(set([item for item in fn_train_complete if item in fn_train_partial]))
#     #fn_train = [ele for ele in fn_train for i in range(3)]

#     fn_test_complete = [os.path.splitext(f)[0] for f in os.listdir(complete_dir_test+"/"+banr)]
#     fn_test_partial = os.listdir(partial_dir_test+"/"+banr)
#     fn_test = list(set([item for item in fn_test_complete if item in fn_test_partial]))
#     #fn_test = [ele for ele in fn_test for i in range(2)]
#     # test = random.sample(filenames, int(len(filenames) * 0.2))
#     # train = np.setdiff1d(filenames,test).tolist()
    
#     dictionary = {
#         "taxonomy_id": str(banr),
#         "taxonomy_name": str(banr),
#         "test": fn_test,
#         "train": fn_train,
#         "val": []
#         }
    
#     dict_lst.append(dictionary)

# json_object = json.dumps(dict_lst, indent=4)
# with open(output_path+"treeSim_real10.json", "w") as outfile:
#     outfile.write(json_object)

