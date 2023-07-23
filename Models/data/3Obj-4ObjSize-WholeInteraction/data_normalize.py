import glob 
import os
import pickle
import numpy as np
import h5py
for hdf5_path in glob.glob(os.path.join('./', '*.hdf5')):
    hf = h5py.File(hdf5_path, 'r')
    gt_positions = hf['positions'][()]
    gt_modes = hf['modes'][()]
    gt_edges = hf['edges'][()]
    gt_ids = hf['ids'][()]
    print(gt_positions.shape, gt_modes.shape, gt_edges.shape, gt_ids.shape)

    gt_num_b, gt_num_t, gt_num_o = gt_ids.shape
    gt_all_o = {}
    for gt_i in range(gt_num_b):
        for gt_j in range(gt_num_t):
            for gt_k in range(gt_num_o):
                # if gt_in_camera[i,j,k] == 1:
                gt_id_o = str(gt_i)+'_'+str(int(gt_ids[gt_i,gt_j,gt_k]))
                if gt_id_o not in gt_all_o:
                    gt_all_o[gt_id_o] = []
                    gt_all_o[gt_id_o].append((gt_positions[gt_i,gt_j,gt_k][0], gt_positions[gt_i,gt_j,gt_k][1], gt_modes[gt_i,gt_j,gt_k]))
                else:
                    gt_all_o[gt_id_o].append((gt_positions[gt_i,gt_j,gt_k][0], gt_positions[gt_i,gt_j,gt_k][1], gt_modes[gt_i,gt_j,gt_k])) # (x,y,mode)
    gt_all_o['edges'] = gt_edges
    print(len(gt_all_o))

    # store gt traj
    traj_file_gt = hdf5_path.replace('.hdf5', '_gt.pkl')
    with open(traj_file_gt, 'wb') as tf:
        pickle.dump(gt_all_o, tf)

# Output
# (191, 100, 3, 2) (191, 100, 3) (191, 100, 3) (191, 100, 3)
# 574
# (204, 100, 3, 2) (204, 100, 3) (204, 100, 3) (204, 100, 3)
# 613
# (4928, 100, 3, 2) (4928, 100, 3) (4928, 100, 3) (4928, 100, 3)
# 14785

for pkl_path in glob.glob(os.path.join('./', '*.pkl')):
    with open(pkl_path, "rb") as tf:
        print(pkl_path)
        data_dict = pickle.load(tf)
        print(len(data_dict))

        data_dict_normalized = {}
        data_x = []
        data_y = []
        for key in data_dict:
            if key != 'edges':
                each_sample = np.asarray(data_dict[key], dtype=np.float32)
                data_x = data_x + list(each_sample[:,0])
                data_y = data_y + list(each_sample[:,1])
        mean_x = np.asarray(data_x).mean()
        std_x = np.asarray(data_x).std()
        mean_y = np.asarray(data_y).mean()
        std_y = np.asarray(data_y).std()
        for key in data_dict:
            each_sample = np.asarray(data_dict[key], dtype=np.float32)
            if key != 'edges':
                each_sample[:,0] = (each_sample[:,0] - mean_x) / std_x
                each_sample[:,1] = (each_sample[:,1] - mean_y) / std_y
            else:
                print('edges')
            data_dict_normalized[key] = each_sample
                
        pkl_path_normalized = pkl_path.replace('.pkl', '_normalized.pkl')
        with open(pkl_path_normalized, 'wb') as tf:
            pickle.dump(data_dict_normalized, tf)

        with open(pkl_path_normalized, "rb") as tf:
            data_dict = pickle.load(tf)
            print(len(data_dict))