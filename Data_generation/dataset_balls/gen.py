"""
This script is adapted from the RTRBM code by Ilya Sutskever from
http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.tar
"""
import numpy as np
import os
import h5py
from torch import randint
from tqdm import tqdm, trange
import random
import cv2
from config import default, config_list
from PIL import ImageColor
import moviepy.editor as mpy
import argparse
from glob import glob
import imageio
from joblib import Parallel, delayed
import multiprocessing
from datetime import datetime
from copy import deepcopy
import os.path as osp

import pdb

parent_dir = osp.abspath(osp.dirname(__file__))

SHAPE_LISTS = ['circle16', 'star16', 'cross16', 'cross_rotate16', 'diamond16']

SHAPES = [imageio.imread(osp.join(parent_dir, 'shapes', '{name}.png'.format(name=name))) for name in SHAPE_LISTS]

def is_overlapping(loc1, loc2, s1, s2, shape1, shape2, layer1, layer2, present1, present2):
    x1, y1 = loc1
    x2, y2 = loc2

    if(present1 ==0 or present2==0):
        return False
    if(layer1 != layer2):
        return False
    if (shape1 == 0 and shape2 == 0):
        # Circle
        if (((x1 - x2) ** 2 + (y1 - y2) ** 2)  < (s1 + s2) ** 2):
            return True
        return False
    else:
        if (x1 + s1 < x2 - s2 or x2 + s2 < x1 - s1 or y1 + s1 < y2 - s2 or y2 + s2 < y1 - s1):
            return False
        return True
    
def update_speed(loc1, loc2, v1, v2, mass1, mass2):
    w = loc1 - loc2  # w is the vector connecting the center of objects
    w = w / np.linalg.norm(w)

    v1_mag = w.dot(v1) # magnitude of speed which is along the center line for obj1. this is a scalar number
    v2_mag = w.dot(v2)# magnitude of speed which is along the center line for obj1. this is a scalar number
    # in fact if the masses are equal it's adding the magnitude of speeds along directions and just adding the new added speeds to the preious one.
    # thiese equations are obtained from two equations: 1. v1old+v2old=v1new+v2new 2. m1*(v1new-v1old)=m2*(v2new-v2old)

    v1 =v1+ w * (2*mass2/(mass1+mass2))*(v2_mag - v1_mag)
    v2 =v2+ w * (2*mass1/(mass1+mass2))*(v1_mag - v2_mag)

    return v1, v2
    

def make_sequence(
        seqlen, # 'seqlen': 100
        canvas_size, 
        camera_size,
        maxnum,
        num_objs,
        obj_shapes,
        obj_sizes,
        obj_layers,
        obj_masses,
        obj_colors,
        obj_modes,
        speed,
        restricted,
        first_nonoverlap,
        updates_per_second,
        max_attempts,
        interaction,
        part_interaction,
        fps,
        rng
):
    """
    Make a single sequence
    Args:
        seqlen: length of the sequence
        canvas_size: (h, w)
        camera_size: (h, w)
        maxnum: maximum number of objects. This must be larger than the
            potential number of objects
        num_objs: (type, value). value is (low, high)
        obj_shapes: (type, value). value is a list of shape names
        obj_sizes: (type, value).
        obj_layers: (type, value)
        obj_masses: (type, value)
        obj_colors: (type, value). value is a list of html color names
        speed: number of pixels to move per frame
        restricted: for the first frame, whether all objects should be within camera
        first_nonoverlap: for the first frame, whether there should be no overlap
        updates_per_second: controls the how fine-grained is the movement
        max_attempts:
        interaction: bool

    Returns:
        locations, velocities, masses, shapes, sizes, colors, present, layers,
        all (T, MAX, N)

    """
    T = seqlen
    MAX = maxnum
    can_h, can_w = canvas_size
    cam_h, cam_w = camera_size
    
    # Float
    masses = np.zeros((T, MAX))
    # Strings
    shapes = np.zeros((T, MAX))
    # Integer
    layers = np.zeros((T, MAX))
    # Float
    sizes = np.zeros((T, MAX))
    # In RGB order
    colors = np.zeros((T, MAX, 3))
    modes = np.zeros((T, MAX))
    present = np.zeros((T, MAX))
    # (x, y), in image coordinate
    locations = np.zeros((T, MAX, 2))
    velocities = np.zeros((T, MAX, 2))
    
    # Actual number of objects for this sequence
    if num_objs[0] == 'random':
        if num_objs[1][0] == num_objs[1][1]:
            O = num_objs[1][0]
        else:
            O = rng.randint(low=num_objs[1][0], high=num_objs[1][1])
    else:
        O = num_objs[1]
    
    edges = np.zeros((T, O*(O-1)//2))


    def real_to_virtual(st_real, ed_real, st_virtual, ed_virtual, x_real, y_real):
        x_virtual = (x_real - st_real) / (ed_real - st_real) * (ed_virtual - st_virtual) + st_virtual
        y_virtual = (y_real - st_real) / (ed_real - st_real) * (ed_virtual - st_virtual) + st_virtual
        return x_virtual, y_virtual
    
    def virtual_to_real(st_real, ed_real, st_virtual, ed_virtual, x_virtual, y_virtual):
        x_real = (x_virtual - st_virtual) / (ed_virtual - st_virtual) * (ed_real - st_real) + st_real
        y_real = (y_virtual - st_virtual) / (ed_virtual - st_virtual) * (ed_real - st_real) + st_real
        return x_real, y_real   
 
    def assign_attributes(attr, type, O, attrname=None):
        """
        :param attribute: (T, MAX)
        :param type: (type, value). If type is not random, then value should
                be a list of length O. Otherwise, value is a list of any size.
        :param O: actual number of objects
        :return:
                (T, O)
        """
        if type[0] == 'random':
            # (O,)
            values = rng.choice(type[1], O)
        elif type[0] == 'random_noreplace':
            # (O,)
            values = rng.choice(type[1], O, replace=False) # no repeated elements
        elif type[0] == 'custom':
            # TODO: this is not elegant. change it
            # Custom color for two layer, (2,)
            two_colors = rng.choice(type[1], 2, replace=False)
            values = np.repeat(two_colors, O // 2)
        else:
            values = type[1]
        if attrname == 'color':
            values = np.array([ImageColor.getrgb(c) for c in values])
            values = values / 255.
        if attrname == 'shape':
            values = np.array([SHAPE_LISTS.index(s) for s in values])
            assert np.all(values != -1)

        attr[:, :O] = np.repeat(np.array(values)[None, :], T, axis=0)
        
    def init_velocities(velocities, where_actions):
        # # Initialize speed, (O, 2)
        # magnitudes = (rng.randn(O, 2))
        # Simplify the motion with two types of motion only
        magnitudes = np.array([[0,1],[1,0]])
        where_actions[:, 1] = int(1)
        # print(magnitudes)

        # print(magnitudes)
        # Oormalize, and get spped
        magnitudes = magnitudes / np.linalg.norm(magnitudes, axis=-1, keepdims=True)
        # print(magnitudes)
        magnitudes = magnitudes * speed
        # print(magnitudes)
        # Initial velocity
        # import ipdb; ipdb.set_trace()
        velocities[0, :O, :] = magnitudes
    
    def init_locations(locations, modes):
        """
        :param canvas_size: (h, w)
        :param camera_size: (h, w)
        :param first_nonoverlap: bool
        :param restricted: first frame objects must be within the camera
        """
        i = 0
        attempts = 0
        while i < O:
            if (restricted):
                if modes[0, i] == 1:
                    st_virtual = 0
                    ed_virtual = 4
                    st_real = (can_w - cam_w) / 2 + sizes[0, i]
                    ed_real = (can_w - cam_w) / 2 + cam_w - sizes[0, i]
                    low_virtual = 1
                    high_virtual = 3
                    low_real, high_real = virtual_to_real(st_real, ed_real, st_virtual, ed_virtual, low_virtual, high_virtual)
                    locations[0, i, :] = rng.uniform(low=low_real, high=high_real, size=2)
                else:
                    locations[0, i, :] = rng.uniform(low=(can_w - cam_w) / 2 + sizes[0, i], high=(can_w - cam_w) / 2 + cam_w - sizes[0, i], size=2)
            else:
                locations[0, i, :] = rng.uniform(low=sizes[0, i], high=(can_w - sizes[0, i]), size=2)
            good_config = True
            for j in range(i):
                if (is_overlapping(locations[0, i, :], locations[0, j, :], sizes[0, i], sizes[0, j], shapes[0, i],
                                  shapes[0, j], layers[0, i], layers[0, i], present[0, i], present[0, j])):
                    if first_nonoverlap:
                        good_config = False
                        break
            if (not good_config):
                attempts += 1
                if attempts > max_attempts:
                    raise ValueError('Too many attempts when placing objects')
            else:
                i += 1
                attempts = 0
            
    def init_calucate_v(location_real, modes):
        # locations.shape (number_of_objects, 2)
        # modes.shape (number_of_objects)
        v = np.zeros(location_real.shape)
        for i in range(len(location_real)):
            if modes[i] == 0: # bouncing ball
                magnitudes = np.array([0, 1])
                magnitudes = magnitudes / np.linalg.norm(magnitudes, axis=-1, keepdims=True)
                magnitudes = magnitudes * speed
                # print(magnitudes)
                v[i] = magnitudes
            elif modes[i] == 1: # Lotka-Volterra ODE
                st_virtual = 0
                ed_virtual = 4
                st_real = (can_w - cam_w) / 2 + sizes[0, i]
                ed_real = (can_w - cam_w) / 2 + cam_w - sizes[0, i]
                x_real = location_real[i][0]
                y_real = location_real[i][1]
                x_virtual, y_virtual = real_to_virtual(st_real, ed_real, st_virtual, ed_virtual, x_real, y_real)
                v[i][0] = x_virtual - x_virtual * y_virtual
                v[i][1] = -y_virtual + x_virtual * y_virtual
            elif modes[i] == 2: # Spiral ODE
                st_virtual = -2
                ed_virtual = 2
                st_real = (can_w - cam_w) / 2 + sizes[0, i]
                ed_real = (can_w - cam_w) / 2 + cam_w - sizes[0, i]
                x_real = location_real[i][0]
                y_real = location_real[i][1]
                x_virtual, y_virtual = real_to_virtual(st_real, ed_real, st_virtual, ed_virtual, x_real, y_real)
                # print(st_real, ed_real, st_virtual, ed_virtual, x_real, y_real)
                # print(x_virtual, y_virtual)
                # x_real, y_real = virtual_to_real(st_real, ed_real, st_virtual, ed_virtual, x_virtual, y_virtual)
                # print(x_real, y_real)
                v[i][0] = -0.1 * np.power(x_virtual, 3) + 2 * np.power(y_virtual, 3)
                v[i][1] = -2 * np.power(x_virtual, 3) - 0.1 * np.power(y_virtual, 3)
                # print(v[i])
        return v

    def calucate_v_update(location_real, v_virtual, modes, edges, whole_frame, interaction, eps):

        location_real_attempt = np.zeros(location_real.shape)
        location_real_update = np.zeros(location_real.shape)

        for i in range(len(location_real)):
            if modes[whole_frame, i] == 0: # bouncing ball
                location_real_update[i] = location_real[i] + eps * v_virtual[i]
                indices = (location_real_update[i] < (can_w - cam_w) / 2 + sizes[0, i])
                v_virtual[i][indices] = np.abs(v_virtual[i][indices])
                indices = (location_real_update[i] > (can_w - cam_w) / 2 + cam_w - sizes[0, i])
                v_virtual[i][indices] = -1 * np.abs(v_virtual[i][indices])
                location_real_attempt[i] = location_real[i] + eps * v_virtual[i]

            elif modes[whole_frame, i] == 1: # Lotka-Volterra ODE
                st_virtual = 0
                ed_virtual = 4
                st_real = (can_w - cam_w) / 2 + sizes[0, i]
                ed_real = (can_w - cam_w) / 2 + cam_w - sizes[0, i]
                x_real = location_real[i][0]
                y_real = location_real[i][1]
                x_virtual, y_virtual = real_to_virtual(st_real, ed_real, st_virtual, ed_virtual, x_real, y_real)
                
                x_virtual_update = x_virtual + eps * v_virtual[i][0]
                y_virtual_update = y_virtual + eps * v_virtual[i][1]
                location_real_attempt[i][0], location_real_attempt[i][1] = \
                    virtual_to_real(st_real, ed_real, st_virtual, ed_virtual, x_virtual_update, y_virtual_update)
                
                # v_virtual[i][0] = x_virtual_update - x_virtual_update * y_virtual_update
                # v_virtual[i][1] = -y_virtual_update + x_virtual_update * y_virtual_update
            elif modes[whole_frame, i] == 2: # Spiral ODE
                st_virtual = -2
                ed_virtual = 2
                st_real = (can_w - cam_w) / 2 + sizes[0, i]
                ed_real = (can_w - cam_w) / 2 + cam_w - sizes[0, i]
                x_real = location_real[i][0]
                y_real = location_real[i][1]
                x_virtual, y_virtual = real_to_virtual(st_real, ed_real, st_virtual, ed_virtual, x_real, y_real)
                # print(x_real, y_real, x_virtual, y_virtual)
                x_virtual_update = x_virtual + eps * v_virtual[i][0]
                y_virtual_update = y_virtual + eps * v_virtual[i][1]
                location_real_attempt[i][0], location_real_attempt[i][1] = \
                    virtual_to_real(st_real, ed_real, st_virtual, ed_virtual, x_virtual_update, y_virtual_update)
                # print(location_real_update[i][0], location_real_update[i][1], x_virtual_update, y_virtual_update)
                # v_virtual[i][0] = -0.1 * np.power(x_virtual_update, 3) + 2 * np.power(y_virtual_update, 3)
                # v_virtual[i][1] = -2 * np.power(x_virtual_update, 3) - 0.1 * np.power(y_virtual_update, 3)
        
        exit_overlap = False
        if interaction:
            overlap_ind = []
            overlap_dit = {}
            edge_idx = -1
            for i in range(len(location_real)):
                for j in range(i): # 1,0; 2,0; 2,1; 3,0; 3,1; 3,2;
                    edge_idx = edge_idx + 1
                    if (is_overlapping(location_real_attempt[i, :], location_real_attempt[j, :], sizes[whole_frame, i], sizes[whole_frame, j], shapes[whole_frame, i], shapes[whole_frame, j],
                                        layers[whole_frame, i], layers[whole_frame, j], present[whole_frame, i], present[whole_frame, j])):
                        overlap_ind.append((i,j))
                        exit_overlap = True
                        edges[whole_frame, edge_idx] = 1
            for ind in overlap_ind:
                if str(ind[0]) not in overlap_dit:
                    overlap_dit[str(ind[0])] = []
                overlap_dit[str(ind[0])].append(ind[1])
                if str(ind[1]) not in overlap_dit:
                    overlap_dit[str(ind[1])] = []
                overlap_dit[str(ind[1])].append(ind[0])
            
            for key in overlap_dit:
                if len(overlap_dit[key]) > 1:
                    rind = random.randint(0, len(overlap_dit[key])-1)
                    rval = overlap_dit[key][rind]
                    overlap_dit[key] = []
                    overlap_dit[key].append(rval)
            overlap_dit_mode = overlap_dit.copy()
            for key in overlap_dit:
                overlap_dit_mode[key] = int(modes[whole_frame, int(overlap_dit[key][0])])
            for key in overlap_dit:
                modes[whole_frame:, int(key)] = overlap_dit_mode[key]
                if overlap_dit_mode[key] == 0:
                    v_virtual[int(key)] = v_virtual[int(overlap_dit[key][0])]
        
        for i in range(len(location_real)):
            if modes[whole_frame, i] == 0: # bouncing ball
                location_real_update[i] = location_real[i] + eps * v_virtual[i]
                indices = (location_real_update[i] < (can_w - cam_w) / 2 + sizes[0, i])
                v_virtual[i][indices] = np.abs(v_virtual[i][indices])
                indices = (location_real_update[i] > (can_w - cam_w) / 2 + cam_w - sizes[0, i])
                v_virtual[i][indices] = -1 * np.abs(v_virtual[i][indices])
                location_real_update[i] = location_real[i] + eps * v_virtual[i]

            elif modes[whole_frame, i] == 1: # Lotka-Volterra ODE
                st_virtual = 0
                ed_virtual = 4
                st_real = (can_w - cam_w) / 2 + sizes[0, i]
                ed_real = (can_w - cam_w) / 2 + cam_w - sizes[0, i]
                x_real = location_real[i][0]
                y_real = location_real[i][1]
                x_virtual, y_virtual = real_to_virtual(st_real, ed_real, st_virtual, ed_virtual, x_real, y_real)

                v_virtual[i][0] = x_virtual - x_virtual * y_virtual
                v_virtual[i][1] = -y_virtual + x_virtual * y_virtual
                
                x_virtual_update = x_virtual + eps * v_virtual[i][0]
                y_virtual_update = y_virtual + eps * v_virtual[i][1]
                location_real_update[i][0], location_real_update[i][1] = \
                    virtual_to_real(st_real, ed_real, st_virtual, ed_virtual, x_virtual_update, y_virtual_update)
                
                v_virtual[i][0] = x_virtual_update - x_virtual_update * y_virtual_update
                v_virtual[i][1] = -y_virtual_update + x_virtual_update * y_virtual_update
            elif modes[whole_frame, i] == 2: # Spiral ODE
                st_virtual = -2
                ed_virtual = 2
                st_real = (can_w - cam_w) / 2 + sizes[0, i]
                ed_real = (can_w - cam_w) / 2 + cam_w - sizes[0, i]
                x_real = location_real[i][0]
                y_real = location_real[i][1]
                x_virtual, y_virtual = real_to_virtual(st_real, ed_real, st_virtual, ed_virtual, x_real, y_real)

                v_virtual[i][0] = -0.1 * np.power(x_virtual, 3) + 2 * np.power(y_virtual, 3)
                v_virtual[i][1] = -2 * np.power(x_virtual, 3) - 0.1 * np.power(y_virtual, 3)

                # print(x_real, y_real, x_virtual, y_virtual)
                x_virtual_update = x_virtual + eps * v_virtual[i][0]
                y_virtual_update = y_virtual + eps * v_virtual[i][1]
                location_real_update[i][0], location_real_update[i][1] = \
                    virtual_to_real(st_real, ed_real, st_virtual, ed_virtual, x_virtual_update, y_virtual_update)
                # print(location_real_update[i][0], location_real_update[i][1], x_virtual_update, y_virtual_update)
                v_virtual[i][0] = -0.1 * np.power(x_virtual_update, 3) + 2 * np.power(y_virtual_update, 3)
                v_virtual[i][1] = -2 * np.power(x_virtual_update, 3) - 0.1 * np.power(y_virtual_update, 3)
        return location_real_update, v_virtual, modes, edges

        
    # Initialization
    attrs = [masses, layers, sizes, modes]
    configs = [obj_masses, obj_layers, obj_sizes, obj_modes]
    for attr, config in zip(attrs, configs):
        assign_attributes(attr, config, O)
    ### For Part Interaction ### 
    if part_interaction:
        layers[modes == 1] = 1
    ### For Part Interaction ###
    assign_attributes(colors, obj_colors, O, attrname='color')
    assign_attributes(shapes, obj_shapes, O, attrname='shape')
    
    present[:, :O] = 1
    # init_velocities(velocities, where_actions)
    init_locations(locations, modes)

    # Updates over time
    # (O, 2)
    location_real = locations[0, :O].copy()
    # # (O, 2)
    v_virtual = init_calucate_v(location_real, modes[0, :O].copy())
    velocities[0, :O] = v_virtual.copy()

    whole_frame = -1
    for t in range(0, T // fps):
        # print('t = {}'.format(t))
        eps = 1 / updates_per_second
        for ups in range(updates_per_second):
            # print('eps: {}'.format(_))
            if ups % (updates_per_second//fps) == 0:
                whole_frame = whole_frame + 1
                locations[whole_frame, :O] = location_real.copy()
                velocities[whole_frame, :O] = v_virtual.copy()
                # print(location_real, v_virtual)
            location_real, v_virtual, modes, edges = calucate_v_update(location_real, v_virtual, modes, edges, whole_frame, interaction, eps)
            

            # size_t = sizes[t, :O, np.newaxis]
            # eps = 1 / updates_per_second
            # # Bounce with border
            # # Oeed to do this first
            # # Attemptive update
            # x_ = x + eps * v
            # indices = (x_ - size_t < 0)
            # v[indices] = np.abs(v[indices])
            # indices = (x_ + size_t > np.repeat(np.array([can_w, can_h]).reshape((1, 2)), O, axis=0))
            # v[indices] = -1 * np.abs(v[indices])
            
            # # Bounce with objects
            # # Attemptive update
            # x_ = x + eps * v
            # if interaction:
            #     for i in range(O):
            #         for j in range(i):
            #             if (is_overlapping(x_[i, :], x_[j, :], sizes[t, i], sizes[t, j], shapes[t, i], shapes[t, j],
            #                               layers[t, i], layers[t, j], present[t, i], present[t, j])):
            #                 v[i,:], v[j,:] = update_speed(x_[i,:], x_[j, :], v[i,:],v[j,:], masses[t,i],masses[t,j])
            # x = x + eps * v
            
    return locations, velocities, modes, edges, masses, shapes, sizes, colors, present, layers


def camera_canvas(canvas, canvas_size, camera_size):
    can_h, can_w = canvas_size
    cam_h, cam_w = camera_size
    return canvas[int(can_h / 2 - cam_h / 2):int(can_h / 2 + cam_h / 2),
           int(can_w / 2 - cam_w / 2):int(can_w / 2 + cam_w / 2)]

def draw(
        canvas_size,
        camera_size,
        positions,
        shapes,
        layers,
        sizes,
        colors,
        present,
        round_position
):
    can_h, can_w = canvas_size
    cam_h, cam_w = camera_size
    T, O, _ = positions.shape
    in_camera = np.zeros((T, O))
    
    videos = []
    for t in range(T):
        canvas = np.zeros(canvas_size + (3,))
        for l in range(int(np.max(layers)) + 1):
            for i in range(O):
                if present[t, i] == 1 and layers[t, i] == l:
                    prev_canvas = canvas.copy()
                    if round_position:
                        x = int(np.around(positions[t, i, 1]))
                        y = int(np.around(positions[t, i, 0]))
                    else:
                        x, y = positions[t, i]
                    # size = int(sizes[t, i])
                    size = sizes[t, i]
                    # (H, W)
                    img = SHAPES[int(shapes[t, i])]
                    # img = (img / 255.)[..., None]
                    canvas = draw_shape(canvas, (x, y), size, img, colors[t, i])

                    prev_frame = camera_canvas(prev_canvas, canvas_size, camera_size)
                    curr_frame = camera_canvas(canvas, canvas_size, camera_size)
                    in_camera[t, i] = not np.array_equal(prev_frame, curr_frame)
            
        frame = camera_canvas(canvas, canvas_size, camera_size)
        frame = (frame * 255).astype(np.uint8)
        videos.append(frame)

    ids = make_id(in_camera)
    # (T, H, W, 3), (T, O), (T, O)
    return np.stack(videos, axis=0), in_camera, ids

def draw_shape(canvas, position, size, shape, color):
    """
    Draw a shape
    
    :param canvas: (H, W, 3)
    :param position: (x, y), in pixel space
    :param size: integer, in pixel
    :param shape: image, (H, W, 3)
    :return:
    """
    H, W, _ = canvas.shape
    h, w, *_ = shape.shape
    shape_corners = np.float32([[0,0], [0, h], [w, 0]])
    x_min = position[0] - size
    y_min = position[1] - size
    x_max = position[0] + size
    y_max = position[1] + size
    shape = 255 - shape
    shape_corners_transformed = np.float32([[x_min, y_min], [x_min, y_max], [x_max, y_min]])
    M = cv2.getAffineTransform(shape_corners, shape_corners_transformed)
    transformed = cv2.warpAffine(shape, M, (W, H))
    # import ipdb;ipdb.set_trace()
    transformed = (transformed /  255.)
    # if len(transformed.shape) == 2:
    #     transformed = transformed[..., None]
    canvas = canvas * (1 - transformed) + color * transformed
    return canvas
    
        
def make_id(in_camera):
    """
    
    :param in_camera: (T, O)
    :param present: (T, O)
    :return: id, (T, O)
    """
    T, O = in_camera.shape
    id_count = 1
    prev_in_camera = np.zeros(O)
    curr_ids = np.zeros(O)
    # What we want:
    #   1. If not in camera, we want id to be 0
    #   2. If not in camera in the previous step but in for this step, assign new id
    #   3. Otherwise keep it unchanged
    ids = []
    for t in range(T):
        # Not in camera
        # (O,)
        curr_in_camera = in_camera[t]
        curr_ids[curr_in_camera == False] = 0
        # Previous not in, now in
        indices_appear = (prev_in_camera == False) & (curr_in_camera == True)
        num_appear = indices_appear.sum()
        curr_ids[indices_appear] = id_count + np.arange(num_appear)

        ids.append(curr_ids.copy())

        id_count += num_appear
        prev_in_camera = curr_in_camera
        
    # (T, O)
    ids = np.stack(ids, axis=0)
    
    return ids

def make_data(
        options,
        render_options,
        numdata,
        seed
):
    
    def wrapper(i, rng):
        # All (T, O) except for colors (T, O, 3)
        locations, velocities, modes, edges, masses, shapes, sizes, colors, present, layers = make_sequence(**options, rng=rng)
        # (T, H, W, 3), (T, O), (T, O)
        video, in_camera, ids = draw(
            canvas_size=options.canvas_size,
            camera_size=options.camera_size,
            positions=locations,
            shapes=shapes,
            layers=layers,
            sizes=sizes,
            colors=colors,
            present=present,
            round_position=render_options.round_position
        )
        return video, layers, locations, velocities, modes, edges, sizes, masses, in_camera, present, ids, colors, shapes

    seeds = np.random.RandomState(seed).randint(0, 2**32-1, size=numdata)
    rngs = [np.random.RandomState(s) for s in seeds]
    # print(multiprocessing.cpu_count())
    (video_list, layer_list, location_list, velocity_list, mode_list, edge_list, size_list, mass_list, in_camera_list,
     present_list, id_list, color_list, shape_list) = \
     zip(*Parallel(n_jobs=multiprocessing.cpu_count())(delayed(wrapper)(i, rngs[i]) for i in trange(numdata)))
     
    idx = np.ones((numdata), dtype=np.bool8)
    for i in range(numdata):
        # for part object out of camera
        loc = location_list[i]
        n_loc = (loc - (cfg.options.canvas_size[0] - cfg.options.camera_size[0])//2) / cfg.options.camera_size[0]
        higher_whether = n_loc > (1.0 - np.max(cfg.options.obj_sizes[1])/cfg.options.camera_size[0])
        lower_whether = n_loc < (0 + np.max(cfg.options.obj_sizes[1])/cfg.options.camera_size[0])
        ex_whether = higher_whether.any() or lower_whether.any()
        if ex_whether:
            idx[i] = False
        # for wired mode switching cases
        if np.sum(mode_list[i]) != 900: # 100 frames * (0+1+2) = 300
            idx[i] = False
    
    # (N, T, H, W, 3)
    videos = np.stack(video_list)[idx]
    # (N, T, O)
    layers = np.stack(layer_list)[idx]
    locations = np.stack(location_list)[idx]
    velocities = np.stack(velocity_list)[idx]
    modes = np.stack(mode_list)[idx]
    edges = np.stack(edge_list)[idx]
    sizes = np.stack(size_list)[idx]
    masses = np.stack(mass_list)[idx]
    present = np.stack(present_list)[idx]
    colors = np.stack(color_list)[idx]
    shapes = np.stack(shape_list)[idx]
    in_camera = np.stack(in_camera_list)[idx]
    ids = np.stack(id_list)[idx]
    
    # TODO: this is important.
    # zero out objects not in camera
    sizes[in_camera == False] = 0
    locations[in_camera == False] = 0
    
    return videos, layers, locations, velocities, modes, edges, sizes, masses, present, ids, in_camera, colors, shapes

def dump_data(
        videos, layers, positions, velocities, modes, edges, sizes, masses, present, ids, in_camera, colors, shapes,
        path
):
    # print(positions.shape, modes.shape, edges.shape) # (N, T, O, 2)   (N, T, O)  (N, T, O*(O-1)/2)
    # print(positions[0].transpose((1,0,2))[1], modes[0].transpose((1,0))[1])
    
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.create_dataset('imgs', data=videos)
        f.create_dataset('layers', data=layers)
        f.create_dataset('positions', data=positions)
        f.create_dataset('velocities', data=velocities)
        f.create_dataset('modes', data=modes)
        f.create_dataset('edges', data=edges)
        f.create_dataset('masses', data=masses)
        f.create_dataset('sizes', data=sizes)
        f.create_dataset('ids', data=ids)
        f.create_dataset('present', data=present)
        f.create_dataset('colors', data=colors)
        f.create_dataset('shapes', data=shapes)
        f.create_dataset('in_camera', data=in_camera)
        
def make_gif(videos, path='./gif', num=10, fps=5):
    """
    Show some gif.
    
    :param videos: (N, T, H, W, 3)
    :param path: directory
    """
    os.makedirs(path, exist_ok=True)
    assert num <= videos.shape[0]
    for i in range(num):
        clip = mpy.ImageSequenceClip(list(videos[i]), fps=fps)
        clip.write_gif(os.path.join(path, f'{i}.gif'))
    
    
if __name__ == '__main__':
    
    # gen_list = ['balls_occlusion', 'balls_interaction', 'balls_two_layer', 'balls_two_layer_dense']
    gen_list = ['balls_occlusion']
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gen-list', nargs='+', type=str, default=gen_list, help='The list of configs to use')
    parser.add_argument('--output-dir', type=str, default='./data')
    # DON'T CHANGE THIS!
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    for name in args.gen_list:
        print(f'Making {name}...')
        cfg = deepcopy(default)
        cfg.merge_from_other_cfg(config_list[name])
        directory = os.path.join(args.output_dir, cfg.name.upper())
        # for s in cfg.split:
        #     split_seed = cfg.split_seeds[s]
        #     path = os.path.join(directory, f'{s}.hdf5')
        #     videos, layers, locations, velocities, modes, edges, sizes, masses, present, ids, in_camera, colors, shapes = make_data(cfg.options, cfg.render_options, cfg.split[s], seed=split_seed)
        #     print(videos.shape, locations.shape, modes.shape, edges.shape)  # (N, T, H, W, 3) (N, T, O, 2)   (N, T, O)  (N, T, O*(O-1)/2)
        #     # normalize locations to range [0,1]
        #     # print(cfg.options.canvas_size[0], cfg.options.camera_size[0], (cfg.options.canvas_size[0] - cfg.options.camera_size[0])//2)
        #     locations = (locations - (cfg.options.canvas_size[0] - cfg.options.camera_size[0])//2) / cfg.options.camera_size[0]
        #     dump_data(videos, layers, locations, velocities, modes, edges, sizes, masses, present, ids, in_camera, colors, shapes, path=path)
        # # Output ad timestep
        # with open(os.path.join(directory, 'version.txt'), 'w') as f:
        #     now = datetime.now()
        #     f.write(now.strftime("%Y-%m-%d, %H:%M:%S"))

        # Show some gif
        if cfg.gif_num > 0:
            videos, layers, locations, velocities, where_actions, edges, sizes, masses, present, ids, in_camera, colors, shapes = make_data(cfg.options, cfg.render_options, cfg.gif_num, seed=0)
            print(videos.shape)
            path = os.path.join(cfg.gif_path, cfg.name)
            # make_gif(videos, path, cfg.gif_num, cfg.gif_fps)
            make_gif(videos, path, videos.shape[0], cfg.gif_fps)