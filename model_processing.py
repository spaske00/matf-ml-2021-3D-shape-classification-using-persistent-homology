import numpy as np
import os
import pymeshlab as ml

def load_vertices(path_to_obj_file):
    vertices = []
    with open(path_to_obj_file, 'r') as f:
        for line in f.readlines():
            if line.startswith('v '):
                _, x, y, z = line.split(' ')
                vertices.append([float(x), float(y), float(z)])
    return np.array(vertices)

def load_models(models_dir):
    models = []
    for directory in os.listdir(models_dir):
        path_to_npy = os.path.join(models_dir, directory, 'model.npy')
        with open(path_to_npy, 'rb') as f:
            models.append(np.load(f))
    return models

# https://stackoverflow.com/questions/65419221/how-to-use-pymeshlab-to-reduce-vertex-number-to-a-certain-number
def compress_model(path_to_obj_file, target_vertices):
    ms = ml.MeshSet()

    ms.load_new_mesh(path_to_obj_file)
    m = ms.current_mesh()

    #Target number of vertex
    TARGET=target_vertices

    #Estimate number of faces to have 100+10000 vertex using Euler
    numFaces = 100 + 2*TARGET

    #Simplify the mesh. Only first simplification will be agressive
    while (ms.current_mesh().vertex_number() > TARGET):
        ms.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=numFaces, preservenormal=True)
        #Refine our estimation to slowly converge to TARGET vertex number
        numFaces = numFaces - (ms.current_mesh().vertex_number() - TARGET)
    

    return ms
   
def compress_models_in_dir(models_dir, target_vertices):
    count = 0
    print('Compressing models in ', models_dir)
    print('Target number of vertices ', target_vertices)
    for directory in os.listdir(models_dir):
        path_to_obj_file = os.path.join(models_dir, directory, 'model.obj')
        print('Compressing: ', path_to_obj_file)
        compressed_model = compress_model(path_to_obj_file, target_vertices)
        vm = compressed_model.current_mesh().vertex_matrix()
        
        with open(os.path.join(models_dir, directory, 'model.npy'), 'wb') as f:
            np.save(f, vm)
            count += 1
    print('Compressed: ', count, ' models')
   