#############################################################################
# Author:
# Hao Zhai @ MiRA, CASIA
# Description:
# Toward a transformation from sparse segmentation files 
# to CloudVolume precomputed format.
# Date:
# Oct 27, 2023
# Reference:
# https://github.com/seung-lab/igneous
# https://github.com/seung-lab/kimimaro
#############################################################################


from cloudvolume import CloudVolume
import kimimaro
from taskqueue import LocalTaskQueue
import igneous.task_creation as tc
import tinybrain

from tqdm import tqdm
from random import uniform
import numpy as np
try:
    import imageio.v2 as imageio
except:
    import imageio
import os
import time
import argparse
from concurrent.futures import ProcessPoolExecutor


parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input_path', type=str, required=True, 
    help='Input path of exported .png manual label files.'
)
parser.add_argument(
    '-c', '--cloudvolume_path', type=str, default='cv_tmp', 
    help='Intermediate path to store CloudVolume stack, '
        'in order to preparing for the skeletonization.'
)
parser.add_argument(
    '-o', '--output_path', type=str, required=True, 
    help='Output path to store calculated files.'
)
parser.add_argument(
    '-s', '--swc_range', nargs=2, type=int, required=True, 
    help='List of integers of [start end+1]-range for .swc files. '
        'If start >= end+1, doing nothing.'
)
parser.add_argument(
    '-b', '--obj_range', nargs=2, type=int, required=True, 
    help='List of integers of [start end+1]-range for .obj files. '
        'If start >= end+1, doing nothing.'
)
parser.add_argument(
    '-m', '--mtl_enable', action='store_true', 
    help='Whether to add .mtl files to .obj files, '
        'in order to show some random texture.'
)
parser.add_argument(
    '-d', '--divide_obj_enable', action='store_true', 
    help='Whether to divide single .obj file to multiple .obj files, '
        'in which each contains one label.'
)
parser.add_argument(
    '-r', '--resolution', nargs=3, type=int, default=[4, 4, 30], 
    help='List of integers of [x y z]-axis resolution.'
)
parser.add_argument(
    '-p', '--parallel', type=int, default=32, 
    help='Number of core to use for parallel processing.'
)
parser.add_argument(
    '-a', '--add_id', type=int, default=0, 
    help='New merged id number - original id number. '
        'If add_id == 0, rewrite the original id. '
        'If add_id > 0, add new id = the original id + add_id.'
)
parser.add_argument(
    '-t', '--threshold_radius', type=int, default=None, 
    help='Threshold when merging two skeletons. '
        'If distance between skeletons > radius_merge, then doing nothing. '
        'Default is None, means no limitation threshold.'
)
args = parser.parse_args()
time_s = time.time()


input_list = [ i for i in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, i)) ]
nonempty_input_list = []
offside_z_s = 26000
offside_z_e = -1
offside_x_s = 150000
offside_x_e = -1
offside_y_s = 100000
offside_y_e = -1
empty_input_layers = 0
exist_z_x_s = {}
exist_z_y_s = {}
image_name_z_x_y_s = {}
for input_layer in input_list:
    try:
        z = int(input_layer)
    except:
        print('ERROR: {:s} is not a valid folder.'.format(input_layer))
        raise TypeError
    layer_txt = open(os.path.join(args.input_path, input_layer, 'index.txt'), 'r')
    layer_x_s = 150000
    layer_x_e = -1
    layer_y_s = 100000
    layer_y_e = -1
    exist_z_x_s[z] = {}
    exist_z_y_s[z] = {}
    for line_num, line in enumerate(layer_txt.readlines()):
        if line_num == 0:
            if not line.startswith('0 0 2048 2048'):
                print('ERROR: folder {:d} first line of Index.'.format(z))
        else:
            image_name_1, image_name_2, x_s, y_s = line.split(' ')
            x_s = int(x_s); y_s = int(y_s)
            if x_s == 0 and y_s == 0:
                continue
            layer_x_s = min(layer_x_s, x_s)
            layer_x_e = max(layer_x_e, x_s + 2048)
            layer_y_s = min(layer_y_s, y_s)
            layer_y_e = max(layer_y_e, y_s + 2048)
            exist_z_x_s[z][x_s] = True
            exist_z_y_s[z][y_s] = True
            image_name_z_x_y_s[(z, x_s, y_s)] = (image_name_1, image_name_2)
    if layer_x_s >= layer_x_e or layer_y_s >= layer_y_e:
        empty_input_layers += 1
    else:
        nonempty_input_list.append(input_layer)
        offside_z_s = min(offside_z_s, z)
        offside_z_e = max(offside_z_e, z + 1)
        offside_x_s = min(offside_x_s, layer_x_s)
        offside_x_e = max(offside_x_e, layer_x_e)
        offside_y_s = min(offside_y_s, layer_y_s)
        offside_y_e = max(offside_y_e, layer_y_e)
print('WARN: {:d} of {:d} layers are empty.'.format(empty_input_layers, len(input_list)))
print('INFO: from [{:d}, {:d}, {:d}].'.format(offside_x_s, offside_y_s, offside_z_s))
print('INFO: to [{:d}, {:d}, {:d}].'.format(offside_x_e, offside_y_e, offside_z_e))

stack_num_z = 32
stack_num_xy_d = 8
exist_z_x_y_s = {}
exist_z_x_y_count = 0
for z_s in range(offside_z_s, offside_z_e, stack_num_z):
    for x_s in range(offside_x_s, offside_x_e, 2048):
        for y_s in range(offside_y_s, offside_y_e, 2048):
            z_e = min(z_s + stack_num_z, offside_z_e)
            for z in range(z_s, z_e):
                if z not in exist_z_x_s.keys() or z not in exist_z_y_s.keys():
                    continue
                if exist_z_x_s[z].get(x_s, False) and exist_z_y_s[z].get(y_s, False):
                    exist_z_x_y_s[z_s] = exist_z_x_y_s.get(z_s, {})
                    exist_z_x_y_s[z_s][x_s] = exist_z_x_y_s[z_s].get(x_s, {})
                    exist_z_x_y_s[z_s][x_s][y_s] = True
                    exist_z_x_y_count += 1
                    break
print('INFO: stack number {:d}.'.format(exist_z_x_y_count))      

cloudpath = 'precomputed://file://' + args.cloudvolume_path
vol = CloudVolume(cloudpath)
print(vol.info)
'''
info = CloudVolume.create_new_info(
    num_channels 	= 1,
    layer_type	= 'segmentation',
    data_type	= 'uint32',
    encoding	= 'compressed_segmentation',
    resolution	= [
        args.resolution[0]*stack_num_xy_d, 
        args.resolution[1]*stack_num_xy_d, 
        args.resolution[2]
    ],
    voxel_offset	= [0, 0, 0],
    chunk_size	= [2048//stack_num_xy_d, 2048//stack_num_xy_d, stack_num_z],
    volume_size	= [
        offside_x_e//stack_num_xy_d - offside_x_s//stack_num_xy_d,
        offside_y_e//stack_num_xy_d - offside_y_s//stack_num_xy_d,
        offside_z_e - offside_z_s
    ]
)
vol = CloudVolume(cloudpath, compress='gzip', info=info)
vol.commit_info()
vol.provenance.description = "Intermediate Data for Generating .swc Files"
vol.provenance.owners = ['zhaihao2020@ia.ac.cn']
vol.commit_provenance()
print('INFO: size', vol.shape)

process_list = []
for z_s in exist_z_x_y_s.keys():
    z_e = min(z_s + stack_num_z, offside_z_e)
    for x_s in exist_z_x_y_s[z_s].keys():
        for y_s in exist_z_x_y_s[z_s][x_s].keys():
            if not exist_z_x_y_s[z_s][x_s].get(y_s, False):
                continue
            process_list.append('{:d}_{:d}_{:d}'.format(z_s, x_s, y_s))
def _process(process_str):
# for process_str in tqdm(process_list):
    z_s, x_s, y_s = process_str.split('_')
    z_s = int(z_s); x_s = int(x_s); y_s = int(y_s)    
    z_e = min(z_s + stack_num_z, offside_z_e)
    x_e = min(x_s+2048, offside_x_e)
    y_e = min(y_s+2048, offside_y_e)
    image_stack = np.zeros(
        (
            (x_e-x_s)//stack_num_xy_d,
            (y_e-y_s)//stack_num_xy_d,
            z_e - z_s
        ), 
        dtype=np.uint32
    )
    for z in range(z_s, z_e):
        input_layer = '{:010d}'.format(z)
        if input_layer not in nonempty_input_list:
            continue
        if (z, x_s, y_s) not in image_name_z_x_y_s.keys():
            continue
        image_name_1, image_name_2 = image_name_z_x_y_s[(z, x_s, y_s)]
        image = imageio.imread(os.path.join(args.input_path, input_layer,
            image_name_1 + '_' + image_name_2 + '.png'))
        assert image.shape[0] == 2048 and image.shape[1] == 2048
        
        image = image.transpose()
        image = image[:(x_e-x_s), :]
        image = image[:, :(y_e-y_s)]
        
        image = tinybrain.downsample_segmentation(
            image, 
            factor=(stack_num_xy_d,stack_num_xy_d,1), 
            num_mips=1, sparse=True
        )[0]
        
        image = image.astype(np.uint32)
        image_stack[:, :, z-z_s] = image[:image_stack.shape[0], :image_stack.shape[1]]

    vol[
        (x_s-offside_x_s)//stack_num_xy_d:(x_e-offside_x_s)//stack_num_xy_d,
        (y_s-offside_y_s)//stack_num_xy_d:(y_e-offside_y_s)//stack_num_xy_d,
        z_s-offside_z_s:z_e-offside_z_s
    ] = image_stack[:, :, :, np.newaxis]
    # print('INFO: stack [{:d}, {:d}] is done.'.format(z_s, z_e))

with ProcessPoolExecutor(max_workers=args.parallel) as executor:
    executor.map(_process, process_list)
print('INFO: all layer is into cloudvolume, {:.2f} s.'.format(time.time()-time_s))
'''
tq = LocalTaskQueue(parallel=args.parallel)

target_mip = 0
target_shape = (512, 512, 512)

if args.obj_range[0] < args.obj_range[1]:
    tasks = tc.create_meshing_tasks(
        cloudpath,
        mip=target_mip, 
        shape=target_shape, 
        fill_missing=True,
        # mesh_dir='mesh_mip_1'
        mesh_dir='mesh_mip_0',
    )
    tq.insert(tasks)
    tq.execute()
    print('INFO: meshing done, {:.2f} s.'.format(time.time()-time_s))
    tasks = tc.create_mesh_manifest_tasks(cloudpath)
    tq.insert(tasks)
    tq.execute()
    print('INFO: meshing manifested, {:.2f} s.'.format(time.time()-time_s))

if args.swc_range[0] < args.swc_range[1]:
    tasks = tc.create_skeletonizing_tasks(
        cloudpath,
        mip=target_mip,
        shape=target_shape,
        fill_missing=True,
        teasar_params={
            'scale': 4, 
            'const': 500,
            'pdrf_exponent': 4,
            'pdrf_scale': 100000,
            'soma_detection_threshold': 1100,
            # 'soma_acceptance_threshold': 3500,
            'soma_acceptance_threshold': 2500,
            # 'soma_invalidation_scale': 1,
            'soma_invalidation_scale': 6,
            'soma_invalidation_const': 300,
            'max_paths': None
        }, # parameters from kimimaro (TEASAR)
        dust_threshold=1000
    )
    tq.insert(tasks)
    tq.execute()
    print('INFO: skeletonization done, {:.2f} s.'.format(time.time()-time_s))

    tasks = tc.create_unsharded_skeleton_merge_tasks(
        cloudpath,
        magnitude=3,
        dust_threshold=1000,
        tick_threshold=3500
    )
    tq.insert(tasks)
    tq.execute()
    print('INFO: skeletonization merged, {:.2f} s.'.format(time.time()-time_s))

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    vol = CloudVolume(cloudpath)

    for id in tqdm(range(args.swc_range[0], args.swc_range[1])):
        try:
            skel = vol.skeleton.get(id)
        except:
            print('WARN: id {:d} is missing.'.format(id))
            continue
        skel = kimimaro.join_close_components([skel], radius=args.threshold_radius)
        skel.id += args.add_id
        vol.skeleton.upload(skel)
        
        skel = vol.skeleton.get(id)
        skel.vertices[:, 0] += offside_x_s * args.resolution[0]
        skel.vertices[:, 1] += offside_y_s * args.resolution[1]
        skel.vertices[:, 2] += offside_z_s * args.resolution[2]
        skel_str = skel.to_swc()
        with open(os.path.join(args.output_path, '{:d}.swc'.format(id)), 'w') as f:
            f.write(skel_str)
        if args.add_id > 0:
            skel = vol.skeleton.get(id + args.add_id)
            skel.vertices[:, 0] += offside_x_s * args.resolution[0]
            skel.vertices[:, 1] += offside_y_s * args.resolution[1]
            skel.vertices[:, 2] += offside_z_s * args.resolution[2]
            skel_str = skel.to_swc()
            with open(os.path.join(args.output_path, '{:d}.swc'.format(id + args.add_id)), 'w') as f:
                f.write(skel_str)
        # print('INFO: id {:d} is done.'.format(id))
    print('INFO: all id is exported to swc files, {:.2f} s.'.format(time.time()-time_s))

if args.obj_range[0] < args.obj_range[1]:
    if args.divide_obj_enable:
        for id in tqdm(range(args.obj_range[0], args.obj_range[1])):
            '''
            Ka: ambient color (r, g, b)
            Kd: diffuse color (r, g, b)
            Ks: specular color (r, g, b)
            Ns: specular exponent (0.0 - 1000.0)
            d: non-transparency
            illum: 2 means Ks is used
            '''
            d = 1.0; Ns = 100.0; illum = 2
            try:
                mesh = vol.mesh.get(id)
            except:
                print('WARN: id {:d} is missing.'.format(id))
                continue
            obj_filename = '{:d}.obj'.format(id)
            mtl_filename = '{:d}.mtl'.format(id)
            obj_data = ''
            mtl_data = ''
            obj_data += 'mtllib ' + mtl_filename + '\n'

            obj_data += 'g ' + str(id) + '\n'
            obj_vertices = [ 'v {:.5f} {:.5f} {:.5f}'.format(*vertex) for vertex in mesh.vertices ]
            obj_data += '\n'.join(obj_vertices) + '\n'
            if args.mtl_enable:
                obj_data += 'usemtl mat_' + str(id) + '\n'
            obj_data += 's {:d}'.format(1) + '\n'
            obj_faces = [ 'f {} {} {}'.format(*face) for face in (mesh.faces+1) ] # obj is 1 indexed
            obj_data += '\n'.join(obj_faces) + '\n'
            obj_data += '\n'
            r = uniform(0.0, 1.0)
            g = uniform(0.0, 1.0)
            b = uniform(0.0, 1.0)
            mtl_data += 'newmtl mat_' + str(id) + '\n'
            mtl_data += 'Ns {:.5f}'.format(Ns) + '\n'
            mtl_data += 'Ka {:.5f} {:.5f} {:.5f}'.format(r, g, b) + '\n'
            mtl_data += 'Kd {:.5f} {:.5f} {:.5f}'.format(r, g, b) + '\n'
            mtl_data += 'Ks {:.5f} {:.5f} {:.5f}'.format(0.5, 0.5, 0.5) + '\n'
            mtl_data += 'd {:.5f}'.format(d) + '\n'
            mtl_data += 'illum {:d}'.format(illum) + '\n'
            mtl_data += '\n'

            obj_data = obj_data.encode('utf8')
            mtl_data = mtl_data.encode('utf8')

            with open(os.path.join(args.output_path, obj_filename), 'wb') as f:
                f.write(obj_data)
            if args.mtl_enable:
                with open(os.path.join(args.output_path, mtl_filename), 'wb') as f:
                    f.write(mtl_data)
            # print('INFO: id {:d} is done.'.format(id))
    else:
        d = 1.0; Ns = 100.0; illum = 2
        if args.obj_range[0] < args.obj_range[1] - 1:
            obj_filename = '{:d}-{:d}.obj'.format(args.obj_range[0], args.obj_range[1]-1)
            mtl_filename = '{:d}-{:d}.mtl'.format(args.obj_range[0], args.obj_range[1]-1)
        else:
            obj_filename = '{:d}.obj'.format(args.obj_range[0])
            mtl_filename = '{:d}.mtl'.format(args.obj_range[0])
        obj_data = ''
        mtl_data = ''
        obj_data += 'mtllib ' + mtl_filename + '\n'
        obj_faces_start = 0

        for id in tqdm(range(args.obj_range[0], args.obj_range[1])):   
            try:
                mesh = vol.mesh.get(id)
            except:
                print('WARN: id {:d} is missing.'.format(id))
                continue
            
            obj_data += 'g ' + str(id) + '\n'
            obj_vertices = [ 'v {:.5f} {:.5f} {:.5f}'.format(*vertex) for vertex in mesh.vertices ]
            obj_data += '\n'.join(obj_vertices) + '\n'
            if args.mtl_enable:
                obj_data += 'usemtl mat_' + str(id) + '\n'
            obj_data += 's {:d}'.format(1) + '\n'
            obj_faces = [ 'f {} {} {}'.format(*face) for face in (mesh.faces+obj_faces_start+1) ] # obj is 1 indexed
            obj_data += '\n'.join(obj_faces) + '\n'
            obj_data += '\n'
            obj_faces_start += len(mesh.vertices)
            r = uniform(0.0, 1.0)
            g = uniform(0.0, 1.0)
            b = uniform(0.0, 1.0)
            mtl_data += 'newmtl mat_' + str(id) + '\n'
            mtl_data += 'Ns {:.5f}'.format(Ns) + '\n'
            mtl_data += 'Ka {:.5f} {:.5f} {:.5f}'.format(r, g, b) + '\n'
            mtl_data += 'Kd {:.5f} {:.5f} {:.5f}'.format(r, g, b) + '\n'
            mtl_data += 'Ks {:.5f} {:.5f} {:.5f}'.format(0.5, 0.5, 0.5) + '\n'
            mtl_data += 'd {:.5f}'.format(d) + '\n'
            mtl_data += 'illum {:d}'.format(illum) + '\n'
            mtl_data += '\n'
            # print('INFO: id {:d} is done.'.format(id))

        obj_data = obj_data.encode('utf8')
        mtl_data = mtl_data.encode('utf8')

        with open(os.path.join(args.output_path, obj_filename), 'wb') as f:
            f.write(obj_data)
        if args.mtl_enable:
            with open(os.path.join(args.output_path, mtl_filename), 'wb') as f:
                f.write(mtl_data)
    print('INFO: all id is exported to obj files, {:.2f} s.'.format(time.time()-time_s))