import os
import glob
import multiprocessing as mp
from multiprocessing import Pool
import trimesh
import sys
import traceback
import logging


logger = logging.getLogger()
logger.setLevel(logging.ERROR)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    Suggested by https://github.com/mikedh/trimesh/issues/507
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def to_off():

    file_name = "airplane_0627.off"
    output_file = os.path.join('airplane_0627_scaled.off')

    if os.path.exists(output_file):
        print('Exists: {}'.format(output_file))
        return

    try:
        with HiddenPrints():
            input = trimesh.load(file_name)
            mesh = as_mesh(input)
            total_size = (mesh.bounds[1] - mesh.bounds[0]).max()
            centers = (mesh.bounds[1] + mesh.bounds[0]) / 2

            mesh.apply_translation(-centers)
            mesh.apply_scale(1 / total_size)
            mesh.export(output_file)
        print('Finished: {}'.format(file_name))

    except:
        print('Error with {}: {}'.format(file_name, traceback.format_exc()))

