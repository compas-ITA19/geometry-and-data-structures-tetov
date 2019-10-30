''' Assignment for module0, 02_geometry_and_datastructures in ITA19 COMPAS course
'''

__author__ = "Anton T Johansson"

from math import sqrt
from typing import List, Iterator

import numpy as np

from compas import get
from compas.datastructures import Mesh
from compas_plotters import MeshPlotter

# HELPERS


def vector_2pt_xy(point1: List[float], point2: List[float]) -> List[float]:
    ''' Returns vector between two points

        >>> vector_2pt_xy([-1., -4., 8.], [2., 9., 2.])
        [3.0, 13.0, -6.0]
    '''

    if len(point1) != len(point2):
        raise ValueError

    vector = []
    for value1, value2 in zip(point1, point2):
        vector.append(value2-value1)

    return vector


def flip_matrix(matrix: List) -> Iterator:
    ''' Flip matrix

        >>> list(flip_matrix([[1, 2], [3, 4]]))
        [(3, 1), (4, 2)]
    '''
    # flip matrix
    # https://stackoverflow.com/questions/8421337/rotating-a-two-dimensional-array-in-python
    return zip(*matrix[::-1])


def midpoint_convex_polygon(verts: List[List[float]]) -> List[float]:
    ''' Find polygon centroid given a 2d convex polygon defined by list of vertices
    '''

    flipped_verts = flip_matrix(verts)

    sum_components = [sum(comp) for comp in flipped_verts]

    return [comp / len(sum_components) for comp in sum_components]


def translate(p_from, p_to) -> List[float]:
    if len(p_from) != len(p_to):
        raise ValueError

    # make 3d if not 3d, and raise if not 2 to 3 dimensions
    if len(p_from) == 2:
        p_from.append(0.)
        p_to.append(0.)
    elif len(p_from) != 3:
        raise ValueError

    flipped_from = flip_matrix(p_from)
    flipped_to = flip_matrix(p_to)

    x, y, z = zip(flipped_from, flipped_to)

    return [x[1] - x[0], y[1] - y[0], z[1], z[0]]


def cross_product(vector1: List[float], vector2: List[float]) -> List[float]:
    return [vector1[1] * vector2[2] - vector1[2] * vector2[1],
            vector1[2] * vector2[0] - vector1[0] * vector2[2],
            vector1[0] * vector2[1] - vector1[1] * vector2[0]]


def magnitude_vector(v: List[float]) -> float:
    ''' Return magnitude of vector

        >>> magnitude_vector([5., -1., 2.]) == sqrt(30)
        True
        >>> magnitude_vector([-2, -6, 3])
        7.0
    '''
    # pad if not 3d, raise if not 2 or 3 dimensional
    if len(v) == 2:
        v.append(0.)
    elif len(v) != 3:
        raise ValueError

    return sqrt(v[0]**2 + v[1]**2 + v[2]**2)


def normalize_vector(v: List[float]) -> List[float]:
    ''' Return unit vector

        >>> normalize_vector([4, 0, 3])
        [0.8, 0.0, 0.6]
    '''
    # pad if not 3d, raise if not 2 or 3 dimensional
    if len(v) == 2:
        v.append(0.)
    elif len(v) != 3:
        raise ValueError

    magnitude = magnitude_vector(v)

    return [c / magnitude for c in v]


def traverse_mesh(mesh, start_key) -> List[int]:
    ''' Calculates path through vertices from right to left on a rectaungular 2d mesh
    '''

    # get dict with keys and coords
    vert_coords = {}
    for vkey in mesh.vertices():
        vert_coords[vkey] = mesh.vertex_coordinates(vkey)

    # save path
    traversed_verts = []

    # start from user input vert
    start_vert = vert_coords.pop(start_key)
    traversed_verts.append(start_key)

    # setup goal x value
    x_values = []
    for key in vert_coords:
        x_values.append(vert_coords[key][0])

    x_goal = min(x_values)

    while start_vert[0] != x_goal:

        # find magnitudes of vectors between vertices and start
        magnitudes = {}
        for key in vert_coords:
            # TODO: Remove points from dict if x < start_x
            if vert_coords[key][0] < start_vert[0]:
                vector = vector_2pt_xy(start_vert, vert_coords[key])
                magnitude = magnitude_vector(vector)
                magnitudes[key] = magnitude

        key_min = min(magnitudes.keys(), key=(lambda k: magnitudes[k]))

        traversed_verts.append(key_min)
        start_vert = vert_coords.pop(key_min)

    return traversed_verts


# TASKS

def orthonormal_vectors(vector1: List[float], vector2: List[float]) -> List[List[float]]:
    ''' Geometry task 1
        Returns orthonormal basis given two vectors


        >>> orthonormal_vectors([2,0,0], [1,2,0])
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
    '''
    first = normalize_vector(vector1)

    second = normalize_vector(cross_product(vector1, vector2))

    # no need to noramlize since input is already normalized
    third = cross_product(first, second)

    return [first, second, third]


def area_convex_polygon(verts: List[List[float]]) -> float:
    ''' Geometry task 2
        Returns area of convex polygon defined by corner points
        >>>
    '''
    midpoint = midpoint_convex_polygon(verts)

    # move midpoint to origo
    translated_verts = []
    for v in verts:
        translated_verts.append(translate(v, midpoint))

    # get vector n to n+1 and n to midpoint (latter is same as n)
    # feed those to cross_prod funct
    areas = []
    for i, tv in enumerate(translated_verts):
        n = tv
        n_plus = translated_verts[i % len(translated_verts)]

        to_n_plus = vector_2pt_xy(n, n_plus)
        to_mid = vector_2pt_xy(n, midpoint)

        cross = cross_product(to_n_plus, to_mid)

        areas.append(magnitude_vector(cross))

    return sum(areas)


def task3_wo_numpy(vectors1: List[List[float]], vectors2: List[List[float]]) -> List[List[float]]:
    ''' Geometry task 3a

        >>> task3_wo_numpy([[2, 5, -1], [12, 15, 0], [9, 8, 1]], [[ -12, -4, 1], [9, 4, 1], [0, 5, 2]])
        [[1, 10, 52], [15, -12, -87], [11, -18, 45]]

    '''

    results = []
    for v1, v2 in zip(vectors1, vectors2):
        cross = cross_product(v1, v2)
        results.append(cross)

    return results


def task3_w_numpy(vectors1: List[float], vectors2: List[float]) -> List[float]:
    ''' Geometry task 3b

        >>> task3_w_numpy([[2, 5, -1], [12, 15, 0], [9, 8, 1]], [[ -12, -4, 1], [9, 4, 1], [0, 5, 2]])
        array([[  1,  10,  52],
               [ 15, -12, -87],
               [ 11, -18,  45]])

    '''

    v1_array = np.array(vectors1)
    v2_array = np.array(vectors2)

    return np.cross(v1_array, v2_array)


def visualize_mesh_traversal() -> None:
    ''' Datastructures task
    '''

    mesh = Mesh.from_obj(get('faces.obj'))

    x_values = {}
    for vkey in mesh.vertices_on_boundary():
        x_values[vkey] = mesh.vertex_coordinates(vkey)[0]

    max_x = max(x_values.values())

    print("Vertices on the right edge of the mesh:")
    print([key for key in x_values if x_values[key] == max_x])

    start_key = int(input("\nSelect start vertex: "))

    path_verts = traverse_mesh(mesh, start_key)

    print('\nPath calculated, starting MeshPlotter.')

    plotter = MeshPlotter(mesh, figsize=(16, 10))
    plotter.draw_vertices(
        text={key: key for key in path_verts},
        radius=0.2,
        facecolor={key: '#ff0000' for key in path_verts})

    plotter.draw_edges()
    plotter.draw_faces()

    plotter.show()


if __name__ == "__main__":
    import doctest
    print("\nModule0, 02_datastructures_and_geometry assignment by " + __author__ + "\n")
    print("The geometry tasks are demonstrated through their doctests.")
    print("The datastructures task is demonstrated using the mesh plotter.\n")

    print('\nRunning doctests...')
    results, _ = doctest.testmod()
    if results == 0:
        print('Doctest exited without errors.\n')

    visualize_mesh_traversal()
