# Load packages which form part of the Python 2.7 core
import multiprocessing


# Load additional packages and check if they are installed
try:
    import numpy as np
except ImportError as e:
    print 'NumPy is not installed. Try to run *pip install numpy*.'
    quit()

try:
    from scipy.optimize import fmin_l_bfgs_b
    from scipy.linalg import orth
except ImportError as e:
    print 'SciPy is not installed. Try to run *pip install scipy*.'
    quit()


def length(x, start_point, end_point, number_of_inner_nodes, dimension, metric, basis_matrix):
    """ This function computes an approximation of the length functional for local geodesics. It also provides the
    gradient of this approximation.

    Args:
      x (numpy.array): The positions of the interior curve points, as shifts in the orthonormal direction to the tangent of the line joining start_point to end_point.
      start_point (numpy.array): The first end point of the curve.
      end_point (numpy.array): The last end point of the curve.
      number_of_inner_points (int): The number of nodes along the curve, less the end points.
      dimension (int): The dimension of the problem. Computed from the atomistic simulation environment.
      metric (func): A Python function taking a single vector argument the same dimension as the start point.
      basis_matrix (numpy.array): A dimension x dimension matrix describing the rotation of the line joining start_point to end_point over to the first basis direction.

    Returns:
      float, numpy.array: The approximate length of the curve and the corresponding gradient.

    """

    # Convert the vector x into points describing a curve in dimension dimensional space
    state = [start_point]
    for i in xrange(0, number_of_inner_nodes):
        # Embed the shift into dimension dimensional space and rotate, then place in appropriate position relative to initial line
        state.append(basis_matrix.dot(np.hstack((np.zeros(1), x[(i)*(dimension-1):(i+1)*(dimension-1)]))) \
               + (float(i+1)/float(number_of_inner_nodes+1)) * np.subtract(end_point, start_point) + start_point)
    state.append(end_point)

    # Initialise the length variable
    l = 0.0

    # For all of the interior nodes...
    for i in xrange(0, number_of_inner_nodes+1):
        # Compute length contribution
        l += np.linalg.norm(np.subtract(state[i+1], state[i])) * (metric(state[i+1]) + metric(state[i]))

    # Return the value of length
    return 0.5*l


def find_geodesic_midpoint(start_point, end_point, number_of_inner_points, dimension, node_number, metric):
    """ This function computes the local geodesic curve joining start_point to end_point using the L-BFGS method.

    Args:
      start_point (numpy.array): The first end point of the curve.
      end_point (numpy.array): The last end point of the curve.
      number_of_inner_points (int): The number of nodes along the curve, less the end points.
      dimension (int): The dimension of the problem. Computed from the atomistic simulation environment.
      node_number (int): The node number for which we are calculating a new position for.
      metric (func): A Python function taking a single vector argument the same dimension as the start point.

    Returns:
      numpy.array: The midpoint along the approximate local geodesic curve.

    """

    # Set the first column of our output matrix as tangent
    mx = np.subtract(end_point, start_point)/np.linalg.norm(np.subtract(end_point, start_point))

    # Find the first non-zero entry of the tangent vector (exists as start and endpoints are different)
    j = np.nonzero(mx)[0][0]

    # For the remaining dim - 1 columns choose unit basis vectors of the form (0,...,0,1,0,...,0) with the nonzero entry
    # not in position j.
    for i in xrange(1, dimension):
        if j != i:
            e = np.zeros(dimension)
            e[i] = 1
            mx = np.vstack((mx, e))

    mx = mx.transpose()

    # With the resulting matrix, perform the Gram-Schmidt orthonormalisation procedure on the transpose of the matrix
    # and return it.
    m,n = np.shape(mx)
    Q=np.zeros([m,n])
    R=np.zeros([n,n])
    v=np.zeros(m)

    for j in range(n):

        v[:] = mx[:,j]
        for i in range(j):
            r = np.dot(Q[:,i], mx[:,j]); R[i,j] = r
            v[:] = v[:] - r*Q[:,i]
        r = np.linalg.norm(v); R[j,j]= r
        Q[:,j] = v[:]/r

    basis_matrix = Q

    # Perform the L-BFGS method on the length functional
    geodesic, f_min, detail = fmin_l_bfgs_b(func=length, x0=np.zeros(number_of_inner_points*(dimension-1)),
                                            args=(start_point, end_point, number_of_inner_points, dimension, metric,
                                                  basis_matrix),
                                            approx_grad=1)

    # Determine the index of the middle value
    i = (number_of_inner_points) / 2

    # Extract and reform the midpoint value - see length function for details
    midpoint = basis_matrix.dot(np.hstack((np.zeros(1), geodesic[(i)*(dimension-1):(i+1)*(dimension-1)]))) \
               + (float(i+1)/float(number_of_inner_points+1)) * np.subtract(end_point, start_point) + start_point

    # Return the node number and new midpoint
    return [node_number, midpoint]


def compute_geodesic(curve_obj, local_num_nodes, tol, metric, processes=1):
    """ This function creates a new task to compute a geodesic midpoint and submits it to the worker pool.

    Args:
      curve_obj (curve): A GeometricMD curve object describing the initial trajectory between start and end configurations.
      local_num_nodes (int): The number of points to use when computing the local geodesics.
      tol (float): The tolerance by which if the total curve movement falls below this number then the Birkhoff method stops.
      metric (func): A Python function taking a single vector argument the same dimension as the start point.
      processes (optional, int): The number of processes to parallelise the task over.
   """

    # Determine the dimension of the Hamiltonian system
    dimension = len(curve_obj.get_points()[0])

    # If the user intends to use the algorithm on one core then...
    if processes == 1:

        # Main loop of the Birkhoff algorithm, continues until curve.movement < tol then breaks out
        while True:

            # Iterating over each node in the trajectory find a new position based on the geodesic midpoint
            # joining it's neighbours
            for node_number in curve_obj:
                curve_obj.set_node_position(node_number, find_geodesic_midpoint(curve_obj.points[node_number - 1],
                                                                                 curve_obj.points[node_number + 1],
                                                                                 local_num_nodes,
                                                                                 dimension,
                                                                                 node_number, metric)[1])

            # If the movement of the curve is below the tol threshold then exit the main loop
            if curve_obj.movement < tol:
                break

            # Indicate that the next iteration is to be completed
            curve_obj.set_node_movable()

    # Otherwise the user has indicated they would like to perform a parallel computation...
    else:

        # Create a callback function that updates the node position once it is calculated
        def update_curve(result):
            curve_obj.set_node_position(result[0], result[1])

        # Create a pool of worker processes to work in parallel
        pool = multiprocessing.Pool(processes=processes)

        # Main loop of the Birkhoff algorithm, continues until curve.movement < tol then breaks out
        while True:

            # Iterating over each node in the trajectory create a task to find a new position based on the
            # geodesic midpoint joining it's neighbours. Add this task to the pool queue.
            for node_number in curve_obj:
                pool.apply_async(func=find_geodesic_midpoint,
                                 args=(curve_obj.points[node_number - 1], curve_obj.points[node_number + 1],
                                       local_num_nodes, dimension, node_number, metric),
                                 callback=update_curve)

            # If all the nodes in the trajectory have been moved...
            if curve_obj.all_nodes_moved():

                # If the movement of the curve is below the tol threshold then exit the main loop
                if curve_obj.movement < tol:
                    break

                # Indicate that the next iteration is to be completed
                curve_obj.set_node_movable()

        # Once the algorithm has executed close the pool
        pool.close()
        pool.join()

