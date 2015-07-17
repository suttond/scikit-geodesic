# Load packages which form part of the Python 2.7 core
import multiprocessing
import math

# Load additional packages and check if they are installed
try:
    import numpy as np
except ImportError as e:
    print 'NumPy is not installed. Try to run *pip install numpy*.'
    quit()

try:
    from scipy.optimize import fmin_l_bfgs_b, check_grad
    from scipy.linalg import orth
except ImportError as e:
    print 'SciPy is not installed. Try to run *pip install scipy*.'
    quit()


def generate_points(x, start_point, end_point, rotation_matrix, total_number_of_points, co_dimension):
    """ This function computes the local geodesic curve joining start_point to end_point using the L-BFGS method.

    Args:
      x (numpy.array) :
          Array of vectors in co-dimension dimensional space, stacked flat. This vector characterises points of the
          curve as translations from the line joining start_point to end_point.
      start_point (numpy.array) :
          The first end point of the curve.
      end_point (numpy.array) :
          The last end point of the curve.
      rotation_matrix (numpy.array) :
          A numpy.array describing the rotation from co_dimension + 1 dimensional space to the tangent space of the line
          joining start_point to end_point.
      total_number_of_points (int):
          The total number of points used in the local geodesic computation, including endpoints.
      co_dimension (int):
          The dimension of the configuration space, less one.

    Returns:
      numpy.array :
          The midpoint along the approximate local geodesic curve.

    """

    # Compute tangent direction of line joining start and end points
    tangent = np.subtract(end_point, start_point)/(total_number_of_points-1)

    # Initialise list to store points
    points = []

    # Generate points that are uniformly distributed along the initial line
    for i in xrange(total_number_of_points):
        points.append(np.add(start_point, float(i)*tangent))

    # Shift the points as encoded in x
    for i in xrange(0, len(x)/co_dimension):

        # Embed vector i into co_dimension + 1 dimensional space
        unrotated_shift = np.hstack((np.zeros(1), x[i*co_dimension:(i+1)*co_dimension]))

        # Convert vector, by rotation, from shift from e_1 basis direction to shift from tangent direction
        shift = rotation_matrix.dot(unrotated_shift)

        # Append point to list
        points[i+1] = np.add(points[i+1], shift)

    return points


def compute_metric(points, metric_function):
    """ Takes a list of NumPy arrays describing points molecular configurations, evaluates the metric at each point and
    returns a list of metric values at those points.

    Args:
      points (list) :
          A list of NumPy arrays describing molecular configurations.
      metric_function (func) :
          A Python function which gives the value of sqrt(2(E - V)) at a given point.


    Returns:
      list :
          A list of metric values at the corresponding points.

    """

    # Initialise the list to store metric values
    metric = []

    # For each point, compute the metric at that point
    for point in points:
        metric.append(metric_function(point))

    return metric


def norm(x, matrix):
    """ Computes the value of sqrt(<x, matrix*x>).

    Args:
      x (numpy.array) :
          A vector, stored as a NumPy array, to compute the norm for.
      matrix (numpy.array) :
          A matrix, stored as a NumPy array, used in the computation of <x, matrix*x>.


    Returns:
      float :
          The value of sqrt(<x, matrix*x>).

    """

    return math.sqrt(np.inner(x, matrix.dot(x)))


def norm_gradient(x, matrix):
    """ Computes the gradient of sqrt(<x, matrix*x>).

    Args:
      x (numpy.array) :
          A vector, stored as a NumPy array, to compute the norm for.
      matrix (numpy.array) :
          A matrix, stored as a NumPy array, used in the computation of <x, matrix*x>.


    Returns:
      numpy.array :
          The gradient of sqrt(<x, matrix*x>).

    """

    a = (matrix + matrix.transpose())

    return a.dot(x) / (2 * norm(x, matrix))


def length(x, start_point, end_point, rotation_matrix, total_number_of_points, co_dimension, metric):
    """ This function computes the length of the local geodesic as a function of shifts from the line joining
    start_point to end_point. It also returns the gradient of this function for the L-BFGS method.

    Args:
      x (numpy.array) :
          Array of vectors in co-dimension dimensional space, stacked flat. This vector characterises points of the
          curve as translations from the line joining start_point to end_point.
      start_point (numpy.array) :
          The first end point of the curve.
      end_point (numpy.array) :
          The last end point of the curve.
      rotation_matrix (numpy.array) :
          A numpy.array describing the rotation from co_dimension + 1 dimensional space to the tangent space of the line
          joining start_point to end_point.
      total_number_of_points (int) :
          The total number of points used in the local geodesic computation, including endpoints.
      co_dimension (int) :
          The dimension of the configuration space, less one.
      metric (func) :
          A Python function which when given a list of NumPy arrays, returns a list of metric values on those arrays.

    Returns:
      float :
          The approximate length of the geodesic.
      numpy.array :
          The gradient of the approximate length of the geodesic.

    """

    # Convert the shifts x into points in the full dimensional space
    points = generate_points(x, start_point, end_point, rotation_matrix, total_number_of_points, co_dimension)

    # Pre-compute the metric values to minimise repeated metric evaluations
    a = compute_metric(points, metric)

    # Compute quantities used to determine the length and gradient
    n = np.subtract(points[1], points[0])
    b = np.linalg.norm(n)
    c = norm_gradient(n, np.eye(co_dimension+1, co_dimension+1))
    u = (a[1][0]+a[0][0])

    # Initialise the length with the trapezoidal approximation of the first line segments length
    l = u * b
    # Initialise a list to store the gradient
    g = []

    for i in xrange(1, len(points)-1):

        # Compute the quantities needed for the next trapezoidal rule approximation.
        n = np.subtract(points[i+1], points[i])
        d = np.linalg.norm(n)
        e = norm_gradient(n, np.eye(co_dimension+1, co_dimension+1))
        v = (a[i+1][0]+a[i][0])

        # Add length of line segment to total length
        l += v * d

        # Compute next gradient component and update gradient
        g.append(rotation_matrix.transpose().dot(a[i][1] * (b + d) + u * c - v * e)[1:])

        # Pass back calculated values for efficiency
        b = d
        c = e
        u = v

    return 0.5 * l, 0.5 * np.asarray(g).flatten()


def get_rotation(start_point, end_point, dimension):
    """ Computes the transformation from dimension dimensional space to the tangent space of the line
          joining start_point to end_point.

    Args:
      start_point (numpy.array) :
          The first end point of the curve.
      end_point (numpy.array) :
          The last end point of the curve.
      dimension (int) :
          The dimension of the configuration space.

    Returns:
      numpy.array :
          The matrix representing the linear transformation from dimension dimensional space to the tangent space of
          the line joining start_point to end_point.

    """

    # Compute tangent direction of line joining start and end points
    tangent = np.subtract(end_point, start_point)

    # Set the first column of our output matrix as tangent
    mx = tangent

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
    m, n = np.shape(mx)
    Q = np.zeros([m, n])
    R = np.zeros([n, n])
    v = np.zeros(m)

    for j in range(n):

        v[:] = mx[:,j]
        for i in range(j):
            r = np.dot(Q[:,i], mx[:,j]); R[i,j] = r
            v[:] = v[:] - r*Q[:,i]
        r = np.linalg.norm(v); R[j,j]= r
        Q[:,j] = v[:]/r

    return Q


def find_geodesic_midpoint(start_point, end_point, number_of_inner_points, dimension, node_number, metric, grad_metric):
    """ This function computes the local geodesic curve joining start_point to end_point using the L-BFGS method.

    Args:
      start_point (numpy.array) :
          The first end point of the curve.
      end_point (numpy.array) :
          The last end point of the curve.
      number_of_inner_points (int) :
          The number of nodes along the curve, less the end points.
      dimension (int) :
          The dimension of the problem. Computed from the atomistic simulation environment.
      node_number (int) :
          The node number for which we are calculating a new position for.
      metric (func) :
          A Python function taking a single vector argument the same dimension as the start point.
      grad_metric (func) :
          A Python function taking a single vector argument the same dimension as the start point. Returning a vector
          the same length as it's argument.

    Returns:
      numpy.array :
          The midpoint along the approximate local geodesic curve.

    """

    # Define a function that returns sqrt(2(E-V)) and it's gradient based on a given configuration
    def metric_and_metric_grad(point):

        # Return sqrt(2(E-V)) and it's gradient
        return [metric(point), grad_metric(point)]

    Q = get_rotation(start_point, end_point, dimension)

    # Perform the L-BFGS method on the length functional
    geodesic, f_min, detail = fmin_l_bfgs_b(func=length, x0=np.zeros(number_of_inner_points*(dimension-1)),
                                            args=(start_point, end_point, Q,  number_of_inner_points+2,
                                                  dimension-1, metric_and_metric_grad,))

    # If something went wrong with the L-BFGS algorithm print an error message for the end user
    if detail['warnflag'] != 0:
        print 'BFGS Warning:' + detail['task']


    # Convert the obtained geodesic from it's shift description to the full point description
    points = np.reshape(generate_points(geodesic, start_point, end_point, Q, number_of_inner_points+2, dimension-1),
                        (number_of_inner_points+2, dimension))

    # Compute the midpoint
    if number_of_inner_points % 2 == 1:
        # If there is an odd number of inner points then return the middle element of the array
        midpoint = points[(number_of_inner_points + 1) / 2]
    else:
        # If there is an even number of inner points return the midpoint of the two middle points - this prevents
        # artificial movement of the curve due to the algorithm.
        midpoint = 0.5 * (points[number_of_inner_points / 2] + points[(number_of_inner_points / 2) + 1])

    # Return the node number and new midpoint
    return [node_number, midpoint]


def compute_geodesic(curve_obj, local_num_nodes, tol, metric, grad_metric, processes=1):
    """ This function creates a new task to compute a geodesic midpoint and submits it to the worker pool.

    Args:
      curve_obj (curve) :
          A GeometricMD curve object describing the initial trajectory between start and end configurations.
      local_num_nodes (int) :
          The number of points to use when computing the local geodesics.
      tol (float) :
          The tolerance by which if the total curve movement falls below this number then the Birkhoff method stops.
      metric (func) :
          A Python function taking a single vector argument the same dimension as the start point.
      processes (optional, int) :
          The number of processes to parallelise the task over.

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
                                                                                 node_number,
                                                                                 metric,
                                                                                 grad_metric)[1])

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
                                       local_num_nodes, dimension, node_number, metric, grad_metric),
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

