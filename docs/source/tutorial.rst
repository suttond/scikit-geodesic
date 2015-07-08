Tutorial
========

In this tutorial we will use the scikit-geodesic package to compute a geodesic in an isotropic Riemannian manifold with
coefficient exp(-<n,x>). The script is available in the *example* directory of the code.

.. code-block:: python

    # An Example Script Illustrating how to find the geodesic for an isotropic Riemannian manifold with metric coefficient
    # exp(-<n,x>) where n is a constant vector.
    from math import exp
    import numpy as np
    from geodesic.geometry import Curve
    from geodesic.curve_shorten import compute_geodesic
    from multiprocessing import cpu_count

    # Set dimension of the problem
    dimension = 4

    # Set parameters for computation
    number_of_global_nodes = 16
    number_of_local_nodes = 8
    maximum_average_node_movement = 0.001
    number_of_cpu = cpu_count()

    # Create start and end point NumPy arrays
    start_point = np.zeros(dimension)
    start_point[0] = -1

    end_point = np.zeros(dimension)
    end_point[0] = 1

    # Create constant vector n
    alpha = 0.65
    n = alpha*np.ones(dimension)
    n[0] = 0

    # Define function to describe metric coefficient
    def metric_coefficient(x):
        return exp(-np.inner(n,x))

    print 'Starting Example Calculation...'

    # Create curve object for calculation
    curve = Curve(start_point, end_point, number_of_global_nodes)

    # Apply curve shortening procedure to minimise length
    compute_geodesic(curve, number_of_local_nodes, maximum_average_node_movement, metric_coefficient, number_of_cpu)

    # Print shortened curve points
    print curve.get_points()
