import numpy as np


class Curve:
    """

    The purpose of this object is to provide a Curve object that has similar behaviour to that described in [Sutton2013]_.

    Attributes:
      start_point (numpy.array): A NumPy array describing the first point in the curve.
      end_point (numpy.array): A NumPy array describing the last point in the curve.
      number_of_nodes (int): The total number of nodes that the curve is to consist of, including the start and end points.
      tangent (numpy.array): The tangent of the straight line segment joining the start_point to the end_point, rescaled according to [Sutton2013]_.
      points (numpy.array): An NumPy array containing all the points of the curve.
      default_initial_state (numpy.array): A NumPy array consisting of flags that indicate which nodes are movable initially.
      movement (float): A variable which records the average movement of the nodes.
      nodes_moved (numpy.array): A binary NumPy array indicating whether a node has been moved. Used to determine when all the nodes in the curve have been moved.
      node_movable (numpy.array): A binary NumPy array indicating whether a node is movable.
      number_of_distinct_nodes_moved (int): A counter recording the total number of nodes that have moved.

    """
    def __init__(self, start_point, end_point, number_of_nodes):
        """The constructor for the Curve class.

        Note:
          This class is intended to be used by the SimulationServer module.

        Args:
          start_point (ase.atoms): An ASE atoms object describing the initial state. A calculator needs to be set on this object.
          end_point (ase.atoms): An ASE atoms object describing the final state.
          number_of_nodes (int): The number of nodes that the curve is to consist of, including the start and end points.

        """

        # Pass the initialiser arguments directly into the class attributes
        self.start_point = start_point
        self.end_point = end_point
        self.number_of_nodes = int(number_of_nodes)

        # Compute the tangent vector - the rescaled vector of the line joining the start and end points
        self.tangent = (1/(float(self.number_of_nodes)-1))*np.subtract(self.end_point, self.start_point)

        # Compute the initial curve, the straight line joining the start point to the end point
        self.points = np.asarray([self.start_point], dtype='float64')
        for i in xrange(0, int(self.number_of_nodes-1)):
            self.points = np.concatenate((self.points, [np.add(self.points[i], self.tangent)]), axis=0)
        np.concatenate((self.points, [self.end_point]), axis=0)

        # Create and definite initial node_movable configuration. In this case even numbered nodes first.
        self.default_initial_state = np.zeros(self.number_of_nodes, dtype='int')
        for i in xrange(self.number_of_nodes-1):
            if i % 2 != 0:
                self.default_initial_state[i] = 2

        # Create all of the flags in the curve to indicate that the nodes in default_initial_state are movable.
        self.movement = 0.0
        self.nodes_moved = np.ones(self.number_of_nodes, dtype='int')
        self.node_movable = np.copy(self.default_initial_state)
        self.number_of_distinct_nodes_moved = 0


    def set_node_movable(self):
        """ Resets all of the flags in the curve to indicate that the current iteration of the Birkhoff algorithm is over.

        """

        # Reset the total movement of the curve to zero
        self.movement = 0.0

        # Reset the flag that indicates if a node has been moved
        self.nodes_moved = np.ones(self.number_of_nodes, dtype='int')

        # Reset the flags that indicate which nodes are movable back to the initial state
        self.node_movable = np.copy(self.default_initial_state)

        # Reset the count indicating the total number of distinct nodes that have been moved back to zero
        self.number_of_distinct_nodes_moved = 0

    def set_node_position(self, node_number, new_position):
        """ Update the position of the node at node_number to new_position. This processes the logic for releasing
        neighbouring nodes for further computation.

        Arguments:
            node_number (int): The node number of the node whose position is to be updated.
            new_position (numpy.array): The new position of the node.

        """

        # Arithmetic to measure the total movement of a curve
        self.movement += float(1/float(self.number_of_nodes)) * \
            np.linalg.norm(np.subtract(new_position, self.points[node_number]), ord=np.inf)

        # Update position of node with new position assumes new_position is float64 numpy array
        self.points[node_number] = new_position

        # Arithmetic for determining if an existing nodes neighbours have been moved
        self.node_movable[node_number-1] += 1
        self.node_movable[node_number+1] += 1
        self.node_movable[0] = 0
        self.node_movable[-1] = 0
        if node_number == 2:
            self.node_movable[1] += 1
            self.node_movable[0] = 0
        if node_number == self.number_of_nodes-3:
            self.node_movable[-2] += 1
            self.node_movable[-1] = 0

        # Arithmetic to keep track of how many nodes have been moved distinctly
        self.nodes_moved[node_number] = 0
        self.number_of_distinct_nodes_moved = sum(self.nodes_moved)

    def next(self):
        """ Determine next movable node, given existing information about previously distributed nodes. Used to ensure
            the curve object is Python iterable..

        Returns:
            int: The node number of the next movable node. If no such node exists then it returns None.

        """

        # This will get quicker with NumPy 2.0.0 - https://github.com/numpy/numpy/issues/2269
        try:
            # Determine the node number of the next movable node in the next_movable_node array. This particular test
            # attempts to find a node that has not been previously moved.
            next_movable_node = np.where(np.multiply(self.node_movable, self.nodes_moved) > 1)[0][0]

            # If there is no node available that hasn't been previously moved then...
            if next_movable_node is None:

                # Simply find a node that is movable. This time we include previously moved nodes in our search
                next_movable_node = np.where(self.node_movable > 1)[0][0]

            # If we still couldn't find a node...
            if next_movable_node is None:

                # Raise a StopIteration
                raise StopIteration

            # Otherwise...
            else:

                # Mark the node as no longer movable to prevent it being re-issued.
                self.node_movable[next_movable_node] = 0

                # Return the node number
                return next_movable_node

        except IndexError:

            # Raise a StopIteration
            raise StopIteration

    def get_points(self):
        """ Accessor method for the points attribute.

        Returns:
            numpy.array: An array containing all of the points of the curve.

        """
        return self.points

    def all_nodes_moved(self):
        """ This method determines whether every node in the global curve has been tested for length reduction.

        Returns:
          bool: True if all of the nodes have been tested, False otherwise.

        """
        if self.number_of_distinct_nodes_moved == 2:
            return True
        else:
            return False

    def __iter__(self):
        """ This special method ensures a curve object is iterable.

        Returns:
          self

        """
        return self

