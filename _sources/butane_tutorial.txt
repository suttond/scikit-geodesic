Tutorial
========

Butane Simulation
-----------------
In this tutorial we will use the GeometricMD package to compute a transition path for a butane molecule. The files for
the simulation, along with the scripts, are available in the *example* directory of the GeometricMD package.

Single Process
+++++++++++++++

.. code-block:: python

   from geometricmd.curve_shorten import compute_trajectory
   from geometricmd.geometry import Curve

   # Import ASE read function if getting molecule data from compatible file
   from ase.io import read

   # This example uses the EMT calculator for simplicity
   from ase.calculators.emt import EMT

   # Read in the molecule data for the initial point
   start_point = read('x0.xyz')

   # In order to compute the potential energy a calculator must be attached to the start atoms object
   start_point.set_calculator(EMT())

   # Read in the molecule data for the final point
   end_point = read('xN.xyz')

   # Create a GeometricMD curve object to represent the trajectory.
   traj = Curve(start_point, end_point, 12, 1E+03)

   # Perform the molecular simulation.
   compute_trajectory(traj, 10, 1E+03, 0.01, 'Butane', {'processes': 1})


Multiple Processes
++++++++++++++++++++

.. code-block:: python

   from geometricmd.curve_shorten import compute_trajectory
   from geometricmd.geometry import Curve
   from multiprocessing import cpu_count

   # Import ASE read function if getting molecule data from compatible file
   from ase.io import read

   # This example uses the EMT calculator for simplicity
   from ase.calculators.emt import EMT

   # Read in the molecule data for the initial point
   start_point = read('x0.xyz')

   # In order to compute the potential energy a calculator must be attached to the start atoms object
   start_point.set_calculator(EMT())

   # Read in the molecule data for the final point
   end_point = read('xN.xyz')

   # Create a GeometricMD curve object to represent the trajectory.
   traj = Curve(start_point, end_point, 12, 1E+03)

   # Perform the molecular simulation.
   compute_trajectory(traj, 10, 1E+03, 0.01, 'Butane', {'processes': (cpu_count()-1)})
