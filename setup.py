from distutils.core import setup

setup(
    name='scikit-geodesic',
    version='1.1',
    packages=['geodesic'],
    url='http://suttond.github.io/scikit-geodesic',
    license='LGPL 3.0',
    author='Daniel C. Sutton',
    author_email='sutton.c.daniel@gmail.com',
    description='A SciPy tool for computing geodesics in an isotropic Riemannian manifold of arbitrary dimension. It implements the Birkhoff curve shortening algorithm for finding global geodesics.'
)
