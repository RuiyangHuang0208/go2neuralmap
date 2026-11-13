from setuptools import find_packages, setup


setup(
    name="go2_neural_map",
    version="0.1.0",
    description="Terrain-aware quadruped navigation with proprioceptive neural maps",
    packages=find_packages(include=["utils", "utils.*"]),
    include_package_data=True,
    install_requires=[
        "numpy",
        "torch",
        "PyYAML",
        "trimesh",
    ],
)
