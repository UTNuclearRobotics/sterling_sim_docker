import glob
import os

from setuptools import find_packages, setup

package_name = "sterling"
lib_files = [f for f in glob.glob("sterling/lib/sterling/**/*", recursive=True) if os.path.isfile(f)]


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # ("share/" + package_name, ["config/config.yaml"]),
        # ("share/" + package_name, ["launch/sterling_sim.launch.py"]),
        # ("lib/" + package_name, lib_files),
    ],
    install_requires=["setuptools", "opencv-python", "numpy", "joblib", "scikit-learn", "torch", "pyyaml"],
    zip_safe=True,
    maintainer="nchan",
    maintainer_email="nick.chan@utexas.edu",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "global_costmap = sterling.nodes.global_costmap_builder:main",
        ],
    },
)
