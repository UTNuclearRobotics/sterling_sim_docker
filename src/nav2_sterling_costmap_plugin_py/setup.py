from setuptools import find_packages, setup

package_name = 'nav2_sterling_costmap_plugin_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['sterling_layer.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nchan',
    maintainer_email='nick.chan@utexas.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
