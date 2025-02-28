from setuptools import setup, find_packages

setup(
    name="sterling",  # Name of your package
    version="0.1.0",   # Version number
    packages=find_packages(),  # Automatically find packages in your project
    install_requires=[  # List your dependencies here
        "opencv-python",
        "numpy",
        "joblib",
        "scikit-learn",
        "torch",
        "pyyaml"
    ],
    extras_require={  # Optional dependencies
        # "dev": ["pytest", "black"],
    },
    entry_points={  # Define command-line scripts
        "console_scripts": [
            # "my_script=my_project.module:main",
        ],
    },
    include_package_data=True,  # Include non-Python files specified in MANIFEST.in
)