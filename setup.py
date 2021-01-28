from setuptools import setup


def get_version(filename):
    import ast

    version = None
    with open(filename) as f:
        for line in f:
            if line.startswith("__version__"):
                version = ast.parse(line).body[0].value.s
                break
        else:
            raise ValueError(f"No version found in {filename}.")
    if version is None:
        raise ValueError(filename)
    return version


install_requires = [
    "numpy",
    "scipy",
    "suds-jurko",
    "casadi==3.5.1",
    "matplotlib",
]

module = "gokartmpcc"
package = "gokartmpcc"
src = "src"

version = get_version(filename=f"src/{module}/__init__.py")

setup(
    name=package,
    package_dir={"": src},
    packages=[module],
    version=version,
    zip_safe=False,
    #entry_points={"console_scripts": ["dg-demo = games_zoo:dg_demo", ]},
    install_requires=install_requires,
)