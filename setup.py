from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='gridcell',
    version='0.0',
    description='Analyze experimental data from grid cells',
    long_description=readme(),
    url='https://github.com/danielwe/gridcell',
    author='Daniel Wennberg',
    author_email='daniel.wennberg@gmail.com',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'shapely',
        'seaborn',
        'scikit-image',
    ],
)
