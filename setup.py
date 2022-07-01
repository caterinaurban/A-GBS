from setuptools import setup, find_packages

config = {
    'name': 'A-GBS',
    'version': '1.0',
    'author': 'Caterina Urban',
    'author_email': 'caterina.urban@gmail.com',
    'description': 'Abstract Geometric Boundary Search',
    'url': '',
    'license': 'MPL-2.0',
    'packages': find_packages('src'),
    'package_dir': {'': 'src'},
    'entry_points': {
             'console_scripts': [
                 'a-gbs = agbs.main:main',
                 ]
             },
    'install_requires': [
        'apronpy'
    ],
    'scripts': [],
}

setup(**config)
