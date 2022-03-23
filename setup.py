import setuptools
import os


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('data/viz')

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
  name = 'Thorreznou',         
  version = '0.1.14',    
  author = 'The Bridge Data Science Team 1121',                   
  author_email = 'Thorreznou@gmail.com',
  description = '''Thorreznou.''',   
  long_description=long_description,
  long_description_content_type="text/markdown",
  url = 'https://github.com/erfederuiz/thorreznou',   
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  packages=['thorreznou'],
  package_dir={'thorreznou': 'src/thorreznou'},
  package_data={'thorreznou': extra_files },
  install_requires=[
    'pandas',
    'numpy',    
    'sklearn',
    'seaborn',
    'matplotlib',
    'scipy',
    'Ipython',
    'wordcloud',
    'pillow',
    'imblearn',
    'opencv-python',
    'pathlib'
  ],
  python_requires=">=3.7",
  include_package_data=True,
)
