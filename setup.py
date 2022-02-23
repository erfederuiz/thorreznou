import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
  name = 'tbd',         
  version = '0.1',    
  author = 'The Bridge Data Science Team 1121',                   
  author_email = 'tbd@gmail.com',
  description = '''tbd.''',   
  long_description=long_description,
  long_description_content_type="text/markdown",
  url = 'https://github.com/mabatalla/DSFT1121/tbd',   
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  package_dir={"": "src"},
  packages=setuptools.find_packages(where="src"),
  install_requires=[
    'pandas',
    'numpy',    
    'scikit-learn'   
  ],
  python_requires=">=3.6",
)
