import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
  name = 'Thorreznou',         
  version = '0.1.4',    
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
  package_dir={"": "src"},
  packages=setuptools.find_packages(where="src"),
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
  python_requires=">=3.6",
)
