from setuptools import setup, find_packages

if __name__ == '__main__':
    setup(name='postprocess',
          packages=find_packages(),
          install_requires=['numpy', 'matplotlib'],  # And paraview!
          maintainer='Pierre Seize',
          maintainer_email='pierre.seize@gmail.com'
          )
