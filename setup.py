'''Setup script for bdds.'''


from setuptools import setup


if __name__ == '__main__':
    setup(name='bdds',
          version='0.1',
          description='API for brain decoding datasets',
          author='Shuntaro Aoki',
          author_email='kamitanilab.contact@gmail.com',
          url='https://github.com/KamitaniLab/brain-decoding-datasets',
          license='MIT',
          packages=['bdds'],
          install_requires=['numpy', 'scipy', 'h5py'])
