from setuptools import setup, find_packages
from setuptools.command.install import install

setup(name='langpractice',
      packages=find_packages(),
      version="0.1.0",
      description='A project to determine how language improves some cognitive tasks',
      author='Satchel Grant',
      author_email='grantsrb@stanford.edu',
      url='https://github.com/grantsrb/langpractice.git',
      install_requires= ["numpy",
                         "torch",
                         "tqdm"],
      py_modules=['supervised_gym'],
      long_description='''
            This project compares pretrained models on counting tasks
            to determine the means by which language improves an agents'
            ability to complete a numerical task.
          ''',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X :: Ubuntu',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      )
