from setuptools import setup

setup(name='hgail',
      version='0.1',
      description='Generative Adversarial Imitation Learning',
      author='Blake Wulfe',
      author_email='wulfebw@stanford.edu',
      license='MIT',
      packages=['hgail'],
      zip_safe=False,
      install_requires=[
        'numpy',
        'rllab',
        'tensorflow',
        'gym',
        'h5py',
        'cached_property',
        'joblib',
      ])