from setuptools import setup

setup(name='hgail',
      version='0.1',
      description='Hierarchical Generative Adversarial Imitation Learning',
      author='Blake Wulfe',
      author_email='wulfe@adobe.com',
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
        'scikit-image'
      ])