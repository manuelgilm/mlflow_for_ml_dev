from setuptools import setup

setup(
    name='package',
    version='0.1',
    description='A useful package',
    author="Manuel Gil",
    author_email="manuelgilsitio@gmail.com",
    packages=['package.feature', 'package.ml_training', "package.utils"],
    # install_requires=['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'mlflow==2.3.1']
)