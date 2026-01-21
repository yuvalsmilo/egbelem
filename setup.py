
from setuptools import setup 
  
setup( 
    name='egbelem', 
    version='0.1', 
    description='EgbeLem: Landscape Evolution Model implementing EGBE theory', 
    author='Greg Tucker', 
    author_email='gtucker@colorado.edu', 
    packages=['egbelem'], 
    install_requires=[ 
        'numpy', 
        'landlab', 
    ], 
)
