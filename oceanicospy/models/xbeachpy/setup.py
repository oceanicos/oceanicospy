from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'A handle python package to Xbeach simulations'
LONG_DESCRIPTION = 'This is the first release for Xbeach on 12/09/2024'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="xbeachpy", 
        version=VERSION,
        author="Franklin Ayala",
        author_email="<ffayalac@unal.edu.co>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)