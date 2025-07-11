from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'A handle package for most of the programming tools in the research group OCEANICOS'
LONG_DESCRIPTION = 'This package looks for having at hand most of the scripts OCEANICOS uses for their ocean research/consultancy'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="oceanicospy", 
        version=VERSION,
        author="Oceanicos developer team",
        author_email="<oceanicos_med@unal.edu.co>",
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