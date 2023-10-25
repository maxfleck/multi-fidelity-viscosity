from setuptools import setup, find_packages
setup(
    name="multi_fidelity_viscosity",
    version="0.0",
    packages=find_packages(),
    #entry_points={
    #    'console_scripts': [
    #        'mc_post = mc_post.main:main',
    #    ],
    #},

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    # install_requires=['docutils>=0.3'],
    #package_data={
    #    # If any package contains *.txt or *.rst files, include them:
    #    '': ['*.toml'],
    #},

    # metadata for upload to PyPI
    # author="Me",
    # author_email="me@example.com",
    # description="This is an Example Package",
    # license="PSF",
    # keywords="hello world example examples",
    # url="http://example.com/HelloWorld/",   # project home page, if any
    # project_urls={
    #     "Bug Tracker": "https://bugs.example.com/HelloWorld/",
    #     "Documentation": "https://docs.example.com/HelloWorld/",
    #     "Source Code": "https://code.example.com/HelloWorld/",
    # }

    # could also include long_description, download_url, classifiers, etc.
)
