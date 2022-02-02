import os
from os import path
from setuptools import setup, find_packages
import sys
import versioneer
import warnings

# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
min_version = (3, 7)
if sys.version_info < min_version:
    error = """
watershed-workflow does not support Python {0}.{1}.
Python {2}.{3} and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
""".format(*(sys.version_info[:2] + min_version))
    sys.exit(error)

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as readme_file:
    readme = readme_file.read()


# NOTE: the requirements.txt listed here is VERY incomplete.  This is
# intentional -- most of the pip-based GIS packages don't correctly
# deal with dependencies on GIS libraries.  Instead, the majority of
# packages here MUST be installed via Anaconda or done manually.
# However, there are a few requirements that cannot be provided via
# conda, but CAN be provided via pip, so we get those here...
with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]


setup(
    name='watershed-workflow',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Workflow tool that synthesizes datasets for use in integrated hydrologic models.",
    long_description=readme,
    author="Ethan Coon",
    author_email='etcoon@gmail.com',
    url='https://github.com/ecoon/watershed-workflow',
    python_requires='>={}'.format('.'.join(str(n) for n in min_version)),
    packages=find_packages(exclude=['docs', 'tests']),
    entry_points={
        'console_scripts': [
            # 'command = some.module:some_function',
        ],
    },
    include_package_data=True,
    package_data={
        'watershed_workflow': [
            # When adding files here, remember to update MANIFEST.in as well,
            # or else they will not be included in the distribution on PyPI!
            # 'path/to/data_file',
        ]
    },
    install_requires=requirements,
    license="BSD (3-clause)",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
)

# copy over the rc file as a template
try:
    rcfile = path.join(os.environ['HOME'], '.watershed_worklowrc')
    if not path.exists(rcfile):
        os.copyfile(path.join(here, 'watershed_workflowrc'), rcfile)
except:
    warnings.warn('Warning: cannot figure out where to put .watershed_workflowrc.  Manually copy this to your home directory.')

