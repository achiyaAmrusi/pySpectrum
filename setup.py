from setuptools import setup

setup(
    name='pySpectrum',
    version='1.0',
    packages=['pyspectrum', 'pyspectrum.peak', 'pyspectrum.spectrum', 'pyspectrum.calibration',
              'pyspectrum.peak_fitting', 'pyspectrum.detector_parser', 'pyspectrum.peak_identification'],
    url='https://github.com/achiyaAmrusi/pySpectrum',
    license='MIT license',
    author='Achiya Yosef Amrusi',
    author_email='ahia.amrosi@mail.huji.ac.il',
    description='spectrum and peak analysis tools'
)
