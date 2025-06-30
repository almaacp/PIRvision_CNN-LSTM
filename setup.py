from setuptools import setup, find_packages

setup(
    name='pirvision',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'tensorflow',
        'imblearn',
        'ipywidgets'
    ],
    author='Alma Alya Cipta Putri',
    description='PIR sensor-based human activity classification pipeline',
    long_description='Pipeline klasifikasi aktivitas menggunakan sensor PIR dengan balancing dan deep learning model CNN-LSTM.',
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7'
)