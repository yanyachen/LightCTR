import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='lightctr',
    version='0.1.0',
    author='YaChen Yan',
    author_email='yanyachen21@gmail.com',
    description='Deep Learning Models for Click-Through Rate Prediction',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yanyachen/lightctr',
    packages=setuptools.find_packages(),
    install_requires=[],
    extras_require={
        'cpu': ['tensorflow>=1.4.0,!=1.7.*,!=1.8.*'],
        'gpu': ['tensorflow-gpu>=1.4.0,!=1.7.*,!=1.8.*'],
    },
    entry_points={},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    license='Apache-2.0'
)
