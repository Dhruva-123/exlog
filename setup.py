from setuptools import setup, find_packages

setup(
    name='exlog',
    version='0.1.0',
    author='Rahul Dhruva',
    author_email='rahuldhruva.k9@gmail.com',
    description='A one-line logging explainability tool for ML models',
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    install_requires = [
        "numpy","pandas","scikit-learn","shap","torch","tensorflow","xgboost","lightgbm"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires=">=3.8",
)
