"""
Fraud Detection Workshop - SageMaker Edition Setup
===================================================

Author: Workshop Team
Date: 2026
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fraud-detection-sagemaker",
    version="2.0.0",
    author="Workshop Team",
    author_email="workshop@example.com",
    description="Fraud Detection with XGBoost on AWS SageMaker",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/fraud-detection-sagemaker",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
