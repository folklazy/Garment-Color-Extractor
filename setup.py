from setuptools import setup, find_packages

setup(
    name="garment-color-extractor",
    version="1.0.0",
    description="Lightweight, Shadow-Aware Garment Color Extraction Engine (M5) for Surveillance and Human Parsing",
    author="folklazy",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python-headless>=4.5.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.20.0",
            "pydantic>=2.0.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
    ],
)

