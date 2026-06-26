from setuptools import find_packages, setup


setup(
    name="realsense-vlm-object-locator",
    version="0.1.0",
    description=(
        "Locate a named RGB object in RealSense D435i camera coordinates using "
        "OpenRouter VLM bounding boxes and aligned depth."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "numpy>=1.23",
        "opencv-python>=4.8",
        "pyrealsense2>=2.54",
        "PyYAML>=6.0",
        "python-dotenv>=1.0",
        "requests>=2.31",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0",
        ],
        "grounded-sam": [
            "accelerate>=0.26",
            "Pillow>=10.0",
            "torch>=2.1",
            "transformers>=4.41",
        ],
    },
    entry_points={
        "console_scripts": [
            "object-locator=object_locator.cli:main",
        ],
    },
)
