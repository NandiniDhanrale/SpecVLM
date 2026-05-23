"""Setup script for SpecVLM."""
from setuptools import setup, find_packages

setup(
    name="specvlm",
    version="0.1.0",
    description="Production-grade speculative decoding system for Vision-Language Models",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4.0",
        "transformers>=4.44.0",
        "vllm>=0.6.0",
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.30.0",
        "ray[default]>=2.34.0",
        "pillow>=10.4.0",
    ],
    extras_require={
        "dev": ["pytest>=8.3.0", "black>=24.8.0", "ruff>=0.6.0"],
        "benchmark": ["locust>=2.31.0", "matplotlib>=3.9.0", "seaborn>=0.13.0"],
        "monitoring": ["prometheus-client>=0.20.0", "pynvml>=11.5.0"],
    },
)
