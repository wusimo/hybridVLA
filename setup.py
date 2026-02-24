from setuptools import setup, find_packages

setup(
    name="hybrid-vla",
    version="0.1.0",
    description="HybridVLA: Vision-Language-Action model combining BitVLA, RynnBrain, and Qwen VLM",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1",
        "transformers>=4.40",
        "numpy",
        "Pillow",
        "torchvision",
    ],
    extras_require={
        "lora": ["peft>=0.10"],
        "data": ["tensorflow-datasets", "h5py"],
    },
)
