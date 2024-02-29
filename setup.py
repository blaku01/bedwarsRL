from setuptools import find_packages, setup

setup(
    name="bedwarsRL",
    version="0.0.1",
    description="bedwars AI using Reinforcement Learning",
    author="Gabriel Blaczek, Przemysław Tomala",
    author_email="gabrielblaczek@gmail.com, timuslala@gmail.com",
    url="https://github.com/blaku01/bedwarsRL",
    install_requires=[],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    # entry_points={
    #     "console_scripts": [
    #         "command = src.xyz:main",
    #     ]
    # },
)
