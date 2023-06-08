from setuptools import find_packages, setup

setup(
    name="bros",
    version="1.0.0",
    description="",
    author="Jinho Park",
    author_email="jinhopark8345@gmail.com",
    url="git@github.com:jinhopark8345/py3_code_snippets.git",
    packages=find_packages(exclude=('tests*', 'testing*')),
    entry_points={
        'console_scripts': [
            'run-script-example = module_name.file_name:function_name',
        ]
    }
)
