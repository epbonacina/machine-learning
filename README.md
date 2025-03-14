This is just a repository that contains a bunch of little Machine Learning projects.

## How to use it on Linux

1. Create a virtual environment for your Python dependencies  
`python -m venv .venv`
2. Activate your new virtual environment  
`. .venv/bin/activate`
3. Install all required dependencies  
`make install-dependencies`
4. Execute a project  
`make run ALGO=<project_name>`, where `<project_name>` is the name of any folder that contain
an implementation. For example, `make run ALGO=knn`.
