import subprocess

packages = ['numpy','pandas','scipy']
for package in packages:
    subprocess.run(f'C:\\Users\\luckmanu\\Tools\\Blender\\3.5\\python\\bin\\python.exe -m pip install {package}',
                   shell=True)
