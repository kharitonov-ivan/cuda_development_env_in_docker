# cuda_development_env_in_docker
docker image for cuda development

# How run
```
docker run -p 8080:8080 --gpus '"device=5"' neer201/ml_workspace_gl:latest
```

# Structure
1) image based on an awesome project [ml-workspace](https://github.com/ml-tooling/ml-workspace), because it has VSC and novnc installation already.
2) we should add nvidia graphic driver ([source](https://gitlab.com/nvidia/container-images/driver/-/tree/master/ubuntu18.04))
3) and install virtualGL
4) also we have to add cuda toolkit 


