# Lab 1 — Docker: build, save, run

A concise guide to build a local image, save it to a tarball, and run a container.

## Prerequisites

- Docker installed and running
- Dockerfile present in the current directory

## Commands

Build the image (tag it as lab1:v1):

```bash
docker build -t lab1:v1 .
```

Save the image to a tar file:

```bash
docker save lab1:v1 > my_image.tar
```

Load the image from a tar file (on another host or after cleanup):

```bash
docker load < my_image.tar
```

Run the image (foreground):

```bash
docker run --name lab1_container lab1:v1
```

Run in detached mode and map ports (example: host 8080 → container 80):

```bash
docker run -d -p 8080:80 --name lab1_container lab1:v1
```

Useful verification commands:

```bash
docker images
docker ps -a
docker logs lab1_container
```

Notes:

- Add `-it` to `docker run` for interactive shells.
- Use unique tags for versioning (e.g., lab1:v1.0).
- Clean up unused images/containers with `docker system prune` (use cautiously).
