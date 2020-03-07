# FROM ubuntu:18.04
FROM mltooling/ml-workspace-minimal

# # ---------------------------------------------------------------------------- #
# #                             nvidia graphic driver                            #
# # ---------------------------------------------------------------------------- #
# # We need to fix the gcc version to 7.3 as the current kernel
# # for Ubuntu18.04 is compiled with this version.
# RUN apt update && apt install -y --no-install-recommends \
#         cpp=4:7.3.0-3ubuntu2 \
#         cpp-7=7.3.0-16ubuntu3 \
#         g++=4:7.3.0-3ubuntu2 \
#         g++-7=7.3.0-16ubuntu3 \
#         gcc=4:7.3.0-3ubuntu2 \
#         gcc-7=7.3.0-16ubuntu3 \
#         gcc-7-base=7.3.0-16ubuntu3 \
#         libasan4=7.3.0-16ubuntu3 \
#         libcilkrts5=7.3.0-16ubuntu3 \
#         libgcc-7-dev=7.3.0-16ubuntu3 \
#         libstdc++-7-dev=7.3.0-16ubuntu3 \
#         libubsan0=7.3.0-16ubuntu3 && \
#       apt-mark hold cpp cpp-7 g++ g++-7 gcc gcc-7 gcc-7-base libasan4 \
#         libcilkrts5 libgcc-7-dev libstdc++-7-dev libubsan0

# RUN dpkg --add-architecture i386 && \
#     apt-get update && apt-get install -y --no-install-recommends \
#         apt-utils \
#         build-essential \
#         ca-certificates \
#         curl \
#         kmod \
#         libc6:i386 \
#         libelf-dev && \
#     rm -rf /var/lib/apt/lists/*

# RUN echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ bionic main" > /etc/apt/sources.list && \
#     echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ bionic-updates main" >> /etc/apt/sources.list && \
#     echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu/ bionic-security main" >> /etc/apt/sources.list && \
#     usermod -o -u 0 -g 0 _apt

# RUN curl -fsSL -o /usr/local/bin/donkey https://github.com/3XX0/donkey/releases/download/v1.1.0/donkey && \
#     curl -fsSL -o /usr/local/bin/extract-vmlinux https://raw.githubusercontent.com/torvalds/linux/master/scripts/extract-vmlinux && \
#     chmod +x /usr/local/bin/donkey /usr/local/bin/extract-vmlinux

# #ARG BASE_URL=http://us.download.nvidia.com/XFree86/Linux-x86_64
# ARG BASE_URL=https://us.download.nvidia.com/tesla
# ARG DRIVER_VERSION=418.40.04
# ENV DRIVER_VERSION=$DRIVER_VERSION

# # Install the userspace components and copy the kernel module sources.
# RUN cd /tmp && \
#     curl -fSsl -O $BASE_URL/$DRIVER_VERSION/NVIDIA-Linux-x86_64-$DRIVER_VERSION.run && \
#     sh NVIDIA-Linux-x86_64-$DRIVER_VERSION.run -x && \
#     cd NVIDIA-Linux-x86_64-$DRIVER_VERSION* && \
#     ./nvidia-installer --silent \
#                        --no-kernel-module \
#                        --install-compat32-libs \
#                        --no-nouveau-check \
#                        --no-nvidia-modprobe \
#                        --no-rpms \
#                        --no-backup \
#                        --no-check-for-alternate-installs \
#                        --no-libglx-indirect \
#                        --no-install-libglvnd \
#                        --x-prefix=/tmp/null \
#                        --x-module-path=/tmp/null \
#                        --x-library-path=/tmp/null \
#                        --x-sysconfig-path=/tmp/null \
#                        --no-glvnd-egl-client \
#                        --no-glvnd-glx-client && \
#     mkdir -p /usr/src/nvidia-$DRIVER_VERSION && \
#     mv LICENSE mkprecompiled kernel /usr/src/nvidia-$DRIVER_VERSION && \
#     sed '9,${/^\(kernel\|LICENSE\)/!d}' .manifest > /usr/src/nvidia-$DRIVER_VERSION/.manifest && \
#     rm -rf /tmp/*

# COPY nvidia-driver /usr/local/bin

# # WORKDIR /usr/src/nvidia-$DRIVER_VERSION

# # ARG PUBLIC_KEY=empty
# # COPY ${PUBLIC_KEY} kernel/pubkey.x509

# # ARG PRIVATE_KEY
# # ARG KERNEL_VERSION=generic,generic-hwe-18.04

# # # Compile the kernel modules and generate precompiled packages for use by the nvidia-installer.
# # RUN apt-get update && \
# #     for version in $(echo $KERNEL_VERSION | tr ',' ' '); do \
# #         nvidia-driver update -k $version -t builtin ${PRIVATE_KEY:+"-s ${PRIVATE_KEY}"}; \
# #     done && \
# #     rm -rf /var/lib/apt/lists/*

# # ENTRYPOINT ["nvidia-driver", "init"]

# # ---------------------------------------------------------------------------- #
# #                             end of nvidia driver                             #
# # ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                              begin cuda toolkit                              #
# ---------------------------------------------------------------------------- #

RUN apt-get update && apt-get install -y --no-install-recommends \
gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl && \
rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 10.0.130

ENV CUDA_PKG_VERSION 10-0=$CUDA_VERSION-1

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION \
cuda-compat-10-0 && \
ln -s cuda-10.0 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.0 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=410,driver<411"

# ---------------------------------------------------------------------------- #
#                              end of cuda toolkit                              #
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                             cuda toolkit runtime                             #
# ---------------------------------------------------------------------------- #

ENV NCCL_VERSION 2.4.2

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-libraries-$CUDA_PKG_VERSION \
        cuda-nvtx-$CUDA_PKG_VERSION \
        libnccl2=$NCCL_VERSION-1+cuda10.0 && \
    apt-mark hold libnccl2 && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------- #
#                          end of cuda toolkit runtime                         #
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                              cuda toolkit devel                              #
# ---------------------------------------------------------------------------- #

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-libraries-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-$CUDA_PKG_VERSION \
        # cuda-minimal-build-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        libnccl-dev=$NCCL_VERSION-1+cuda10.0 && \
    rm -rf /var/lib/apt/lists/*
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

# ---------------------------------------------------------------------------- #
#                           end of cuda toolkit devel                          #
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                            begin of virual gl base                           #
# ---------------------------------------------------------------------------- #
RUN dpkg --add-architecture i386 && \
    apt-get update && apt-get install -y --no-install-recommends \
        libxau6 libxau6:i386 \
        libxdmcp6 libxdmcp6:i386 \
        libxcb1 libxcb1:i386 \
        libxext6 libxext6:i386 \
        libx11-6 libx11-6:i386 && \
    rm -rf /var/lib/apt/lists/*
# ---------------------------------------------------------------------------- #
#                            end of  virtual gl base                           #
# ---------------------------------------------------------------------------- #