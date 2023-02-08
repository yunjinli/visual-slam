#!/usr/bin/env bash

set -e

if [[ "$OSTYPE" == "darwin"* ]]; then

    brew install \
         boost \
         opencv \
         cmake \
         pkgconfig \
         clang-format \
         tbb \
         glew \
         ccache \
         glog

else

    # should work for Ubuntu 18.04 and 20.04
    sudo apt-get install \
         build-essential \
         gdb \
         cmake \
         git \
         libtbb-dev \
         libeigen3-dev \
         libglew-dev \
         ccache \
         libjpeg-dev \
         libpng-dev \
         openssh-client \
         liblz4-dev \
         libbz2-dev \
         libboost-regex-dev \
         libboost-filesystem-dev \
         libboost-date-time-dev \
         libboost-program-options-dev \
         libopencv-dev \
         libpython3-dev \
         gfortran \
         libc++-dev \
         libgoogle-glog-dev \
         libatlas-base-dev \
         libsuitesparse-dev \
         wget \
         clang-format

fi
