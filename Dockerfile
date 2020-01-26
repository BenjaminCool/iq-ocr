FROM ubuntu:latest

RUN apt-get update && apt-get install -y tesseract-ocr python3 python3-pip openjdk-8-jdk-headless zip vim curl git cmake gcc g++ pkg-config sudo
RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN mkdir -p /opt/ocr/build

RUN cd /opt/ocr && git clone https://github.com/opencv/opencv.git
RUN cd /opt/ocr && git clone https://github.com/opencv/opencv_contrib.git
RUN cd /opt/ocr && git clone https://github.com/opencv/dldt.git

RUN pip3 install -U pip
RUN pip3 install pytesseract cherrypy Flask opencv-python cython numpy pypillowfight


RUN cd /opt/ocr/build && cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    -D PYTHON_EXECUTABLE=/usr/bin/python3 \
    -D OPENCV_GENERATE_PKGCONFIG=YES \
    -D BUILD_opencv_dnn=YES \
    -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
    -D WITH_IPP=ON ../opencv/ && make -j4 && make install

RUN cd /opt/ocr/dldt/inference-engine/ && mkdir build && \
    bash install_dependencies.sh
RUN cd /opt/ocr/dldt/inference-engine/ && \
    git submodule init && \
    git submodule update --recursive
RUN cd /opt/ocr/dldt/inference-engine/build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_MKL_DNN=ON \
    -DENABLE_CLDNN=ON \
    -DPYTHON_EXECUTABLE=`which python3.6` \
    -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so \
    -DPYTHON_INCLUDE_DIR=/usr/include/python3.6 .. 
RUN cd /opt/ocr/dldt/inference-engine/build && make -j$(nproc --all)

ENV InferenceEngine_DIR=/opt/ocr/dldt/inference-engine/build/

RUN cd /opt/ocr/ && git clone https://github.com/opencv/open_model_zoo.git
RUN cd /opt/ocr/open_model_zoo/demos/ && bash build_demos.sh

EXPOSE 8080

COPY src /opt/ocr/src
COPY test /opt/ocr/test
COPY models /opt/ocr/models

CMD ["python3", "/opt/ocr/src/app.py"]
