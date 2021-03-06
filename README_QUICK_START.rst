FogLAMP "Image classifier" South plugin

Before using this plugin:

RPi camera should be enabled:

.. code-block:: console

  $ sudo raspi-config   # Enable "RPi Camera" in "Interfacing options" -> Camera

On each reboot, camera driver has to be loaded into kernel before using this plugin:

.. code-block:: console

  $ sudo modprobe bcm2835-v4l2

If USB camera is used, corresponding drivers must be loaded into kernel.

Also OpenCV libraries are installed in /usr/local/lib, so in the shell where FogLAMP
is started, before starting it, update the LD_LIBRARY_PATH:

.. code-block:: console

  $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

Later on, LD_LIBRARY_PATH would be updated in ${FOGLAMP_ROOT}/scripts/foglamp itself.


Build
----

Building of this plugin is supported only on Raspberry Pi.

Building this plugin needs building of OpenCV and Tensorflow lite libraries and
that takes lot of space & time. It is recommended to use atleast 16 GB SD card.

For quick start, the downloaded, configured and compiled packages are available
for download directly from google drive at link given below. Hopefully they can
be used to speed-up the setup. But it depends on Raspbian release version, version
of various apt packages & system libraries, update status of OS, RPi version, etc.
So this quick start guide is made available on best effort basis only. If it doesn't
work out, the complete process might have to be followed as listed in README.rst.

Steps to compile & install OpenCV for RPi:

.. code-block:: console

  $ sudo apt install -y build-essential cmake pkg-config
  $ sudo apt install -y libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
  $ sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
  $ sudo apt install -y libxvidcore-dev libx264-dev
  $ sudo apt install -y libgtk2.0-dev libgtk-3-dev
  $ sudo apt install -y libatlas-base-dev gfortran
  
  $ sudo pip3 install numpy
  
  $ cd
  # Download tf.tar.gz from https://drive.google.com/file/d/1ailG5blf6_PRKw55klMSrUd6PMwkvM4j/view
  $ tar xf tf.tar.gz
  
  $ sudo sed -i 's/CONF_SWAPSIZE=100/CONF_SWAPSIZE=1024/g' /etc/dphys-swapfile
  $ sudo /etc/init.d/dphys-swapfile stop
  $ sudo /etc/init.d/dphys-swapfile start

  $ cd ~/tf/opencv-3.4.5/
  $ mkdir -p build
  $ cd build
  $ make -j4
  $ sudo make install

  $ sudo sed -i 's/CONF_SWAPSIZE=1024/CONF_SWAPSIZE=100/g' /etc/dphys-swapfile
  $ sudo /etc/init.d/dphys-swapfile stop
  $ sudo /etc/init.d/dphys-swapfile start


Tensorflow lite header files and static library are also required:

.. code-block:: console

  $ cd ~/tf/tensorflow
  $ ./tensorflow/lite/tools/make/build_rpi_lib.sh


To build FogLAMP Image classifier South plugin:

.. code-block:: console

  $ export TF_ROOT=/home/pi/tf/tensorflow    # path where tensorflow github repo is cloned
  $ mkdir build; cd build; cmake -DFOGLAMP_INSTALL=$FOGLAMP_ROOT ..; make && make install

- By default the FogLAMP develop package header files and libraries
  are expected to be located in /usr/include/foglamp and /usr/lib/foglamp
- If **FOGLAMP_ROOT** env var is set and no -D options are set,
  the header files and libraries paths are pulled from the ones under the
  FOGLAMP_ROOT directory.
  Please note that you must first run 'make' in the FOGLAMP_ROOT directory.

You may also pass one or more of the following options to cmake to override 
this default behaviour:

- **FOGLAMP_SRC** sets the path of a FogLAMP source tree
- **FOGLAMP_INCLUDE** sets the path to FogLAMP header files
- **FOGLAMP_LIB sets** the path to FogLAMP libraries
- **FOGLAMP_INSTALL** sets the installation path of Image classifier plugin

NOTE:
 - The **FOGLAMP_INCLUDE** option should point to a location where all the FogLAMP 
   header files have been installed in a single directory.
 - The **FOGLAMP_LIB** option should point to a location where all the FogLAMP
   libraries have been installed in a single directory.
 - 'make install' target is defined only when **FOGLAMP_INSTALL** is set

Examples:

- no options

  $ cmake ..

- no options and FOGLAMP_ROOT set

  $ export FOGLAMP_ROOT=/some_foglamp_setup

  $ cmake ..

- set FOGLAMP_SRC

  $ cmake -DFOGLAMP_SRC=/home/source/develop/FogLAMP  ..

- set FOGLAMP_INCLUDE

  $ cmake -DFOGLAMP_INCLUDE=/dev-package/include ..
- set FOGLAMP_LIB

  $ cmake -DFOGLAMP_LIB=/home/dev/package/lib ..
- set FOGLAMP_INSTALL

  $ cmake -DFOGLAMP_INSTALL=/home/source/develop/FogLAMP ..

  $ cmake -DFOGLAMP_INSTALL=/usr/local/foglamp ..

******************************
Packaging for 'Image classifier' south
******************************

This repo contains the scripts used to create a foglamp-south-image-classifier Debian package.

The make_deb script
===================

Run the make_deb command:

.. code-block:: console

  $ ./make_deb help
  make_deb [help|clean|cleanall]
  This script is used to create the Debian package of FoglAMP C++ 'Image classifier' south plugin
  Arguments:
   help     - Display this help text
   clean    - Remove all the old versions saved in format .XXXX
   cleanall - Remove all the versions, including the last one
  $

Building a Package
==================

Finally, run the ``make_deb`` command:

.. code-block:: console

   $ ./make_deb
   The package root directory is   : /home/ubuntu/source/foglamp-south-image-classifier
   The FogLAMP required version    : >=1.4
   The package will be built in    : /home/ubuntu/source/foglamp-south-image-classifier/packages/build
   The architecture is set as      : x86_64
   The package name is             : foglamp-south-image-classifier-1.0.0-x86_64

   Populating the package and updating version file...Done.
   Building the new package...
   dpkg-deb: building package 'foglamp-south-modbusc' in 'foglamp-south-image-classifier-1.0.0-x86_64.deb'.
   Building Complete.
   $

Cleaning the Package Folder
===========================

Use the ``clean`` option to remove all the old packages and the files used to make the package.

Use the ``cleanall`` option to remove all the packages and the files used to make the package.
