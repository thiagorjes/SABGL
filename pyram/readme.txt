Requirements:
gcc/g++
Python 2.7.9
OpenCV 2.4.9
CUDA (optional)

Installing Python Packages:
sudo apt-get install python-pip
pip install -r requirements.txt

Building instructions:
CC=/usr/local/bin/gcc-5 CXX=/usr/local/bin/g++-5 \
	cmake -DCMAKE_BUILD_TYPE=Release -G"Eclipse CDT4 - Unix Makefiles" \
	-D PYTHON_LIBRARY=/usr/local/Cellar/python/2.7.10_2/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib \
	-D PYTHON_INCLUDE_PATH=/usr/local/Cellar/python/2.7.10_2/Frameworks/Python.framework/Headers \
	../pyram

TO-DO list:
- Implement VG-RAM training phase using batches and copy each sample test to shared memory, 
	compute hamming distance and store the canditates to find the nearest neighbour in CPU in order to avoid using atomic instructions.
	This can be an alternative to current memory restrictions.