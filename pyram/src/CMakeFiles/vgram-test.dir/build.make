# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/data/CAVALCANTE/SABGL/pyram

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/data/CAVALCANTE/SABGL/pyram

# Include any dependencies generated for this target.
include src/CMakeFiles/vgram-test.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/vgram-test.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/vgram-test.dir/flags.make

src/CMakeFiles/vgram-test.dir/vgram_main.cpp.o: src/CMakeFiles/vgram-test.dir/flags.make
src/CMakeFiles/vgram-test.dir/vgram_main.cpp.o: src/vgram_main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/data/CAVALCANTE/SABGL/pyram/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/vgram-test.dir/vgram_main.cpp.o"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vgram-test.dir/vgram_main.cpp.o -c /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_main.cpp

src/CMakeFiles/vgram-test.dir/vgram_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vgram-test.dir/vgram_main.cpp.i"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_main.cpp > CMakeFiles/vgram-test.dir/vgram_main.cpp.i

src/CMakeFiles/vgram-test.dir/vgram_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vgram-test.dir/vgram_main.cpp.s"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_main.cpp -o CMakeFiles/vgram-test.dir/vgram_main.cpp.s

src/CMakeFiles/vgram-test.dir/vgram_main.cpp.o.requires:

.PHONY : src/CMakeFiles/vgram-test.dir/vgram_main.cpp.o.requires

src/CMakeFiles/vgram-test.dir/vgram_main.cpp.o.provides: src/CMakeFiles/vgram-test.dir/vgram_main.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/vgram-test.dir/build.make src/CMakeFiles/vgram-test.dir/vgram_main.cpp.o.provides.build
.PHONY : src/CMakeFiles/vgram-test.dir/vgram_main.cpp.o.provides

src/CMakeFiles/vgram-test.dir/vgram_main.cpp.o.provides.build: src/CMakeFiles/vgram-test.dir/vgram_main.cpp.o


src/CMakeFiles/vgram-test.dir/vgram_files.cpp.o: src/CMakeFiles/vgram-test.dir/flags.make
src/CMakeFiles/vgram-test.dir/vgram_files.cpp.o: src/vgram_files.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/data/CAVALCANTE/SABGL/pyram/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/vgram-test.dir/vgram_files.cpp.o"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vgram-test.dir/vgram_files.cpp.o -c /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_files.cpp

src/CMakeFiles/vgram-test.dir/vgram_files.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vgram-test.dir/vgram_files.cpp.i"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_files.cpp > CMakeFiles/vgram-test.dir/vgram_files.cpp.i

src/CMakeFiles/vgram-test.dir/vgram_files.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vgram-test.dir/vgram_files.cpp.s"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_files.cpp -o CMakeFiles/vgram-test.dir/vgram_files.cpp.s

src/CMakeFiles/vgram-test.dir/vgram_files.cpp.o.requires:

.PHONY : src/CMakeFiles/vgram-test.dir/vgram_files.cpp.o.requires

src/CMakeFiles/vgram-test.dir/vgram_files.cpp.o.provides: src/CMakeFiles/vgram-test.dir/vgram_files.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/vgram-test.dir/build.make src/CMakeFiles/vgram-test.dir/vgram_files.cpp.o.provides.build
.PHONY : src/CMakeFiles/vgram-test.dir/vgram_files.cpp.o.provides

src/CMakeFiles/vgram-test.dir/vgram_files.cpp.o.provides.build: src/CMakeFiles/vgram-test.dir/vgram_files.cpp.o


# Object files for target vgram-test
vgram__test_OBJECTS = \
"CMakeFiles/vgram-test.dir/vgram_main.cpp.o" \
"CMakeFiles/vgram-test.dir/vgram_files.cpp.o"

# External object files for target vgram-test
vgram__test_EXTERNAL_OBJECTS =

src/vgram-test: src/CMakeFiles/vgram-test.dir/vgram_main.cpp.o
src/vgram-test: src/CMakeFiles/vgram-test.dir/vgram_files.cpp.o
src/vgram-test: src/CMakeFiles/vgram-test.dir/build.make
src/vgram-test: src/vgram.so
src/vgram-test: src/CMakeFiles/vgram-test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/data/CAVALCANTE/SABGL/pyram/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable vgram-test"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vgram-test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/vgram-test.dir/build: src/vgram-test

.PHONY : src/CMakeFiles/vgram-test.dir/build

src/CMakeFiles/vgram-test.dir/requires: src/CMakeFiles/vgram-test.dir/vgram_main.cpp.o.requires
src/CMakeFiles/vgram-test.dir/requires: src/CMakeFiles/vgram-test.dir/vgram_files.cpp.o.requires

.PHONY : src/CMakeFiles/vgram-test.dir/requires

src/CMakeFiles/vgram-test.dir/clean:
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && $(CMAKE_COMMAND) -P CMakeFiles/vgram-test.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/vgram-test.dir/clean

src/CMakeFiles/vgram-test.dir/depend:
	cd /mnt/data/CAVALCANTE/SABGL/pyram && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/data/CAVALCANTE/SABGL/pyram /mnt/data/CAVALCANTE/SABGL/pyram/src /mnt/data/CAVALCANTE/SABGL/pyram /mnt/data/CAVALCANTE/SABGL/pyram/src /mnt/data/CAVALCANTE/SABGL/pyram/src/CMakeFiles/vgram-test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/vgram-test.dir/depend

