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
include src/CMakeFiles/vgram.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/vgram.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/vgram.dir/flags.make

src/CMakeFiles/vgram.dir/vgram_base.cpp.o: src/CMakeFiles/vgram.dir/flags.make
src/CMakeFiles/vgram.dir/vgram_base.cpp.o: src/vgram_base.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/data/CAVALCANTE/SABGL/pyram/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/vgram.dir/vgram_base.cpp.o"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vgram.dir/vgram_base.cpp.o -c /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_base.cpp

src/CMakeFiles/vgram.dir/vgram_base.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vgram.dir/vgram_base.cpp.i"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_base.cpp > CMakeFiles/vgram.dir/vgram_base.cpp.i

src/CMakeFiles/vgram.dir/vgram_base.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vgram.dir/vgram_base.cpp.s"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_base.cpp -o CMakeFiles/vgram.dir/vgram_base.cpp.s

src/CMakeFiles/vgram.dir/vgram_base.cpp.o.requires:

.PHONY : src/CMakeFiles/vgram.dir/vgram_base.cpp.o.requires

src/CMakeFiles/vgram.dir/vgram_base.cpp.o.provides: src/CMakeFiles/vgram.dir/vgram_base.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/vgram.dir/build.make src/CMakeFiles/vgram.dir/vgram_base.cpp.o.provides.build
.PHONY : src/CMakeFiles/vgram.dir/vgram_base.cpp.o.provides

src/CMakeFiles/vgram.dir/vgram_base.cpp.o.provides.build: src/CMakeFiles/vgram.dir/vgram_base.cpp.o


src/CMakeFiles/vgram.dir/vgram_train.cpp.o: src/CMakeFiles/vgram.dir/flags.make
src/CMakeFiles/vgram.dir/vgram_train.cpp.o: src/vgram_train.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/data/CAVALCANTE/SABGL/pyram/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/vgram.dir/vgram_train.cpp.o"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vgram.dir/vgram_train.cpp.o -c /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_train.cpp

src/CMakeFiles/vgram.dir/vgram_train.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vgram.dir/vgram_train.cpp.i"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_train.cpp > CMakeFiles/vgram.dir/vgram_train.cpp.i

src/CMakeFiles/vgram.dir/vgram_train.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vgram.dir/vgram_train.cpp.s"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_train.cpp -o CMakeFiles/vgram.dir/vgram_train.cpp.s

src/CMakeFiles/vgram.dir/vgram_train.cpp.o.requires:

.PHONY : src/CMakeFiles/vgram.dir/vgram_train.cpp.o.requires

src/CMakeFiles/vgram.dir/vgram_train.cpp.o.provides: src/CMakeFiles/vgram.dir/vgram_train.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/vgram.dir/build.make src/CMakeFiles/vgram.dir/vgram_train.cpp.o.provides.build
.PHONY : src/CMakeFiles/vgram.dir/vgram_train.cpp.o.provides

src/CMakeFiles/vgram.dir/vgram_train.cpp.o.provides.build: src/CMakeFiles/vgram.dir/vgram_train.cpp.o


src/CMakeFiles/vgram.dir/vgram_test.cpp.o: src/CMakeFiles/vgram.dir/flags.make
src/CMakeFiles/vgram.dir/vgram_test.cpp.o: src/vgram_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/data/CAVALCANTE/SABGL/pyram/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/vgram.dir/vgram_test.cpp.o"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vgram.dir/vgram_test.cpp.o -c /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_test.cpp

src/CMakeFiles/vgram.dir/vgram_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vgram.dir/vgram_test.cpp.i"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_test.cpp > CMakeFiles/vgram.dir/vgram_test.cpp.i

src/CMakeFiles/vgram.dir/vgram_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vgram.dir/vgram_test.cpp.s"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_test.cpp -o CMakeFiles/vgram.dir/vgram_test.cpp.s

src/CMakeFiles/vgram.dir/vgram_test.cpp.o.requires:

.PHONY : src/CMakeFiles/vgram.dir/vgram_test.cpp.o.requires

src/CMakeFiles/vgram.dir/vgram_test.cpp.o.provides: src/CMakeFiles/vgram.dir/vgram_test.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/vgram.dir/build.make src/CMakeFiles/vgram.dir/vgram_test.cpp.o.provides.build
.PHONY : src/CMakeFiles/vgram.dir/vgram_test.cpp.o.provides

src/CMakeFiles/vgram.dir/vgram_test.cpp.o.provides.build: src/CMakeFiles/vgram.dir/vgram_test.cpp.o


src/CMakeFiles/vgram.dir/vgram_test_dtw.cpp.o: src/CMakeFiles/vgram.dir/flags.make
src/CMakeFiles/vgram.dir/vgram_test_dtw.cpp.o: src/vgram_test_dtw.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/data/CAVALCANTE/SABGL/pyram/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/vgram.dir/vgram_test_dtw.cpp.o"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vgram.dir/vgram_test_dtw.cpp.o -c /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_test_dtw.cpp

src/CMakeFiles/vgram.dir/vgram_test_dtw.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vgram.dir/vgram_test_dtw.cpp.i"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_test_dtw.cpp > CMakeFiles/vgram.dir/vgram_test_dtw.cpp.i

src/CMakeFiles/vgram.dir/vgram_test_dtw.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vgram.dir/vgram_test_dtw.cpp.s"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_test_dtw.cpp -o CMakeFiles/vgram.dir/vgram_test_dtw.cpp.s

src/CMakeFiles/vgram.dir/vgram_test_dtw.cpp.o.requires:

.PHONY : src/CMakeFiles/vgram.dir/vgram_test_dtw.cpp.o.requires

src/CMakeFiles/vgram.dir/vgram_test_dtw.cpp.o.provides: src/CMakeFiles/vgram.dir/vgram_test_dtw.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/vgram.dir/build.make src/CMakeFiles/vgram.dir/vgram_test_dtw.cpp.o.provides.build
.PHONY : src/CMakeFiles/vgram.dir/vgram_test_dtw.cpp.o.provides

src/CMakeFiles/vgram.dir/vgram_test_dtw.cpp.o.provides.build: src/CMakeFiles/vgram.dir/vgram_test_dtw.cpp.o


src/CMakeFiles/vgram.dir/vgram_data.cpp.o: src/CMakeFiles/vgram.dir/flags.make
src/CMakeFiles/vgram.dir/vgram_data.cpp.o: src/vgram_data.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/data/CAVALCANTE/SABGL/pyram/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/CMakeFiles/vgram.dir/vgram_data.cpp.o"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vgram.dir/vgram_data.cpp.o -c /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_data.cpp

src/CMakeFiles/vgram.dir/vgram_data.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vgram.dir/vgram_data.cpp.i"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_data.cpp > CMakeFiles/vgram.dir/vgram_data.cpp.i

src/CMakeFiles/vgram.dir/vgram_data.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vgram.dir/vgram_data.cpp.s"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_data.cpp -o CMakeFiles/vgram.dir/vgram_data.cpp.s

src/CMakeFiles/vgram.dir/vgram_data.cpp.o.requires:

.PHONY : src/CMakeFiles/vgram.dir/vgram_data.cpp.o.requires

src/CMakeFiles/vgram.dir/vgram_data.cpp.o.provides: src/CMakeFiles/vgram.dir/vgram_data.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/vgram.dir/build.make src/CMakeFiles/vgram.dir/vgram_data.cpp.o.provides.build
.PHONY : src/CMakeFiles/vgram.dir/vgram_data.cpp.o.provides

src/CMakeFiles/vgram.dir/vgram_data.cpp.o.provides.build: src/CMakeFiles/vgram.dir/vgram_data.cpp.o


src/CMakeFiles/vgram.dir/vgram_synapse.cpp.o: src/CMakeFiles/vgram.dir/flags.make
src/CMakeFiles/vgram.dir/vgram_synapse.cpp.o: src/vgram_synapse.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/data/CAVALCANTE/SABGL/pyram/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/CMakeFiles/vgram.dir/vgram_synapse.cpp.o"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vgram.dir/vgram_synapse.cpp.o -c /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_synapse.cpp

src/CMakeFiles/vgram.dir/vgram_synapse.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vgram.dir/vgram_synapse.cpp.i"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_synapse.cpp > CMakeFiles/vgram.dir/vgram_synapse.cpp.i

src/CMakeFiles/vgram.dir/vgram_synapse.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vgram.dir/vgram_synapse.cpp.s"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_synapse.cpp -o CMakeFiles/vgram.dir/vgram_synapse.cpp.s

src/CMakeFiles/vgram.dir/vgram_synapse.cpp.o.requires:

.PHONY : src/CMakeFiles/vgram.dir/vgram_synapse.cpp.o.requires

src/CMakeFiles/vgram.dir/vgram_synapse.cpp.o.provides: src/CMakeFiles/vgram.dir/vgram_synapse.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/vgram.dir/build.make src/CMakeFiles/vgram.dir/vgram_synapse.cpp.o.provides.build
.PHONY : src/CMakeFiles/vgram.dir/vgram_synapse.cpp.o.provides

src/CMakeFiles/vgram.dir/vgram_synapse.cpp.o.provides.build: src/CMakeFiles/vgram.dir/vgram_synapse.cpp.o


src/CMakeFiles/vgram.dir/vgram_output.cpp.o: src/CMakeFiles/vgram.dir/flags.make
src/CMakeFiles/vgram.dir/vgram_output.cpp.o: src/vgram_output.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/data/CAVALCANTE/SABGL/pyram/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object src/CMakeFiles/vgram.dir/vgram_output.cpp.o"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vgram.dir/vgram_output.cpp.o -c /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_output.cpp

src/CMakeFiles/vgram.dir/vgram_output.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vgram.dir/vgram_output.cpp.i"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_output.cpp > CMakeFiles/vgram.dir/vgram_output.cpp.i

src/CMakeFiles/vgram.dir/vgram_output.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vgram.dir/vgram_output.cpp.s"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_output.cpp -o CMakeFiles/vgram.dir/vgram_output.cpp.s

src/CMakeFiles/vgram.dir/vgram_output.cpp.o.requires:

.PHONY : src/CMakeFiles/vgram.dir/vgram_output.cpp.o.requires

src/CMakeFiles/vgram.dir/vgram_output.cpp.o.provides: src/CMakeFiles/vgram.dir/vgram_output.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/vgram.dir/build.make src/CMakeFiles/vgram.dir/vgram_output.cpp.o.provides.build
.PHONY : src/CMakeFiles/vgram.dir/vgram_output.cpp.o.provides

src/CMakeFiles/vgram.dir/vgram_output.cpp.o.provides.build: src/CMakeFiles/vgram.dir/vgram_output.cpp.o


src/CMakeFiles/vgram.dir/vgram_error.cpp.o: src/CMakeFiles/vgram.dir/flags.make
src/CMakeFiles/vgram.dir/vgram_error.cpp.o: src/vgram_error.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/data/CAVALCANTE/SABGL/pyram/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object src/CMakeFiles/vgram.dir/vgram_error.cpp.o"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vgram.dir/vgram_error.cpp.o -c /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_error.cpp

src/CMakeFiles/vgram.dir/vgram_error.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vgram.dir/vgram_error.cpp.i"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_error.cpp > CMakeFiles/vgram.dir/vgram_error.cpp.i

src/CMakeFiles/vgram.dir/vgram_error.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vgram.dir/vgram_error.cpp.s"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_error.cpp -o CMakeFiles/vgram.dir/vgram_error.cpp.s

src/CMakeFiles/vgram.dir/vgram_error.cpp.o.requires:

.PHONY : src/CMakeFiles/vgram.dir/vgram_error.cpp.o.requires

src/CMakeFiles/vgram.dir/vgram_error.cpp.o.provides: src/CMakeFiles/vgram.dir/vgram_error.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/vgram.dir/build.make src/CMakeFiles/vgram.dir/vgram_error.cpp.o.provides.build
.PHONY : src/CMakeFiles/vgram.dir/vgram_error.cpp.o.provides

src/CMakeFiles/vgram.dir/vgram_error.cpp.o.provides.build: src/CMakeFiles/vgram.dir/vgram_error.cpp.o


src/CMakeFiles/vgram.dir/vgram_utils.cpp.o: src/CMakeFiles/vgram.dir/flags.make
src/CMakeFiles/vgram.dir/vgram_utils.cpp.o: src/vgram_utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/data/CAVALCANTE/SABGL/pyram/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object src/CMakeFiles/vgram.dir/vgram_utils.cpp.o"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vgram.dir/vgram_utils.cpp.o -c /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_utils.cpp

src/CMakeFiles/vgram.dir/vgram_utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vgram.dir/vgram_utils.cpp.i"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_utils.cpp > CMakeFiles/vgram.dir/vgram_utils.cpp.i

src/CMakeFiles/vgram.dir/vgram_utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vgram.dir/vgram_utils.cpp.s"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram_utils.cpp -o CMakeFiles/vgram.dir/vgram_utils.cpp.s

src/CMakeFiles/vgram.dir/vgram_utils.cpp.o.requires:

.PHONY : src/CMakeFiles/vgram.dir/vgram_utils.cpp.o.requires

src/CMakeFiles/vgram.dir/vgram_utils.cpp.o.provides: src/CMakeFiles/vgram.dir/vgram_utils.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/vgram.dir/build.make src/CMakeFiles/vgram.dir/vgram_utils.cpp.o.provides.build
.PHONY : src/CMakeFiles/vgram.dir/vgram_utils.cpp.o.provides

src/CMakeFiles/vgram.dir/vgram_utils.cpp.o.provides.build: src/CMakeFiles/vgram.dir/vgram_utils.cpp.o


# Object files for target vgram
vgram_OBJECTS = \
"CMakeFiles/vgram.dir/vgram_base.cpp.o" \
"CMakeFiles/vgram.dir/vgram_train.cpp.o" \
"CMakeFiles/vgram.dir/vgram_test.cpp.o" \
"CMakeFiles/vgram.dir/vgram_test_dtw.cpp.o" \
"CMakeFiles/vgram.dir/vgram_data.cpp.o" \
"CMakeFiles/vgram.dir/vgram_synapse.cpp.o" \
"CMakeFiles/vgram.dir/vgram_output.cpp.o" \
"CMakeFiles/vgram.dir/vgram_error.cpp.o" \
"CMakeFiles/vgram.dir/vgram_utils.cpp.o"

# External object files for target vgram
vgram_EXTERNAL_OBJECTS =

src/vgram.so: src/CMakeFiles/vgram.dir/vgram_base.cpp.o
src/vgram.so: src/CMakeFiles/vgram.dir/vgram_train.cpp.o
src/vgram.so: src/CMakeFiles/vgram.dir/vgram_test.cpp.o
src/vgram.so: src/CMakeFiles/vgram.dir/vgram_test_dtw.cpp.o
src/vgram.so: src/CMakeFiles/vgram.dir/vgram_data.cpp.o
src/vgram.so: src/CMakeFiles/vgram.dir/vgram_synapse.cpp.o
src/vgram.so: src/CMakeFiles/vgram.dir/vgram_output.cpp.o
src/vgram.so: src/CMakeFiles/vgram.dir/vgram_error.cpp.o
src/vgram.so: src/CMakeFiles/vgram.dir/vgram_utils.cpp.o
src/vgram.so: src/CMakeFiles/vgram.dir/build.make
src/vgram.so: src/CMakeFiles/vgram.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/data/CAVALCANTE/SABGL/pyram/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX shared library vgram.so"
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vgram.dir/link.txt --verbose=$(VERBOSE)
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && /usr/bin/cmake -E copy /mnt/data/CAVALCANTE/SABGL/pyram/src/vgram.so /mnt/data/CAVALCANTE/SABGL/pyram/mae/vgram

# Rule to build all files generated by this target.
src/CMakeFiles/vgram.dir/build: src/vgram.so

.PHONY : src/CMakeFiles/vgram.dir/build

src/CMakeFiles/vgram.dir/requires: src/CMakeFiles/vgram.dir/vgram_base.cpp.o.requires
src/CMakeFiles/vgram.dir/requires: src/CMakeFiles/vgram.dir/vgram_train.cpp.o.requires
src/CMakeFiles/vgram.dir/requires: src/CMakeFiles/vgram.dir/vgram_test.cpp.o.requires
src/CMakeFiles/vgram.dir/requires: src/CMakeFiles/vgram.dir/vgram_test_dtw.cpp.o.requires
src/CMakeFiles/vgram.dir/requires: src/CMakeFiles/vgram.dir/vgram_data.cpp.o.requires
src/CMakeFiles/vgram.dir/requires: src/CMakeFiles/vgram.dir/vgram_synapse.cpp.o.requires
src/CMakeFiles/vgram.dir/requires: src/CMakeFiles/vgram.dir/vgram_output.cpp.o.requires
src/CMakeFiles/vgram.dir/requires: src/CMakeFiles/vgram.dir/vgram_error.cpp.o.requires
src/CMakeFiles/vgram.dir/requires: src/CMakeFiles/vgram.dir/vgram_utils.cpp.o.requires

.PHONY : src/CMakeFiles/vgram.dir/requires

src/CMakeFiles/vgram.dir/clean:
	cd /mnt/data/CAVALCANTE/SABGL/pyram/src && $(CMAKE_COMMAND) -P CMakeFiles/vgram.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/vgram.dir/clean

src/CMakeFiles/vgram.dir/depend:
	cd /mnt/data/CAVALCANTE/SABGL/pyram && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/data/CAVALCANTE/SABGL/pyram /mnt/data/CAVALCANTE/SABGL/pyram/src /mnt/data/CAVALCANTE/SABGL/pyram /mnt/data/CAVALCANTE/SABGL/pyram/src /mnt/data/CAVALCANTE/SABGL/pyram/src/CMakeFiles/vgram.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/vgram.dir/depend
