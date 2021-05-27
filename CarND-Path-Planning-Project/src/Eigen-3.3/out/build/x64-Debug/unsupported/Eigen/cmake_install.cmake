# Install script for directory: C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/unsupported/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/out/install/x64-Debug")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "CMAKE_OBJDUMP-NOTFOUND")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/unsupported/Eigen/AdolcForward"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/unsupported/Eigen/AlignedVector3"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/unsupported/Eigen/ArpackSupport"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/unsupported/Eigen/AutoDiff"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/unsupported/Eigen/BVH"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/unsupported/Eigen/EulerAngles"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/unsupported/Eigen/FFT"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/unsupported/Eigen/IterativeSolvers"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/unsupported/Eigen/KroneckerProduct"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/unsupported/Eigen/LevenbergMarquardt"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/unsupported/Eigen/MatrixFunctions"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/unsupported/Eigen/MoreVectorization"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/unsupported/Eigen/MPRealSupport"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/unsupported/Eigen/NonLinearOptimization"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/unsupported/Eigen/NumericalDiff"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/unsupported/Eigen/OpenGLSupport"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/unsupported/Eigen/Polynomials"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/unsupported/Eigen/Skyline"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/unsupported/Eigen/SparseExtra"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/unsupported/Eigen/SpecialFunctions"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/unsupported/Eigen/Splines"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE DIRECTORY FILES "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/out/build/x64-Debug/unsupported/Eigen/CXX11/cmake_install.cmake")

endif()

