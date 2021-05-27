# Install script for directory: C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/Eigen" TYPE FILE FILES
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/Cholesky"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/CholmodSupport"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/Core"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/Dense"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/Eigen"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/Eigenvalues"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/Geometry"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/Householder"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/IterativeLinearSolvers"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/Jacobi"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/LU"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/MetisSupport"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/OrderingMethods"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/PaStiXSupport"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/PardisoSupport"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/QR"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/QtAlignedMalloc"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/SPQRSupport"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/SVD"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/Sparse"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/SparseCholesky"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/SparseCore"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/SparseLU"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/SparseQR"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/StdDeque"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/StdList"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/StdVector"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/SuperLUSupport"
    "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/UmfPackSupport"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/Eigen" TYPE DIRECTORY FILES "C:/playground/udacitySelfDrivingCar/prediction/CarND-Path-Planning-Project/src/Eigen-3.3/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

