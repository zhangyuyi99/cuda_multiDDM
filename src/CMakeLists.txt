cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(main LANGUAGES CUDA CXX)
# find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
#find_package(CUDAToolkit REQUIRED)
# find_package(ROOT REQUIRED)
#find_package(Eigen3 REQUIRED)


set(CMAKE_BUILD_TYPE Debug)

set(SRC_INCLUDE
	debug.cpp
	debug.hpp
	DDM.cu
	# cilia.cu
	azimuthal_average.cu
	azimuthal_average.cuh
	azimuthal_average_kernel.cuh
	DDM_kernel.cuh
	# cilia.cuh
	constants.hpp
	video_reader.cpp
	video_reader.hpp
)

set(SRC
	main.cpp
)
 
set(ALL_SRC
	${SRC_INCLUDE}
	${SRC}
)

set(PROJECTNAME "main")

add_executable(${PROJECTNAME} ${ALL_SRC})
target_link_libraries(${PROJECTNAME} PRIVATE  ${CUDA_LIBRARIES} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} nppc nppial nppidei nppif npps nppist cufft nvToolsExt ${ROOT_LIBRARIES} )
target_link_libraries(${PROJECTNAME} PRIVATE ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} nppc nppial nppidei nppif npps nppist cufft nvToolsExt ${ROOT_LIBRARIES} )
#target_link_libraries(${PROJECTNAME} PRIVATE  ${CUDA_LIBRARIES} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} nppc nppial nppidei nppif npps nppist CUDA::cufft CUDA::nvToolsExt ${ROOT_LIBRARIES} )
#target_link_libraries(${PROJECTNAME} PRIVATE ${OpenCV_LIBS} ${CUDA_LIBRARIES} CUDA::cufft CUDA::nvToolsExt Eigen3::Eigen)
