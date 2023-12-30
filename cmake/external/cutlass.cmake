include(ExternalProject)

set(CUTLASS_TAR ${CMAKE_SOURCE_DIR}/third_party/cutlass/cutlass.tar.gz)
set(CUTLASS_SHARED_LIB libcutlass.so)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_party)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${CUTLASS_TAR} 
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_party
  )

set(CUTLASS_ROOT "${CMAKE_CURRENT_BINARY_DIR}/third_party/cutlass")
set(CUTLASS_SHARED_LIB libcutlass.so)
set(CUTLASS_INCLUDE_DIR ${CUTLASS_ROOT}/include)
set(CUTLASS_LIB_DIR ${CUTLASS_ROOT}/lib)
set(CUTLASS_INSTALL ${CUTLASS_ROOT}/install)
set(CUTLASS_DLL_PATH ${CUTLASS_INSTALL}/${CUTLASS_SHARED_LIB})
find_library ( CUTLASS_LIBRARY NAMES cutlass HINTS ${CUTLASS_BUILD}/tools/library )
message("CUTLASS_INCLUDE_DIR:${CUTLASS_INCLUDE_DIR}")
message("CUTLASS_LIB_DIR:${CUTLASS_LIB_DIR}")
message("CUTLASS_DLL_PATH:${CUTLASS_DLL_PATH}")

