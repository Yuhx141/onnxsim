cmake_minimum_required(VERSION 3.16)
project(onnx_remote_transport LANGUAGES CXX)

include(CTest)

option(ONNX_REMOTE_AXCL "Build the AXCL-backed remote worker" OFF)
option(ONNXSIM_REMOTE_DORA "Build the DORA node adapter" OFF)
option(ONNXSIM_REMOTE_ROS2 "Build the optional ROS2 binary bridge" OFF)
option(ONNXSIM_REMOTE_GRPC "Generate the optional host-side gRPC bindings" OFF)
set(DORA_NODE_API_INCLUDE_DIR "" CACHE PATH
    "DORA C node API include directory (contains node_api.h)")
set(DORA_NODE_API_LIBRARY "" CACHE FILEPATH
    "DORA C node API static library")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(onnx_remote_transport STATIC remote_transport.cpp remote_profile.cpp)
target_include_directories(onnx_remote_transport PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if (WIN32)
  target_link_libraries(onnx_remote_transport PUBLIC ws2_32)
else()
  find_package(Threads REQUIRED)
  target_link_libraries(onnx_remote_transport PUBLIC Threads::Threads)
endif()

add_executable(onnx-remote-worker remote_worker.cpp)
target_link_libraries(onnx-remote-worker PRIVATE onnx_remote_transport)

add_executable(onnx-remote-client remote_client.cpp)
target_link_libraries(onnx-remote-client PRIVATE onnx_remote_transport)

add_executable(onnx-remote-compiler remote_compiler.cpp)
target_link_libraries(onnx-remote-compiler PRIVATE onnx_remote_transport)

add_executable(onnx-remote-mock-runner remote_mock_runner.cpp)
target_link_libraries(onnx-remote-mock-runner PRIVATE onnx_remote_transport)

add_executable(onnx-remote-attach-test remote_attach_test.cpp)
target_link_libraries(onnx-remote-attach-test PRIVATE onnx_remote_transport)

# The schema is source-only and does not pull protobuf or gRPC into the
# default build.  Install it so an optional host-side gateway can generate
# bindings with the toolchain available on that host.
install(FILES proto/onnxsim_remote.proto proto/README.md
        DESTINATION share/onnx-remote/proto)
install(FILES remote_transport.h remote_profile.h
        DESTINATION include/onnx-remote)
install(FILES ort_plugin/onnxsim_remote_ep.manifest.json ort_plugin/README.md
        DESTINATION share/onnx-remote/ort-plugin)

if (ONNXSIM_REMOTE_GRPC)
  # Prefer a package config when the host distribution provides one, but also
  # support CMake's FindProtobuf module (common on Debian/Ubuntu and ROS
  # workspaces).
  find_package(absl CONFIG REQUIRED)
  find_package(ZLIB REQUIRED)
  find_package(Protobuf CONFIG QUIET)
  if (NOT Protobuf_FOUND)
    find_package(Protobuf REQUIRED)
  endif()
  find_package(gRPC CONFIG REQUIRED)

  if (TARGET protobuf::protoc)
    set(ONNXSIM_PROTOC "$<TARGET_FILE:protobuf::protoc>")
  else()
    find_program(ONNXSIM_PROTOC protoc REQUIRED)
  endif()
  if (TARGET gRPC::grpc_cpp_plugin)
    set(ONNXSIM_GRPC_CPP_PLUGIN "$<TARGET_FILE:gRPC::grpc_cpp_plugin>")
  else()
    find_program(ONNXSIM_GRPC_CPP_PLUGIN grpc_cpp_plugin REQUIRED)
  endif()

  set(ONNXSIM_GRPC_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/grpc")
  set(ONNXSIM_GRPC_PROTO_CPP "${ONNXSIM_GRPC_GEN_DIR}/onnxsim_remote.pb.cc")
  set(ONNXSIM_GRPC_PROTO_H "${ONNXSIM_GRPC_GEN_DIR}/onnxsim_remote.pb.h")
  set(ONNXSIM_GRPC_SERVICE_CPP "${ONNXSIM_GRPC_GEN_DIR}/onnxsim_remote.grpc.pb.cc")
  set(ONNXSIM_GRPC_SERVICE_H "${ONNXSIM_GRPC_GEN_DIR}/onnxsim_remote.grpc.pb.h")
  add_custom_command(
    OUTPUT ${ONNXSIM_GRPC_PROTO_CPP} ${ONNXSIM_GRPC_PROTO_H}
           ${ONNXSIM_GRPC_SERVICE_CPP} ${ONNXSIM_GRPC_SERVICE_H}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${ONNXSIM_GRPC_GEN_DIR}
    COMMAND ${ONNXSIM_PROTOC}
            --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto
            --cpp_out=${ONNXSIM_GRPC_GEN_DIR}
            --grpc_out=${ONNXSIM_GRPC_GEN_DIR}
            --plugin=protoc-gen-grpc=${ONNXSIM_GRPC_CPP_PLUGIN}
            onnxsim_remote.proto
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/proto/onnxsim_remote.proto
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/proto
    VERBATIM)
  add_library(onnx_remote_grpc STATIC
              ${ONNXSIM_GRPC_PROTO_CPP} ${ONNXSIM_GRPC_SERVICE_CPP})
  target_include_directories(onnx_remote_grpc PUBLIC ${ONNXSIM_GRPC_GEN_DIR})
  target_link_libraries(onnx_remote_grpc PUBLIC protobuf::libprotobuf gRPC::grpc++)
  add_executable(onnx-remote-grpc-gateway grpc_gateway.cpp grpc_gateway_main.cpp)
  target_include_directories(onnx-remote-grpc-gateway PRIVATE ${ONNXSIM_GRPC_GEN_DIR})
  target_link_libraries(onnx-remote-grpc-gateway PRIVATE
                        onnx_remote_transport onnx_remote_grpc gRPC::grpc++)
  if (BUILD_TESTING)
    add_executable(onnx-remote-grpc-gateway-test grpc_gateway.cpp
                   grpc_gateway_test.cpp)
    target_include_directories(onnx-remote-grpc-gateway-test PRIVATE
                               ${ONNXSIM_GRPC_GEN_DIR})
    target_link_libraries(onnx-remote-grpc-gateway-test PRIVATE
                          onnx_remote_transport onnx_remote_grpc gRPC::grpc++)
    add_test(NAME onnx-remote-grpc-gateway-smoke
             COMMAND onnx-remote-grpc-gateway-test)
  endif()
  install(TARGETS onnx-remote-grpc-gateway RUNTIME DESTINATION bin)
  install(TARGETS onnx_remote_grpc ARCHIVE DESTINATION lib)
  install(FILES ${ONNXSIM_GRPC_PROTO_H} ${ONNXSIM_GRPC_SERVICE_H}
          DESTINATION include/onnx-remote/grpc)
endif()

if (BUILD_TESTING)
  add_executable(onnx-remote-op-mapping-test remote_op_mapping_test.cpp)
  target_link_libraries(onnx-remote-op-mapping-test PRIVATE onnx_remote_transport)
  add_test(NAME onnx-remote-op-mapping COMMAND onnx-remote-op-mapping-test)

  add_executable(onnx-remote-transport-test remote_transport_test.cpp)
  target_link_libraries(onnx-remote-transport-test PRIVATE onnx_remote_transport)
  add_test(NAME onnx-remote-transport-roundtrip
           COMMAND onnx-remote-transport-test)
endif()

if (ONNX_REMOTE_AXCL)
  find_path(AXCL_INCLUDE_DIR axcl.h REQUIRED)
  find_library(AXCL_RT_LIBRARY axcl_rt REQUIRED)
  find_library(AXCL_SYS_LIBRARY axcl_sys REQUIRED)
  add_executable(onnx-remote-axcl-worker remote_axcl_worker.cpp)
  target_include_directories(onnx-remote-axcl-worker PRIVATE ${AXCL_INCLUDE_DIR})
  target_link_libraries(onnx-remote-axcl-worker PRIVATE
                        onnx_remote_transport ${AXCL_RT_LIBRARY} ${AXCL_SYS_LIBRARY})
endif()

if (ONNXSIM_REMOTE_DORA)
  if (NOT DORA_NODE_API_INCLUDE_DIR OR NOT DORA_NODE_API_LIBRARY)
    message(FATAL_ERROR
      "ONNXSIM_REMOTE_DORA requires DORA_NODE_API_INCLUDE_DIR and "
      "DORA_NODE_API_LIBRARY")
  endif()
  add_executable(onnx-remote-dora-node dora_node.cpp)
  target_include_directories(onnx-remote-dora-node PRIVATE
                             ${DORA_NODE_API_INCLUDE_DIR})
  target_link_libraries(onnx-remote-dora-node PRIVATE
                        onnx_remote_transport ${DORA_NODE_API_LIBRARY})
  if (UNIX AND NOT APPLE)
    target_link_libraries(onnx-remote-dora-node PRIVATE dl m rt pthread)
  endif()
endif()

if (ONNXSIM_REMOTE_ROS2)
  find_package(ament_cmake REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(std_srvs REQUIRED)
  add_executable(onnx-remote-ros2-bridge ros2_bridge.cpp)
  target_link_libraries(onnx-remote-ros2-bridge PRIVATE onnx_remote_transport)
  ament_target_dependencies(onnx-remote-ros2-bridge rclcpp std_msgs std_srvs)
  install(TARGETS onnx-remote-ros2-bridge
          DESTINATION lib/${PROJECT_NAME})
  ament_package()
endif()
