cmake_minimum_required(VERSION 3.8)
project(hsar_msgs)

# ============================================================
# Dependencies
# ============================================================
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# ============================================================
# Interface Generation
# ============================================================
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/HumanPosture.msg"
  "msg/HumanPose3D.msg"
  DEPENDENCIES std_msgs
)

# ============================================================
# Export Dependencies
# ============================================================
ament_export_dependencies(rosidl_default_runtime)

ament_package()
