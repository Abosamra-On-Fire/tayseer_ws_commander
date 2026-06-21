cmake_minimum_required(VERSION 3.5)
project(tayseer_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetObjectLocation.srv"
  "srv/ListObjects.srv"
  "srv/UpdateObject.srv"
  "action/ArmManipulation.action"
  "action/SlideObject.action"
  "action/Navigate.action"
  DEPENDENCIES geometry_msgs
)

ament_package()