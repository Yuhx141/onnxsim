#include <chrono>
#include <iostream>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <string>
#include <thread>

using namespace std::chrono_literals;
using String = std_msgs::msg::String;

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("onnx_remote_discovery_smoke");
  bool selected = false;
  const auto subscription = node->create_subscription<String>(
      "onnx_remote/discovery_status",
      rclcpp::QoS(10).transient_local().reliable(),
      [&selected](const String::SharedPtr message) {
        if (message->data.find("\"state\":\"selected\"") !=
                std::string::npos &&
            message->data.find("\"runner_id\":\"runner-a\"") !=
                std::string::npos) {
          selected = true;
        }
      });
  (void)subscription;
  const auto deadline = std::chrono::steady_clock::now() + 8s;
  while (rclcpp::ok() && !selected &&
         std::chrono::steady_clock::now() < deadline) {
    rclcpp::spin_some(node);
    std::this_thread::sleep_for(20ms);
  }
  rclcpp::shutdown();
  if (!selected) {
    std::cerr << "ROS2 discovery did not select runner-a\n";
    return 1;
  }
  std::cout << "ROS2 discovery smoke passed\n";
  return 0;
}
