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
  auto node = std::make_shared<rclcpp::Node>("onnx_remote_failover_smoke");
  bool selected = false;
  bool expired = false;
  const auto subscription = node->create_subscription<String>(
      "onnx_remote/discovery_status",
      rclcpp::QoS(10).transient_local().reliable(),
      [&selected, &expired](const String::SharedPtr message) {
        if (message->data.find("\"runner_id\":\"runner-a\"") ==
            std::string::npos)
          return;
        if (message->data.find("\"state\":\"selected\"") !=
            std::string::npos)
          selected = true;
        if (message->data.find("\"state\":\"expired\"") !=
            std::string::npos)
          expired = true;
      });
  (void)subscription;
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (rclcpp::ok() && !expired &&
         std::chrono::steady_clock::now() < deadline) {
    rclcpp::spin_some(node);
    std::this_thread::sleep_for(20ms);
  }
  rclcpp::shutdown();
  if (!selected || !expired) {
    std::cerr << "ROS2 discovery failover did not observe selected then expired\n";
    return 1;
  }
  std::cout << "ROS2 discovery failover smoke passed\n";
  return 0;
}
