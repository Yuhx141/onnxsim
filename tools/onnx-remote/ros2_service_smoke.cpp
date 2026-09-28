#include <chrono>
#include <iostream>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>

using Trigger = std_srvs::srv::Trigger;
using namespace std::chrono_literals;

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("onnx_remote_ros2_service_smoke");
  auto health = node->create_client<Trigger>("health");
  auto capabilities = node->create_client<Trigger>("capabilities");
  if (!health->wait_for_service(5s) || !capabilities->wait_for_service(5s)) {
    std::cerr << "ROS2 bridge services were not available\n";
    rclcpp::shutdown();
    return 1;
  }

  auto call = [&node](const rclcpp::Client<Trigger>::SharedPtr& client,
                      const char* name) {
    auto future = client->async_send_request(std::make_shared<Trigger::Request>());
    if (rclcpp::spin_until_future_complete(node, future, 5s) !=
        rclcpp::FutureReturnCode::SUCCESS) {
      std::cerr << name << " service call timed out\n";
      return false;
    }
    const auto response = future.get();
    if (!response->success) {
      std::cerr << name << " service reported failure: " << response->message
                << '\n';
      return false;
    }
    return true;
  };

  const bool ok = call(health, "health") && call(capabilities, "capabilities");
  rclcpp::shutdown();
  if (ok) std::cout << "ROS2 service smoke passed\n";
  return ok ? 0 : 1;
}
