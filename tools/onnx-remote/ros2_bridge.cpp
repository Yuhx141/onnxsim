// Optional ROS2 control-plane bridge for the dependency-free ONNX transport.
// Tensor/profile payloads stay binary UInt8 messages; ROS2 only supplies
// discovery, health, and capability services.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/u_int8_multi_array.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "remote_profile.h"
#include "remote_transport.h"

using namespace std::chrono_literals;
using UInt8MultiArray = std_msgs::msg::UInt8MultiArray;
using String = std_msgs::msg::String;
using Trigger = std_srvs::srv::Trigger;

static std::string json_escape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const unsigned char ch : value) {
    switch (ch) {
      case '"': escaped += "\\\""; break;
      case '\\': escaped += "\\\\"; break;
      case '\b': escaped += "\\b"; break;
      case '\f': escaped += "\\f"; break;
      case '\n': escaped += "\\n"; break;
      case '\r': escaped += "\\r"; break;
      case '\t': escaped += "\\t"; break;
      default:
        if (ch < 0x20) {
          constexpr char hex[] = "0123456789abcdef";
          escaped += "\\u00";
          escaped += hex[ch >> 4];
          escaped += hex[ch & 0x0f];
        } else {
          escaped += static_cast<char>(ch);
        }
    }
  }
  return escaped;
}

class OnnxRemoteBridge final : public rclcpp::Node {
 public:
  OnnxRemoteBridge() : Node("onnx_remote_bridge") {
    host_ = declare_parameter<std::string>("remote_host", "127.0.0.1");
    port_ = validate_port(declare_parameter<int>("remote_port", 39501),
                          "remote_port");
    configured_host_ = host_;
    configured_port_ = port_;
    connect_timeout_ms_ = declare_parameter<int>("connect_timeout_ms", 2000);
    io_timeout_ms_ = declare_parameter<int>("io_timeout_ms", 0);
    auto_discover_ = declare_parameter<bool>("auto_discover", false);
    discovery_topic_ = declare_parameter<std::string>("discovery_topic",
                                                      "onnx_remote/runners");
    discovery_target_ = declare_parameter<std::string>("discovery_target", "");
    verify_discovery_ = declare_parameter<bool>("verify_discovery", true);
    publish_discovery_status_ =
        declare_parameter<bool>("publish_discovery_status", true);
    discovery_status_topic_ = declare_parameter<std::string>(
        "discovery_status_topic", "onnx_remote/discovery_status");
    announce_period_ms_ = declare_parameter<int>("announce_period_ms", 5000);
    discovery_timeout_ms_ =
        declare_parameter<int>("discovery_timeout_ms", 15000);
    advertise_host_ = declare_parameter<std::string>("advertise_host", host_);
    runner_id_ = declare_parameter<std::string>("runner_id", get_name());
    publish_profile_ = declare_parameter<bool>("publish_profile", true);
    profile_topic_ = declare_parameter<std::string>("profile_topic", "profile");
    result_ = create_publisher<UInt8MultiArray>("result", rclcpp::QoS(10));
    if (publish_profile_) {
      profile_ = create_publisher<String>(profile_topic_, rclcpp::QoS(10));
    }
    discovery_ = create_publisher<String>(
        discovery_topic_, rclcpp::QoS(1).transient_local().reliable());
    discovery_sub_ = create_subscription<String>(
        discovery_topic_, rclcpp::QoS(10).transient_local().reliable(),
        [this](const String::SharedPtr message) { discover(message); });
    if (publish_discovery_status_) {
      discovery_status_ = create_publisher<String>(discovery_status_topic_,
                                                   rclcpp::QoS(10)
                                                       .transient_local()
                                                       .reliable());
    }
    announce_timer_ = create_wall_timer(
        std::chrono::milliseconds(std::max(100, announce_period_ms_)),
        [this]() {
          expire_discovery();
          announce();
        });
    run_ = create_subscription<UInt8MultiArray>(
        "run", rclcpp::QoS(10),
        [this](const UInt8MultiArray::SharedPtr message) { forward(message); });
    health_ = create_service<Trigger>(
        "health", [this](const std::shared_ptr<Trigger::Request>,
                         std::shared_ptr<Trigger::Response> response) {
          onnx_remote::Response capabilities;
          std::string error;
          if (!query_capabilities(capabilities, error)) {
            response->success = false;
            response->message = error.empty() ? "remote worker is unreachable"
                                               : error;
            return;
          }
          response->success = true;
          response->message = "remote worker is ready";
        });
    capabilities_ = create_service<
        Trigger>("capabilities", [this](const std::shared_ptr<Trigger::Request>,
                                    std::shared_ptr<Trigger::Response>
                                        response) {
      onnx_remote::Response capabilities;
      std::string error;
      response->success = query_capabilities(capabilities, error);
      response->message = response->success
                              ? capabilities.manifest
                              : (error.empty() ? "capability query failed"
                                               : error);
    });
  }

 private:
  static int validate_port(int port, const char* parameter) {
    if (port <= 0 || port > 65535)
      throw std::invalid_argument(std::string(parameter) +
                                  " must be between 1 and 65535");
    return port;
  }

  bool query_capabilities_at(const std::string& host, int port,
                             onnx_remote::Response& response,
                             std::string& error) const {
    const int fd = onnx_remote::connect_tcp_timeout(
        host, static_cast<uint16_t>(port), connect_timeout_ms_);
    if (fd < 0) {
      error = "remote worker capability connection failed";
      return false;
    }
    if (!onnx_remote::set_socket_io_timeout(fd, io_timeout_ms_)) {
      onnx_remote::close_socket(fd);
      error = "ROS2 bridge: cannot set capability socket timeout";
      return false;
    }
    onnx_remote::Request request;
    request.op = "capabilities";
    const bool sent = onnx_remote::send_request(fd, request, error);
    const bool received = sent &&
                          onnx_remote::receive_response(fd, response, error);
    onnx_remote::close_socket(fd);
    if (!received) return false;
    if (!response.ok) {
      error = response.error.empty() ? "remote worker is not ready"
                                     : response.error;
      return false;
    }
    return true;
  }

  bool query_capabilities(onnx_remote::Response& response,
                          std::string& error) const {
    return query_capabilities_at(host_, port_, response, error);
  }

  static std::string json_string(const std::string& json,
                                 const std::string& key) {
    const std::string marker = "\"" + key + "\":\"";
    const size_t start = json.find(marker);
    if (start == std::string::npos) return {};
    const size_t value_start = start + marker.size();
    std::string value;
    value.reserve(json.size() - value_start);
    bool escaped = false;
    for (size_t i = value_start; i < json.size(); ++i) {
      const char c = json[i];
      if (!escaped) {
        if (c == '"') return value;
        if (c == '\\') {
          escaped = true;
          continue;
        }
        value += c;
        continue;
      }
      escaped = false;
      switch (c) {
        case '"': value += '"'; break;
        case '\\': value += '\\'; break;
        case '/': value += '/'; break;
        case 'b': value += '\b'; break;
        case 'f': value += '\f'; break;
        case 'n': value += '\n'; break;
        case 'r': value += '\r'; break;
        case 't': value += '\t'; break;
        default:
          // Discovery fields are normally hostnames/IPs. Preserve unknown
          // escapes rather than accepting a truncated or ambiguous value.
          value += '\\';
          value += c;
          break;
      }
    }
    return {};
  }

  static int json_int(const std::string& json, const std::string& key) {
    const std::string marker = "\"" + key + "\":";
    const size_t start = json.find(marker);
    if (start == std::string::npos) return 0;
    const size_t value_start = start + marker.size();
    return std::atoi(json.c_str() + value_start);
  }

  static bool json_bool(const std::string& json, const std::string& key,
                        bool fallback = false) {
    const std::string marker = "\"" + key + "\":";
    const size_t start = json.find(marker);
    if (start == std::string::npos) return fallback;
    const size_t value_start = start + marker.size();
    if (json.compare(value_start, 4, "true") == 0) return true;
    if (json.compare(value_start, 5, "false") == 0) return false;
    return fallback;
  }

  void publish_profile(const onnx_remote::Response& response) {
    if (!publish_profile_ || profile_ == nullptr || response.profile.empty()) return;
    String message;
    message.data = onnx_remote::profile_json(response);
    profile_->publish(std::move(message));
  }

  void publish_discovery_status(const std::string& state,
                                const std::string& runner_id,
                                const std::string& host, int port,
                                const std::string& error = {}) {
    if (!publish_discovery_status_ || discovery_status_ == nullptr) return;
    String message;
    message.data = "{\"schema_version\":1,\"state\":\"" +
                   json_escape(state) + "\",\"runner_id\":\"" +
                   json_escape(runner_id) + "\",\"host\":\"" +
                   json_escape(host) + "\",\"port\":" +
                   std::to_string(port);
    if (!error.empty())
      message.data += ",\"error\":\"" + json_escape(error) + "\"";
    message.data += "}";
    discovery_status_->publish(std::move(message));
  }

  void announce() {
    String message;
    message.data = "{\"schema_version\":1,\"runner_id\":\"" +
                   json_escape(runner_id_) + "\",\"host\":\"" +
                   json_escape(advertise_host_) +
                   "\",\"port\":" + std::to_string(port_) + ",\"target\":\"" +
                   json_escape(discovery_target_) +
                   "\",\"transport\":\"onnx-remote-v5\",\"ready\":true,"
                   "\"ttl_ms\":" + std::to_string(discovery_timeout_ms_) + ","
                   "\"profiling\":[\"off\",\"summary\",\"detailed\"]}";
    discovery_->publish(std::move(message));
  }

  void discover(const String::SharedPtr& message) {
    if (!auto_discover_) return;
    const std::string id = json_string(message->data, "runner_id");
    if (id.empty() || id == runner_id_) return;
    if (json_string(message->data, "transport") != "onnx-remote-v5" ||
        !json_bool(message->data, "ready")) return;
    const std::string target = json_string(message->data, "target");
    if (!discovery_target_.empty() && target != discovery_target_) return;
    const std::string host = json_string(message->data, "host");
    const int port = json_int(message->data, "port");
    if (host.empty() || port <= 0 || port > 65535) return;
    const auto now = std::chrono::steady_clock::now();
    if (discovered_ && id != discovered_runner_id_) {
      const auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
          now - last_discovery_);
      if (age.count() <= discovered_timeout_ms_) {
        // Keep a live lease stable instead of switching based on DDS delivery
        // order when multiple matching runners announce periodically.
        return;
      }
    }
    if (verify_discovery_) {
      onnx_remote::Response capabilities;
      std::string error;
      if (!query_capabilities_at(host, port, capabilities, error)) {
        publish_discovery_status("rejected", id, host, port, error);
        RCLCPP_WARN(get_logger(),
                    "ignoring discovered runner %s at %s:%d: %s", id.c_str(),
                    host.c_str(), port, error.c_str());
        return;
      }
    }
    host_ = host;
    port_ = port;
    last_discovery_ = now;
    const int advertised_ttl = json_int(message->data, "ttl_ms");
    discovered_timeout_ms_ = advertised_ttl > 0
                                 ? std::min(discovery_timeout_ms_, advertised_ttl)
                                 : discovery_timeout_ms_;
    discovered_ = true;
    discovered_runner_id_ = id;
    publish_discovery_status("selected", id, host_, port_);
    RCLCPP_INFO(get_logger(), "auto-discovered runner %s at %s:%d", id.c_str(),
                host_.c_str(), port_);
  }

  void expire_discovery() {
    if (!auto_discover_ || !discovered_ || discovery_timeout_ms_ <= 0) return;
    const auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - last_discovery_);
    if (age.count() <= discovered_timeout_ms_) return;
    RCLCPP_WARN(get_logger(), "discovered runner expired after %ld ms",
                static_cast<long>(age.count()));
    const std::string expired_runner_id = discovered_runner_id_;
    const std::string expired_host = host_;
    const int expired_port = port_;
    host_ = configured_host_;
    port_ = configured_port_;
    publish_discovery_status("expired", expired_runner_id, expired_host,
                             expired_port);
    discovered_ = false;
    discovered_runner_id_.clear();
  }

  void forward(const UInt8MultiArray::SharedPtr& message) {
    onnx_remote::Request request;
    std::string error;
    if (!onnx_remote::decode_request_payload(
            message->data.data(), message->data.size(), request, error)) {
      publish_error(error);
      return;
    }
    const int fd = onnx_remote::connect_tcp_timeout(
        host_, static_cast<uint16_t>(port_), connect_timeout_ms_);
    if (fd < 0) {
      publish_error("ROS2 bridge: remote worker connection failed",
                    request.request_id);
      return;
    }
    if (!onnx_remote::set_socket_io_timeout(fd, io_timeout_ms_)) {
      onnx_remote::close_socket(fd);
      publish_error("ROS2 bridge: cannot set socket timeout",
                    request.request_id);
      return;
    }
    onnx_remote::Response response;
    const bool sent = onnx_remote::send_request(fd, request, error);
    const bool received =
        sent && onnx_remote::receive_response(fd, response, error);
    onnx_remote::close_socket(fd);
    if (!received) {
      publish_error(error.empty() ? "remote worker request failed" : error,
                    request.request_id);
      return;
    }
    if (response.request_id != request.request_id) {
      publish_error("ROS2 bridge: remote worker response request id mismatch",
                    request.request_id);
      return;
    }
    publish_profile(response);
    std::vector<uint8_t> payload;
    if (!onnx_remote::encode_response_payload(response, payload, error)) {
      publish_error(error);
      return;
    }
    UInt8MultiArray output;
    output.data = std::move(payload);
    result_->publish(std::move(output));
  }

  void publish_error(const std::string& error, uint64_t request_id = 0) {
    onnx_remote::Response response;
    response.request_id = request_id;
    response.error = error;
    std::vector<uint8_t> payload;
    std::string encode_error;
    if (!onnx_remote::encode_response_payload(response, payload,
                                              encode_error)) {
      RCLCPP_ERROR(get_logger(), "%s", error.c_str());
      return;
    }
    UInt8MultiArray output;
    output.data = std::move(payload);
    result_->publish(std::move(output));
  }

  std::string host_;
  int port_;
  std::string configured_host_;
  int configured_port_;
  int connect_timeout_ms_;
  int io_timeout_ms_;
  bool auto_discover_;
  bool verify_discovery_ = true;
  bool publish_discovery_status_ = true;
  int announce_period_ms_ = 5000;
  int discovery_timeout_ms_ = 15000;
  bool discovered_ = false;
  int discovered_timeout_ms_ = 15000;
  std::string discovered_runner_id_;
  bool publish_profile_ = true;
  std::chrono::steady_clock::time_point last_discovery_{};
  std::string discovery_topic_;
  std::string discovery_status_topic_;
  std::string discovery_target_;
  std::string advertise_host_;
  std::string runner_id_;
  std::string profile_topic_;
  rclcpp::Publisher<UInt8MultiArray>::SharedPtr result_;
  rclcpp::Publisher<String>::SharedPtr profile_;
  rclcpp::Publisher<String>::SharedPtr discovery_;
  rclcpp::Publisher<String>::SharedPtr discovery_status_;
  rclcpp::Subscription<UInt8MultiArray>::SharedPtr run_;
  rclcpp::Subscription<String>::SharedPtr discovery_sub_;
  rclcpp::TimerBase::SharedPtr announce_timer_;
  rclcpp::Service<Trigger>::SharedPtr health_;
  rclcpp::Service<Trigger>::SharedPtr capabilities_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OnnxRemoteBridge>());
  rclcpp::shutdown();
  return 0;
}
