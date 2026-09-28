{
  description = "onnxsim native remote transport with ROS2 development shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nix-ros-overlay = {
      url = "github:lopsided98/nix-ros-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, nix-ros-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ nix-ros-overlay.overlays.default ];
        };
        ros = pkgs.rosPackages.humble;
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            cmake
            pkg-config
            ros.ament-cmake
            ros.rclcpp
            ros.std-msgs
            ros.std-srvs
            ros.ros2cli
            protobuf
            grpc
            abseil-cpp
            abseil-cpp.dev
            re2
            re2.dev
            c-ares
            c-ares.dev
            zlib
            zlib.dev
            icu
            icu.dev
            zstd
            zstd.out
          ];
          shellHook = ''
            export ROS_DISTRO=humble
            export ROS_VERSION=2
            # Keep CMake's gRPC, protobuf, and Abseil package discovery on the
            # same nixpkgs revision.  A host /usr Abseil config can otherwise
            # be selected ahead of nix gRPC and produce link-time ABI errors.
            export CMAKE_PREFIX_PATH="${pkgs.abseil-cpp.dev}/lib/cmake/absl:${pkgs.protobuf}/lib/cmake/protobuf:${pkgs.grpc}/lib/cmake/grpc:${pkgs.re2.dev}/lib/cmake/re2:''${CMAKE_PREFIX_PATH:-}"
            export LD_LIBRARY_PATH="${pkgs.zstd.out}/lib:''${LD_LIBRARY_PATH:-}"
            echo "onnx-remote ROS2 shell: build with -DONNXSIM_REMOTE_ROS2=ON"
          '';
        };
      });
}
