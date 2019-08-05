{ pkgs ? import <nixpkgs> {} }:

(pkgs.buildFHSUserEnv {
  name = "rust-vulkan-env";
  targetPkgs = pkgs: (with pkgs;
    [
      rustup
      pkg-config
      binutils-unwrapped
      glxinfo
      wayland
      libGL
      vulkan-loader
      xorg.libX11
      xorg.libX11.dev
      xorg.libXcursor
      xorg.libXrandr
      xorg.libxcb
    ]);
  runScript = "bash -c \"export PKG_CONFIG_PATH='${pkgs.xorg.libX11.dev}/lib/pkgconfig:${pkgs.xorg.xorgproto}/share/pkgconfig' && bash\"";
}).env
