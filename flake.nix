{
  description = "Python development environment with uv";
  inputs = {
    # This points to the nixpkgs version you want to use
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };
  outputs = {
    self,
    nixpkgs,
  }: let
    # Systems supported (adjust if you use Mac/ARM)
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
    # Define your libraries in one place to avoid repetition
    libs = with pkgs; [
      stdenv.cc.cc.lib
      zlib
      glib
      libGL
    ];
  in {
    devShells.${system}.default = pkgs.mkShell {
      buildInputs = libs ++ [pkgs.uv]; # Adding uv explicitly
      shellHook = ''
        export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath libs}:$LD_LIBRARY_PATH
        
        # Add project root to PYTHONPATH
        export PYTHONPATH="$PWD:$PYTHONPATH"
        
        # Auto-create venv if it doesn't exist
        if [ ! -d .venv ]; then
          uv venv
        fi
        source .venv/bin/activate
      '';
    };
  };
}
