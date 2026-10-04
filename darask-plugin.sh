#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ ${1:-} == --help ]]; then
    cat <<'HELP'
Darask Paint ai-diffusion Linux launcher
  --setup-only  Install/validate dependencies without starting the server.
  DARASK_DATA_DIR          Override this plugin's writable data directory.
  DARASK_TORCH_BACKEND     cpu (default) or cu128 (NVIDIA CUDA 12.8).
  DARASK_DOWNLOAD_MODEL    1: download default model; 0: skip (Diffusion only).
On NixOS use `nix run .`; this script also enters the Nix environment automatically.
HELP
    exit 0
fi

if [[ -f /etc/NIXOS && ${DARASK_NIX_ENV:-} != 1 ]]; then
    cd -- "$script_dir"
    exec nix run . -- "$@"
fi

setup_only=0
if [[ ${1:-} == --setup-only ]]; then
    setup_only=1
    shift
fi
backend=${DARASK_TORCH_BACKEND:-cpu}
case "$backend" in
    cpu|cu128) ;;
    *) echo 'DARASK_TORCH_BACKEND must be cpu or cu128' >&2; exit 1 ;;
esac
data_home=${XDG_DATA_HOME:-}
if [[ $data_home != /* ]]; then data_home="$HOME/.local/share"; fi
app_dir=${DARASK_DATA_DIR:-$data_home/darask-paint-ai-diffusion}
mkdir -p -- "$app_dir"
app_dir=$(cd -- "$app_dir" && pwd)
exec 9>"$app_dir/launcher.lock"
if ! flock -n 9; then
    echo "This plugin is already running or being installed: $app_dir" >&2
    exit 1
fi

# Use Nix's interpreter, never a downloaded FHS-linked Python on NixOS.
python_bin=${DARASK_PYTHON:-python3.12}
export UV_PYTHON_DOWNLOADS=never
export UV_PYTHON_PREFERENCE=only-system
venv="$app_dir/env"
python="$venv/bin/python"
marker="$app_dir/.darask-linux-setup"
expected="$(sha256sum "$script_dir/darask-plugin.sh" | cut -d' ' -f1);python=$(command -v "$python_bin");backend=$backend"
constraints="$app_dir/torch-constraints.txt"
printf '%s
' 'torch==2.11.0' 'torchvision==0.26.0' 'torchaudio==2.11.0' > "$constraints"

comfy_version=a95e461916de9cbda2e89140ab86a8a7c3f9702a
comfy_dir="$app_dir/ComfyUI-$comfy_version"
models="$app_dir/models/checkpoints"
mkdir -p -- "$models"
if [[ ! -x $python || ! -f $comfy_dir/main.py || $(cat "$marker" 2>/dev/null || true) != "$expected" ]]; then
    rm -f -- "$marker"
    uv venv --clear --python "$python_bin" "$venv"
    if [[ ! -f $comfy_dir/main.py ]]; then
        archive="$app_dir/comfyui.tar.gz"
        curl --fail --location --retry 3 \
            "https://github.com/comfyanonymous/ComfyUI/archive/$comfy_version.tar.gz" -o "$archive.part"
        mv -- "$archive.part" "$archive"
        tar -xzf "$archive" -C "$app_dir"
        rm -- "$archive"
    fi
    uv pip install --python "$python" --torch-backend "$backend" -c "$constraints" \
        torch torchvision torchaudio -r "$comfy_dir/requirements.txt"
    "$python" -I -c 'import torch, torchvision, torchaudio, av, safetensors; print("PyTorch:", torch.__version__)'
    printf '%s' "$expected" > "$marker"
fi
# Keep user checkpoints outside the versioned source tree during upgrades.
if [[ ! -L $comfy_dir/models/checkpoints/darask ]]; then
    ln -s "$models" "$comfy_dir/models/checkpoints/darask"
fi
if ! find "$models" "$comfy_dir/models/checkpoints" -type f \
    \( -name '*.safetensors' -o -name '*.ckpt' \) -print -quit | grep -q .; then
    download=${DARASK_DOWNLOAD_MODEL:-}
    if [[ -z $download && -t 0 ]] && (( ! setup_only )); then
        read -r -p 'Download DreamShaper 8 (SD1.5, about 2 GB)? [Y/n] ' answer
        case "$answer" in n|N) download=0 ;; *) download=1 ;; esac
    fi
    if [[ $download == 1 ]]; then
        model="$models/DreamShaper_8_pruned.safetensors"
        curl --fail --location --retry 3 \
            'https://huggingface.co/Lykon/DreamShaper/resolve/228d79cb20811466f5c5710aa91f05dabd0b8a14/DreamShaper_8_pruned.safetensors' \
            -o "$model.part"
        mv -- "$model.part" "$model"
    else
        echo "Put an SD1.5/SDXL checkpoint in $models (or set DARASK_DOWNLOAD_MODEL=1)."
    fi
fi
if (( setup_only )); then exit 0; fi
comfy_args=(--comfy-arg=--cpu)
if [[ $backend == cu128 ]]; then
    "$python" -I -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable; check your driver or use DARASK_TORCH_BACKEND=cpu"'
    comfy_args=()
fi
echo 'AI Diffusion: http://127.0.0.1:8424 — Ctrl+C to stop'
exec "$python" "$script_dir/darask_server.py" --port 8424 --comfy-port 8188 \
    --comfy-python "$python" --comfy-main "$comfy_dir/main.py" \
    --comfy-log "$app_dir/comfyui.log" "${comfy_args[@]}" "$@"
