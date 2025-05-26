import subprocess

def has_nvidia_smi():
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def main():
    base_packages = [
        "streamlit==1.24.0",
        "matplotlib==3.7.1",
        "pillow==10.0.0",
        "seaborn==0.12.2",
        "pandas==2.0.3",
        "numpy==1.24.4",
    ]

    torch_version = "2.1.0"
    if has_nvidia_smi():
        # Variante GPU (CUDA 11.8)
        torch_pkgs = [
            f"torch=={torch_version}+cu118",
            f"torchvision==0.15.2+cu118",
            f"torchaudio==2.1.0+cu118",
        ]
        extra_index = "--find-links https://download.pytorch.org/whl/cu118"
    else:
        # Variante CPU
        torch_pkgs = [
            f"torch=={torch_version}+cpu",
            f"torchvision==0.15.2+cpu",
            f"torchaudio==2.1.0+cpu",
        ]
        extra_index = "--find-links https://download.pytorch.org/whl/cpu"

    with open("requirements.txt", "w") as f:
        for pkg in base_packages + torch_pkgs:
            f.write(pkg + "\n")
        f.write(extra_index + "\n")

if __name__ == "__main__":
    main()
