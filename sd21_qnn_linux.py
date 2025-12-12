import os
import sys
import shutil
import argparse
import glob

import numpy as np
import torch
from diffusers import UNet2DConditionModel
from diffusers.models.embeddings import get_timestep_embedding
from diffusers import DPMSolverMultistepScheduler
from tokenizers import Tokenizer
from PIL import Image

try:  # in case Python <3.9
    from typing import List
except ImportError:
    List = list  # type: ignore


# ---------------------------
# Configurable paths
# ---------------------------

# Default QNN binaries and context directories (adjust on device as needed)
QNN_BIN_DIR = os.environ.get("QNN_BIN_DIR", "/qnn_sdk/2.39.0.250926/bin/aarch64-oe-linux-gcc11.2")
QNN_LIB_DIR = os.environ.get("QNN_LIB_DIR", "/qnn_sdk/2.39.0.250926/lib/aarch64-oe-linux-gcc11.2")

# Directory containing qnn-net-run executable and HTP backend library
QNN_BINARIES_PATH = os.environ.get("QNN_BINARIES_PATH", QNN_BIN_DIR)

# Directory containing the three context binaries (text encoder, UNet, VAE)
MODELS_CONTEXT_PATH = os.environ.get(
    "SD21_QNN_CONTEXT_DIR", "/model/build/stable_diffusion_v2_1_w8a16/precompiled/qualcomm-qcs9075-proxy/"
)


# ---------------------------
# Helper: run qnn-net-run
# ---------------------------


def run_qnn_net_run(model_context: str, input_data_list):
    """Execute a QNN context on Linux using qnn-net-run.

    model_context: path to the context binary (.serialized.bin or .bin)
    input_data_list: list of numpy arrays as inputs
    """

    tmp_dirpath = os.path.abspath("tmp")
    os.makedirs(tmp_dirpath, exist_ok=True)

        # Write input raw files and build input_list.txt
    input_list_lines = []
    for idx, input_data in enumerate(input_data_list):
        raw_path = os.path.join(tmp_dirpath, f"input_{idx}.raw")

        input_data = np.ascontiguousarray(input_data)
        input_data.tofile(raw_path)
        input_list_lines.append(raw_path)

    input_list_path = os.path.join(tmp_dirpath, "input_list.txt")
    with open(input_list_path, "w") as f:
        f.write(" ".join(input_list_lines))

    qnn_net_run = os.path.join(QNN_BINARIES_PATH, "qnn-net-run")
    backend_lib = os.path.join(QNN_LIB_DIR, "libQnnHtp.so")

    log_path = os.path.join(tmp_dirpath, "log.txt")

    cmd = (
        f"\"{qnn_net_run}\" "
        f"--retrieve_context \"{model_context}\" "
        f"--backend \"{backend_lib}\" "
        f"--input_list \"{input_list_path}\" "
        f"--output_dir \"{tmp_dirpath}\" "
        f"--log_level verbose "
        f"> \"{log_path}\" 2>&1"
    )

    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"qnn-net-run failed, see log: {log_path}")

        # Use glob to locate the single .raw output file
    output_dir = os.path.join(tmp_dirpath, "Result_0")
    output_files = glob.glob(os.path.join(output_dir, "*.raw"))

    if len(output_files) == 0:
        raise FileNotFoundError(f"QNN output .raw file not found in {output_dir}")
    if len(output_files) > 1:
        raise FileNotFoundError(
            f"Found multiple .raw files in {output_dir}, unsure which to use: {output_files}"
        )

    out_path = output_files[0]
    print(f"Found QNN output file: {out_path}")
    output = np.fromfile(out_path, dtype=np.float32)

    shutil.rmtree(tmp_dirpath)

    return output


# ---------------------------
# QNN wrappers for 3 components
# ---------------------------


def run_text_encoder(token_ids: np.ndarray, hidden_size: int) -> np.ndarray:
    """Run the text_encoder QNN context.

    token_ids: shape (77,) or (1, 77)
    returns: shape (1, 77, hidden_size)
    """

    if token_ids.ndim == 1:
        token_ids = token_ids[None, :]
    token_ids = token_ids.astype(np.float32)

    #ctx = os.path.join(MODELS_CONTEXT_PATH, "stable_diffusion_v2_1-text_encoder-qualcomm_sa8775p.bin")
    ctx = os.path.join(MODELS_CONTEXT_PATH, "Stable-Diffusion-v2.1_text_encoder_w8a16.bin")
    out = run_qnn_net_run(ctx, [token_ids])
    out = out.reshape((1, 77, hidden_size))
    return out


def run_unet(latent_in: np.ndarray,
             timestep: np.ndarray,
             text_emb: np.ndarray) -> np.ndarray:
    """Run the U-Net QNN context.

    latent_in:  (1, 64, 64, 4)  NHWC
    timestep:   scalar timestep value
    text_emb:   (1, 77, hidden_size)
    returns:    (1, 64, 64, 4)  NHWC
    """

    #ctx = os.path.join(MODELS_CONTEXT_PATH, "stable_diffusion_v2_1-unet-qualcomm_sa8775p.bin")
    ctx = os.path.join(MODELS_CONTEXT_PATH, "Stable-Diffusion-v2.1_unet_w8a16.bin")
    # Based on the error "input tensor with name timestep and index 0", the model expects timestep first.
    # The standard diffusers UNet has inputs (sample, timestep, encoder_hidden_states, ...),
    # but the exported QNN model might have a different signature.
    # The error also implies timestep is a scalar (4 bytes).
    out = run_qnn_net_run(ctx, [timestep.astype(np.float32), latent_in, text_emb])
    out = out.reshape((1, 64, 64, 4))
    return out


def run_vae(latent_in: np.ndarray) -> np.ndarray:
    """Run the VAE decoder QNN context.

    latent_in: (1, 64, 64, 4) NHWC
    returns: (512, 512, 3) uint8 RGB
    """

    #ctx = os.path.join(MODELS_CONTEXT_PATH, "stable_diffusion_v2_1-vae-qualcomm_sa8775p.bin")
    ctx = os.path.join(MODELS_CONTEXT_PATH, "Stable-Diffusion-v2.1_vae_w8a16.bin")
    out = run_qnn_net_run(ctx, [latent_in])

    out = np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)
    out = out.reshape((512, 512, 3))
    return out


# ---------------------------
# Tokenizer / Scheduler / Time embedding
# ---------------------------


def make_time_embedding_fn(pretrained_model_name: str = "stabilityai/stable-diffusion-2-1-base"):
    """Create the time embedding function.

    Prefer local cache: if cache has content, avoid network access;
    only download via from_pretrained when cache is missing or empty.
    """

    cache_dir = os.path.abspath("./cache/diffusers")

    # If the cache directory already has content, assume the model is cached; avoid network downloads
    local_files_only = False
    if os.path.isdir(cache_dir) and any(os.scandir(cache_dir)):
        local_files_only = True

    # When local_files_only=True and files are missing, from_pretrained raises immediately,
    # ensuring no network attempts in offline environments.
    try:
        time_embeddings = UNet2DConditionModel.from_pretrained(
            pretrained_model_name,
            subfolder="unet",
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        ).time_embedding
    except Exception as e:
        print("Failed to download/load UNet time embeddings.")
        print(f"Error: {e}")
        print("If you are in mainland China, set Hugging Face mirror:")
        print("  export HF_ENDPOINT=https://hf-mirror.com")
        print("  export HF_HUB_ENABLE_HF_TRANSFER=1 (Optional, for faster downloads, but hf-mirror may reject parallel requests)")
        print(f"Then remove {cache_dir} and re-run this script. Alternatively, pre-download the model and set cache_dir.")
        try:
            if os.path.isdir(cache_dir):
                print(f"Removing possibly corrupted cache_dir: {cache_dir}")
                shutil.rmtree(cache_dir)
        except Exception as rm_e:
            print(f"Warning: failed to remove cache_dir, try to remove it({rm_e}) manually if needed.")
        sys.exit(1)

    def get_time_embedding(timestep: int) -> np.ndarray:
        t = torch.tensor([timestep])
        t_emb = get_timestep_embedding(t, 320, True, 0)
        emb = time_embeddings(t_emb).detach().numpy()
        return emb

    return get_time_embedding


def make_tokenizer(max_length: int = 77):
    local_tokenizer_path = os.path.abspath("clip-vit-large-patch14-tokenizer.json")

    if os.path.exists(local_tokenizer_path):
        tokenizer = Tokenizer.from_file(local_tokenizer_path)
    else:
        # Only on first run with network available will it download
        try:
            tokenizer = Tokenizer.from_pretrained("openai/clip-vit-large-patch14")
            tokenizer.save(local_tokenizer_path)
        except Exception as e:
            print("Failed to download/load CLIP tokenizer.")
            print(f"Error: {e}")
            print("If you are in mainland China, set Hugging Face mirror:")
            print("  export HF_ENDPOINT=https://hf-mirror.com")
            print("  export HF_HUB_ENABLE_HF_TRANSFER=1 (Optional)")
            # If download fails, delete the local file (partial writes may corrupt it)
            try:
                if os.path.exists(local_tokenizer_path):
                    print(f"Removing possibly corrupted tokenizer file: {local_tokenizer_path}")
                    os.remove(local_tokenizer_path)
            except Exception as rm_e:
                print(f"Warning: failed to remove tokenizer file: {rm_e}")
            sys.exit(1)

    tokenizer.enable_truncation(max_length)
    tokenizer.enable_padding(pad_id=49407, length=max_length)

    def run_tokenizer(prompt: str) -> np.ndarray:
        token_ids = tokenizer.encode(prompt).ids
        return np.array(token_ids, dtype=np.float32)

    return run_tokenizer


def make_scheduler(user_step: int, guidance_scale: float):
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
    )
    scheduler.set_timesteps(user_step)

    def run_scheduler(noise_pred_uncond: np.ndarray,
                      noise_pred_text: np.ndarray,
                      latent_in: np.ndarray,
                      timestep: int) -> np.ndarray:
        # NHWC -> NCHW
        noise_pred_uncond_nchw = np.transpose(noise_pred_uncond, (0, 3, 1, 2)).copy()
        noise_pred_text_nchw = np.transpose(noise_pred_text, (0, 3, 1, 2)).copy()
        latent_in_nchw = np.transpose(latent_in, (0, 3, 1, 2)).copy()

        noise_pred_uncond_t = torch.from_numpy(noise_pred_uncond_nchw)
        noise_pred_text_t = torch.from_numpy(noise_pred_text_nchw)
        latent_in_t = torch.from_numpy(latent_in_nchw)

        noise_pred = noise_pred_uncond_t + guidance_scale * (
            noise_pred_text_t - noise_pred_uncond_t
        )

        latent_out = scheduler.step(noise_pred, timestep, latent_in_t).prev_sample
        latent_out = latent_out.numpy()
        latent_out = np.transpose(latent_out, (0, 2, 3, 1)).copy()  # back to NHWC
        return latent_out

    def get_timestep(step: int) -> np.int32:
        return np.int32(scheduler.timesteps.numpy()[step])

    return run_scheduler, get_timestep


# ---------------------------
# Main pipeline
# ---------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run Stable Diffusion 2.1 on QNN (Linux) using qnn-net-run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--prompt", type=str,
                        default="decorated modern country house interior, 8 k, light reflections",
                        help="Text prompt")
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of diffusion steps (20 or 50)")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed")
    parser.add_argument("--guidance", type=float, default=7.5,
                        help="Classifier-free guidance scale [5.0, 15.0]")
    parser.add_argument("--output", type=str, default="sd21_qnn_linux.png",
                        help="Output image path")

    args = parser.parse_args()

    assert args.steps in (20, 50), "steps should be 20 or 50"

    # Hyperparams for SD 2.1
    pretrained_model_name = "sd2-community/stable-diffusion-2-1-base"
    hidden_size = 1024

    # Build helpers
    get_time_embedding = make_time_embedding_fn(pretrained_model_name)
    run_tokenizer = make_tokenizer(max_length=77)
    run_scheduler, get_timestep = make_scheduler(args.steps, args.guidance)

    # Tokenize
    print("Running tokenizer...")
    uncond_tokens = run_tokenizer("")
    cond_tokens = run_tokenizer(args.prompt)

    # Text encoder
    print("Running text encoder on QNN...")
    uncond_text_embedding = run_text_encoder(uncond_tokens, hidden_size)
    user_text_embedding = run_text_encoder(cond_tokens, hidden_size)

    # Initialize latent
    print("Initializing latent...")
    torch.manual_seed(args.seed)
    random_init_latent = torch.randn((1, 4, 64, 64)).numpy()
    latent_in = random_init_latent.transpose((0, 2, 3, 1)).copy()  # NCHW -> NHWC

    # Diffusion loop
    for step in range(args.steps):
        print(f"Step {step + 1}/{args.steps}...")
        timestep = get_timestep(step)

        # Time embedding
        t_emb = get_time_embedding(int(timestep))

        # Unconditional / conditional U-Net
        # Pass the scalar timestep as well
        uncond_noise = run_unet(latent_in, timestep, uncond_text_embedding)
        cond_noise = run_unet(latent_in, timestep, user_text_embedding)

        # Scheduler
        latent_in = run_scheduler(uncond_noise, cond_noise, latent_in, int(timestep))

    # VAE decode
    print("Running VAE decoder on QNN...")
    output_image = run_vae(latent_in)

    # Save image
    img = Image.fromarray(output_image, mode="RGB")
    img.save(args.output)
    print(f"Saved image to {args.output}")


if __name__ == "__main__":
    main()
