# Stable diffusion 2.1 deployment on QCS9075
Stable Diffusion is a widely adopted text-to-image generative model, enabling high-quality image synthesis from natural language prompts. Deploying Stable Diffusion efficiently on edge devices remains challenging due to its large model size, multi-stage architecture, and high computational requirements.

This demo demonstrates on-device deployment and execution of Stable Diffusion 2.1 on the Qualcomm QCS9075 platform using Qualcomm AI Runtime (QNN).

The Stable Diffusion pipeline is split into three pre-compiled model context files: text encoder, UNet, and VAE. These model binaries are either exported using Qualcomm AI Hub tools or downloaded from Hugging Face with version alignment to the target QNN SDK. The demo runs fully on-device and generates images from text prompts without relying on cloud inference.

The project highlights the complete workflow for generative AI deployment on Qualcomm platforms, including model context preparation, QNN SDK integration, Python runtime setup, and end-to-end text-to-image inference.

## Export or download the pre-compiled model context file

There are 2 ways to get the model files.

### Method 1 - Export by QC's python Script.

Clone https://github.com/quic/ai-hub-models on PC and export the pre-compiled model.


```powershell
PowerShell  
cmd
D:\py\venv\Scripts\Activate.ps1
python -m qai_hub_models.models.stable_diffusion_v2_1.export --target-runtime precompiled_qnn_onnx  --device "QCS9075" --fetch-static-assets v0.39.1
```

### Method 2 - Download from Huggingface

Download from Huggingface, and find the corresponding version.

Pay attention to the model version, such as v0.39.1. We need to find the QNN SDK version information corresponding to this model from the tool-versions.yaml on the download page or from the model context file, and the QNN SDK version on QCS9100 must be at least consistent with it.

If the version in tool-versions.yaml is qairt: 2.38.0.250901140452_125126-auto, the target machine must have at least V2.38.0 installed.

Version information can also be found in the model file under Linux using the strings command:

```bash
Shell  
cmd
strings Stable-Diffusion-v2.1_text_encoder_w8a16.bin | grep "v2."
```

```text
v2.39.0.250925215840_163802-auto
Vrv2.39.0.250925215840_163802-auto.fcaeba5a50
Vrv2.39.0.250925215840_163802-auto.fcaeba5a50
```

Additionally, we can also build/convert the model context files by ourselves following https://qpm.qualcomm.com/#/main/tools/details/Tutorial_for_Stable_Diffusion_2_1_Compute

Three model context files will be downloaded: text-encoder, unet and vae :

```text
Stable-Diffusion-v2.1_text_encoder_w8a16.bin
Stable-Diffusion-v2.1_unet_w8a16.bin
Stable-Diffusion-v2.1_vae_w8a16.bin
```

## Install QNN SDK

Select the right version of QNN SDK as above, and install it from https://qpm.qualcomm.com/#/main/tools/details/Qualcomm_AI_Runtime_SDK on a host PC. After the SDK is installed, copy the SDK binaries and binaries to the QCS9075/QCS9100 target device as folowing structure.

```text
2.39.0.250926/
|-- bin
|   `-- aarch64-oe-linux-gcc11.2
|       |-- genie-app
|       |-- genie-t2e-run
|       |-- genie-t2t-run
|       |-- qnn-context-binary-generator
|       |-- qnn-context-binary-utility
|       |-- qnn-net-run
|       |-- qnn-platform-validator
|       |-- qnn-profile-viewer
|       |-- qnn-throughput-net-run
|       |-- qtld-net-run
|       |-- snpe-diagview
|       |-- snpe-net-run
|       |-- snpe-parallel-run
|       |-- snpe-platform-validator
|       `-- snpe-throughput-net-run
`-- lib
    |-- aarch64-oe-linux-gcc11.2
    |   |-- libGenie.so
    |   |-- libPlatformValidatorShared.so
    |   |-- libQnnChrometraceProfilingReader.so
    |   |-- libQnnCpu.so
    |   |-- libQnnCpuNetRunExtensions.so
    |   |-- libQnnDsp.so
    |   |-- libQnnDspNetRunExtensions.so
    |   |-- libQnnDspV66CalculatorStub.so
    |   |-- libQnnDspV66Stub.so
    |   |-- libQnnGenAiTransformer.so
    |   |-- libQnnGenAiTransformerCpuOpPkg.so
    |   |-- libQnnGenAiTransformerModel.so
    |   |-- libQnnGpu.so
    |   |-- libQnnGpuNetRunExtensions.so
    |   |-- libQnnGpuProfilingReader.so
    |   |-- libQnnHta.so
    |   |-- libQnnHtaNetRunExtensions.so
    |   |-- libQnnHtp.so
    |   |-- libQnnHtpNetRunExtensions.so
    |   |-- libQnnHtpOptraceProfilingReader.so
    |   |-- libQnnHtpPrepare.so
    |   |-- libQnnHtpProfilingReader.so
    |   |-- libQnnHtpV68CalculatorStub.so
    |   |-- libQnnHtpV68Stub.so
    |   |-- libQnnHtpV69CalculatorStub.so
    |   |-- libQnnHtpV69Stub.so
    |   |-- libQnnHtpV73CalculatorStub.so
    |   |-- libQnnHtpV73Stub.so
    |   |-- libQnnHtpV75CalculatorStub.so
    |   |-- libQnnHtpV75Stub.so
    |   |-- libQnnHtpV79CalculatorStub.so
    |   |-- libQnnHtpV79Stub.so
    |   |-- libQnnIr.so
    |   |-- libQnnJsonProfilingReader.so
    |   |-- libQnnModelDlc.so
    |   |-- libQnnSaver.so
    |   |-- libQnnSystem.so
    |   |-- libQnnTFLiteDelegate.so
    |   |-- libSNPE.so
    |   |-- libSnpeDspV66Stub.so
    |   |-- libSnpeHta.so
    |   |-- libSnpeHtpPrepare.so
    |   |-- libSnpeHtpV68CalculatorStub.so
    |   |-- libSnpeHtpV68Stub.so
    |   |-- libSnpeHtpV73CalculatorStub.so
    |   |-- libSnpeHtpV73Stub.so
    |   |-- libSnpeHtpV75CalculatorStub.so
    |   |-- libSnpeHtpV75Stub.so
    |   |-- libSnpeHtpV79CalculatorStub.so
    |   |-- libSnpeHtpV79Stub.so
    |   |-- libcalculator.so
    |   |-- libhta_hexagon_runtime_qnn.so
    |   `-- libhta_hexagon_runtime_snpe.so
    `-- hexagon-v73
        `-- unsigned
            |-- libCalculator_skel.so
            |-- libQnnHtpV73.so
            |-- libQnnHtpV73QemuDriver.so
            |-- libQnnHtpV73Skel.so
            |-- libQnnSaver.so
            |-- libQnnSystem.so
            |-- libSnpeHtpV73Skel.so
            |-- libqnnhtpv73.cat
            `-- libsnpehtpv73.cat
```

## Python setup

If your device has pip3, please ignore this section.

Since only a minimal python3 is integrated in our system([Qualcomm Linux](https://docs.qualcomm.com/doc/RNO-250630224842/topic/ReleaseNote.html?product=895724676033554725&version=1.5)), we have to install a full-fledged Python build, and pip3 is required as well.

Download python3.12.9 and python3.12.9-lib rpm package and extract files from them on a Linux host:

```bash
# Shell cmd on host
wget https://yum.oracle.com/repo/OracleLinux/OL9/appstream/aarch64/getPackage/python3.12-3.12.9-1.el9.aarch64.rpm
wget https://yum.oracle.com/repo/OracleLinux/OL9/appstream/aarch64/getPackage/python3.12-libs-3.12.9-1.el9.aarch64.rpm
#extract files
rpm2cpio python3.12-3.12.9-1.el9.aarch64.rpm | cpio -idmv 
rpm2cpio python3.12-libs-3.12.9-1.el9.aarch64.rpm | cpio -idmv
```

#a usr directory will be extracted,copy it on the device
```bash
scp -r usr/bin/* usr@device-ip:/usr/bin/
scp -r usr/include/* usr@devcie-ip:/usr/include/
scp -r usr/share/* usr@device-ip:/usr/share
scp -r usr/lib/*  usr@device-ip:/usr/lib
scp -r usr/lib64/* usr@device-ip:/usr/lib
```

Link a lib64 directory on the device


```bash
ln -s /usr/lib /usr/lib64
ldconfig
```

Install pip3 on the device

```bash
# Shell cmd
wget https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
#check pip3
pip3 --version
```

Create a virtual environment(venv) and activate it:

```text  
python3 -m venv sd21-project-env
source sd21-project-env/bin/activate
```

Install python modules by the runner script in the venv


```bash
# Shell cmd
pip install diffusers==0.35.2 \
            numpy==2.3.5 \
            pillow==12.0.0 \
            tokenizers==0.22.1 \
            torch==2.9.1 \
            transformers==4.57.1 \
            accelerate \
            transfer \
            -i https://pypi.tuna.tsinghua.edu.cn/simple #optional for China mainland
```

## Run Generative script

Clone or copy this project on the device


```bash
#In China mainland, use a huggingface mirror site instead
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1  #Optional, may speed up the download in first execution
                                   #but has a high chance to be reject by hf-mirror.com

python3 sd21_qnn_linux.py --prompt "A kitten is practicing martial arts" --steps 20 --seed 1 --guidance 7.5 --output sd21_qnn.png
```

The first execution will take a long time to download some resources from huggingface or the mirror site.

The above steps were executed and verified on [Qualcomm Linux](https://docs.qualcomm.com/doc/RNO-250630224842/topic/ReleaseNote.html?product=895724676033554725&version=1.5).


You will generate below picture by
```
python3 sd21_qnn_linux.py --prompt "A kitten is practicing martial arts" --steps 20 --seed 1 --guidance 7.5 --output sd21_qnn.png
```

![](sd21_qnn.png)
