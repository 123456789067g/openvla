from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch

# ========== 可选：支持 HEIC ==========
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("⚠️ 没有安装 pillow-heif，HEIC 可能打不开。运行: pip install pillow-heif")

# ========== 设备选择 ==========
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
print(f"👉 使用设备: {device}, 数据类型: {dtype}")

# ========== 加载模型 ==========
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device)

# ========== 加载照片 ==========
image_path = "picture/IMG_2581.HEIC"   # 修改成你自己的路径
image = Image.open(image_path)

# ========== Prompt ==========
prompt = "In: What action should the robot take to pick up the object?\nOut:"

# ========== 处理输入 ==========
inputs = processor(prompt, image).to(device, dtype=dtype)

# ========== 推理 ==========
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
print("Predicted action:", action)
