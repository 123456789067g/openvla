from fastapi import FastAPI, UploadFile, File, Form
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import uvicorn
import io
 
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
 
# ========== 加载模型（只初始化一次） ==========
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device)
 
# ========== FastAPI 应用 ==========
app = FastAPI()
 
@app.post("/infer")
async def infer(
    file: UploadFile = File(...),
    instruction: str = Form(...)
):
    """
    上传图片 + 指令，返回 OpenVLA 动作结果
    """
    # 1. 读取上传的图片
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
 
    # 2. 构造 prompt
    prompt = f"In: {instruction}\nOut:"
 
    # 3. 模型处理
    inputs = processor(prompt, image).to(device, dtype=dtype)
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
 
    # 4. 返回 JSON
    return {"action": action.tolist()}  # 转 list 便于 JSON 序列化
 
# ========== 启动入口 ==========
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)