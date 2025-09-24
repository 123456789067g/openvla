from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
import io
 
# ========== 初始化模型 ==========
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
print(f"👉 使用设备: {device}, 数据类型: {dtype}")
 
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device)
 
# ========== 建 API 应用 ==========
app = FastAPI()
 
@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    # 1. 读取上传的图像
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
 
    # 2. Prompt（你可以换成自己的）
    prompt = "In: Catch the black object\nOut:"
 
    # 3. 处理输入
    inputs = processor(prompt, image).to(device, dtype=dtype)
 
    # 4. 模型推理
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
 
    # 5. 返回结果
    return {"predicted_action": action}