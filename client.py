# client.py
import requests
import cv2
import json
 
API_URL = "http://127.0.0.1:8000/infer"
 
def capture_and_send(instruction: str):
    # 打开相机
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开相机")
        return None
 
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("❌ 拍照失败")
        return None
 
    # 保存临时图片
    img_path = "frame.jpg"
    cv2.imwrite(img_path, frame)
 
    # 上传到 API
    with open(img_path, "rb") as f:
        files = {"file": f}
        data = {"instruction": instruction}
        resp = requests.post(API_URL, files=files, data=data)
 
    if resp.status_code == 200:
        result = resp.json()
        print("✅ 模型返回:", result)
 
        # 保存到本地 JSON 文件，供 translate.py 使用
        with open("action.json", "w") as f:
            json.dump(result, f)
 
        return result
    else:
        print("❌ 请求失败:", resp.status_code, resp.text)
        return None
 
if __name__ == "__main__":
    instruction = "pick up the red block"  # 示例任务
    capture_and_send(instruction)