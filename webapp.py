import json
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块 (根据实际项目调整)
try:
    from blind_watermark import WaterMark
    has_blind_watermark = True
except ImportError as e:
    has_blind_watermark = False
    print(f"Warning: blind_watermark module not found: {e}")

# 加载配置
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# 创建FastAPI应用
app = FastAPI(
    title="盲水印应用",
    description="提供图片文本盲水印的添加和提取功能",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API路由
@app.get("/")
async def root():
    return {"message": "Welcome to Python Project API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# 示例API端点 (根据实际项目功能调整)
if has_blind_watermark:
    @app.post("/api/add_watermark")
    async def add_watermark(
        image: UploadFile = File(...),
        password: int = 0,
        watermark_text: str = None,
    ):
        try:
            # 保存上传的文件
            with open("temp_image.png", "wb") as f:
                f.write(await image.read())
            
            # 使用盲水印库添加水印
            bwm = WaterMark(password_img=int(password))
            bwm.read_img("temp_image.png")
            
            if watermark_text:
                # 文本水印
                bwm.read_wm(watermark_text, mode='str')
                bwm.embed("output_with_watermark.png")
                len_wm = len(bwm.wm_bit)
                print(len_wm)
                # 先返回图片文件，同时在响应头中包含水印长度信息
                response = FileResponse(
                    path="output_with_watermark.png",
                    media_type="image/png",
                    filename="output_with_watermark.png"
                )
                response.headers["X-WM-Shape"] = str(len_wm)
                return response
            else:
                raise HTTPException(status_code=400, detail="请提供水印文本或水印图片")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/extract_watermark")
    async def extract_watermark(
        image: UploadFile = File(...),
        password: int = 0,
        wm_shape: int = 1,
    ):
        try:
            # 保存上传的文件
            with open("temp_image_with_wm.png", "wb") as f:
                f.write(await image.read())
            
            # 使用盲水印库提取水印
            bwm = WaterMark(password_img=int(password))
            text_water_mark = bwm.extract("temp_image_with_wm.png", wm_shape=wm_shape,mode='str')
            print(text_water_mark)
            return JSONResponse({"status": "success", "output": text_water_mark})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# 创建Gradio界面
def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# 盲水印应用")
        
        if has_blind_watermark:
            with gr.Tab("添加水印"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(label="原始图片")
                        input_text_watermark = gr.Textbox(label="文本水印", placeholder="请输入水印文本")
                        input_password = gr.Number(label="密码", value=1, precision=0)
                        add_button = gr.Button("添加水印")
                    with gr.Column():
                        output_image = gr.Image(label="带水印的图片")
                        output_wm_shape = gr.Textbox(label="水印长度", placeholder="提取水印时需要使用")
                
                def add_watermark_func(image, text_watermark, password):
                    if image is None:
                        return "请上传原始图片", ""
                    
                    if not text_watermark:
                        return "请输入水印文本或上传水印图片", ""
                    
                    try:
                        # 保存图片
                        import cv2
                        import numpy as np
                        if isinstance(image, np.ndarray):
                            # 如果是NumPy数组，检查并转换色彩通道
                            # Gradio传递的是RGB格式，OpenCV需要BGR格式
                            if image.shape[2] == 3:
                                # 转换RGB到BGR
                                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                                cv2.imwrite("temp_image.png", image_bgr)
                            else:
                                cv2.imwrite("temp_image.png", image)
                        else:
                            # 如果是PIL Image对象，使用save方法保存
                            image.save("temp_image.png")
                        
                        # 使用盲水印库
                        bwm = WaterMark(password_img=int(password))
                        bwm.read_img("temp_image.png")
                        bwm.read_wm(text_watermark, mode='str')
                        bwm.embed("output_with_watermark.png")
                        len_wm = len(bwm.wm_bit)
                        return "output_with_watermark.png", str(len_wm)
     
                    except Exception as e:
                        return f"错误: {str(e)}", ""
                
                # 提示信息
                gr.Markdown("### 使用提示")
                gr.Markdown("- 文本水印建议使用1-50个字符的短文本")
                gr.Markdown("- 推荐使用英文和数字，避免使用过多特殊字符")
                gr.Markdown("- 提取时请确保上传的是通过本系统添加水印的图片")
                gr.Markdown("- 提取文本水印时需要使用生成的水印长度")
                
                add_button.click(
                    fn=add_watermark_func,
                    inputs=[input_image, input_text_watermark, input_password],
                    outputs=[output_image, output_wm_shape]
                )
            
            with gr.Tab("提取水印"):
                with gr.Row():
                    with gr.Column():
                        input_image_with_wm = gr.Image(label="带水印的图片")
                        input_password = gr.Textbox(label="密码", placeholder="请输入密码")
                        input_wm_shape = gr.Textbox(label="水印长度", placeholder="添加水印时生成的长度")
                        extract_button = gr.Button("提取水印")
                    with gr.Column():
                        output_text_watermark = gr.Textbox(label="提取的文本水印")
                
                def extract_watermark_func(image, wm_shape, password):
                    if image is None:
                        return "请上传带水印的图片"
                    
                    try:
                        # 保存图片
                        import cv2
                        import numpy as np
                        if isinstance(image, np.ndarray):
                            # 如果是NumPy数组，检查并转换色彩通道
                            # Gradio传递的是RGB格式，OpenCV需要BGR格式
                            if image.shape[2] == 3:
                                # 转换RGB到BGR
                                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                                cv2.imwrite("temp_image_with_wm.png", image_bgr)
                            else:
                                cv2.imwrite("temp_image_with_wm.png", image)
                        else:
                            # 如果是PIL Image对象，使用save方法保存
                            image.save("temp_image_with_wm.png")
                        
                        # 使用盲水印库
                        bwm = WaterMark(password_img=int(password))
                        
                        if not wm_shape:
                            return "请输入水印长度"
                        try:
                            # 提取文本水印
                            wm_extract = bwm.extract("temp_image_with_wm.png", wm_shape=int(wm_shape), mode='str')
                            if wm_extract:
                                # 清理提取的文本，去除不可见字符
                                cleaned_str = ''.join(char for char in wm_extract if char.isprintable())
                                if cleaned_str:
                                    return cleaned_str
                            return f"提取失败: {wm_extract}"
                        except Exception as e:
                            return f"提取失败: {str(e)}"

                    except Exception as e:
                        return f"提取失败: {str(e)}"

                
                extract_button.click(
                    fn=extract_watermark_func,
                    inputs=[input_image_with_wm, input_wm_shape, input_password],
                    outputs=[output_text_watermark]
                )
        else:
            gr.Markdown("## 未检测到盲水印模块")
            gr.Markdown("请确保项目模块已正确导入")
    
    return demo

# 挂载Gradio到FastAPI
gradio_app = create_gradio_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/webapp")

# 启动应用
if __name__ == "__main__":
    uvicorn.run(
        "webapp:app",
        host=config.get("host", "0.0.0.0"),
        port=config.get("port", 8774),
        reload=True
    )