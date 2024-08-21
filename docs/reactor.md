# ReActor

Stable Diffusion ReActor 插件可以完成图片换脸(swap face)

代码仓库位于：[github.com/Gourieff/sd-webui-reactor](https://github.com/Gourieff/sd-webui-reactor)

分析它的代码可知：  

* 面部识别使用 insightface 提供的预训练模型 buffalo_l。核心代码位于 ./scripts/reactor_swapper.py:  

```Python
# line 142-158, 人脸检测使用 insightface 预训练的 buffalo_l
def getAnalysisModel():
    global ANALYSIS_MODEL
    if ANALYSIS_MODEL is None:
        ANALYSIS_MODEL = insightface.app.FaceAnalysis(
            name="buffalo_l", providers=PROVIDERS, root=os.path.join(models_path, "insightface") # note: allowed_modules=['detection', 'genderage']
        )
    return ANALYSIS_MODEL

# 换脸默认使用 insightface 预训练的 inswapper_128.onnx
def getFaceSwapModel(model_path: str):
    global FS_MODEL
    global CURRENT_FS_MODEL_PATH
    if CURRENT_FS_MODEL_PATH is None or CURRENT_FS_MODEL_PATH != model_path:
        CURRENT_FS_MODEL_PATH = model_path
        FS_MODEL = insightface.model_zoo.get_model(model_path, providers=PROVIDERS)

    return FS_MODEL

# line 300-304, 
def analyze_faces(img_data: np.ndarray, det_size=(640, 640), det_thresh=0.5, det_maxnum=0):
    logger.info("Applied Execution Provider: %s", PROVIDERS[0])
    face_analyser = copy.deepcopy(getAnalysisModel())
    face_analyser.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)
    return face_analyser.get(img_data, max_num=det_maxnum)
```

* 换脸使用 insightface 预训练的 inswapper_128.onnx，代码调用顺序为：  
    。/scripts/reactor_api.py: 用FastAPI封装的API接口 reactor_api -> swap_face
    ./scripts/reactor_swapper.py: swap_face -> operate

* upscale

* face restore  
