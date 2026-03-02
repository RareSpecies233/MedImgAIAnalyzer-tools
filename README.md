# MedImgAIAnalyzer 的 小工具合集
(开发环境：在macOS上使用uv管理的python)

- [cpp 写的 npz 转 glb「3D 模型」工具](cppglb)（源码：`npz_to_glb.cpp`，可执行文件：`npz_to_glb`），请使用 [make-macOS.sh](cppglb/make-macOS.sh) 在 macOS 上编译
- [cpp 写的 dcm/nii/png 与 npz 互转工具](cppImgCvt)（源码：`main.cpp`，可执行文件：`imgcvt`），请使用 [make-macOS.sh](cppImgCvt/make-macOS.sh) 在 macOS 上编译
- [cpp 写的使用 ONNX 模型推理 npz 工具](cpponnx)（源码：`inference_onnx.cpp`，可执行文件：`inference_onnx`），请使用 [buildonnxcpp.sh](cpponnx/buildonnxcpp.sh) 在 macOS 上编译
- [U-SAM 项目](U-SAM)：部分脚本会复用其中代码与权重
- [npz 的 PyTorch 推理工具](infer_pth.py)
- [npz 的 ONNX 推理工具](infer_onnx.py)
- [pth 导出 ONNX 工具](export_onnx.py)
- [推理辅助工具](usam_infer_utils.py)（[infer_onnx.py](infer_onnx.py)与[infer_pth.py](infer_pth.py)用到此文件）
- [推理参数文件示例](inference_params.json)
- [npz 阅读器](npzReader.py)
- [npz 3D Web 查看器](npz_to_web.py)
- [glb Web 查看器](glb_to_web.py)
- [医疗影像格式转换器（dcm/nii/png/npz）](medical_image_converter.py)
- [cpp 写的 npz2png 转换器](npz2png.cpp)，请使用 [buildmacOS.sh](buildmacOS.sh) 在 macOS 上编译
- [通用 cpp 编译脚本](buildmacOS.sh)
- [PyTorch 环境检测脚本](ifpytorch.py)
- [支持prompt mode 3的onnx/pth推理脚本](New_infer_mode3.py)
- [多个GLB转为html 支持同步旋转缩放](pack_glb_to_html.py)


## medical_image_converter.py 用法

脚本支持以下 5 种转换：

1. dcm 转 npz
2. nii 转 npz
3. png 转 npz
4. npz 转 dcm
5. npz 转 nii

`* -> npz` 输出结构与 `infer_onnx.py` 兼容，固定包含：`image` 和 `label` 两个键。

示例：

```bash
# 1) dcm -> npz
python medical_image_converter.py --mode dcm2npz --input ./a.dcm --output ./a.npz

# 2) nii -> npz（3D nii 默认取中间切片，可用 --slice-index 指定）
python medical_image_converter.py --mode nii2npz --input ./a.nii.gz --output ./a.npz

# 3) png -> npz（可输入单张 png 或 png 文件夹）
python medical_image_converter.py --mode png2npz --input ./a.png --output ./a.npz

# 4) npz -> dcm（默认读取 npz 的 image 键）
python medical_image_converter.py --mode npz2dcm --input ./a.npz --output ./a.dcm

# 5) npz -> nii
python medical_image_converter.py --mode npz2nii --input ./a.npz --output ./a.nii.gz
```

