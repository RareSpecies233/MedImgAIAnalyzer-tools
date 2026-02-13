# MedImgAIAnalyzer 的 小工具合集
(开发环境：在macOS上使用uv管理的python)

- [cpp写的 npz转glb「3d模型」工具](cppglb) ，请使用[make-macOS.sh](cppglb/make-macOS.sh)在macOS上编译
- [cpp写的 dcm/nii/png转npz转dcm/nii工具](cppImgCvt)，请使用[make-macOS.sh](cppImgCvt/make-macOS.sh)在macOS上编译
- [cpp写的 使用onnx模型推理npz的工具](cpponnx)，请使用[buildonnxcpp.sh](cpponnx/buildonnxcpp.sh)在macOS上编译
- [U-SAM项目](U-SAM) 有一些脚本会用到其中的部分代码
- [npz 推理工具](infer_pth.py)
- [npz 使用onnx的推理工具](infer_onnx.py)
- [pth 导出 onnx 工具](export_onnx.py)
- [上面三者的辅助工具](usam_infer_utils.py)
- [npz 阅读器](npzReader.py)
- [npz 3D 查看器](npz_to_web.py)
- [glb 转换为 web 查看器](glb_to_web.py)
- [医疗影像格式转换器（dcm/nii/png/npz）](medical_image_converter.py)
- [cpp写的 npz2png 转换器](npz2png.cpp)，请使用[buildmacOS.sh](buildmacOS.sh)在macOS上编译
- [cpp编译工具](buildmacOS.sh)
- [pytorch环境监测脚本](ifpytorch.py)

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

