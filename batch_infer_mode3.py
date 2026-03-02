#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys


def ask(prompt: str, default: str | None = None) -> str:
    if default:
        text = input(f"{prompt} [{default}]: ").strip()
        return text or default
    return input(f"{prompt}: ").strip()


def main() -> None:
    txt_path = input("请输入包含以空格分割 npz 文件路径的 txt 文件路径: ").strip()
    if not txt_path:
        print("未输入任何路径，退出")
        return
    txt_p = Path(txt_path)
    if not txt_p.is_file():
        print(f"文件不存在或不是文件: {txt_p}，退出")
        return
    content = txt_p.read_text(encoding="utf-8").strip()
    if not content:
        print("txt 文件为空，退出")
        return
    npz_paths = content.split()

    mode = ask("推理后端（onnx/pth）", "onnx")
    if mode not in ("onnx", "pth"):
        print("无效的 mode，必须是 'onnx' 或 'pth'。")
        return

    model = ask("模型路径（.onnx 或 .pth）")
    if not model or not Path(model).is_file():
        print("模型文件不存在，退出")
        return

    config = ask("可选 config 路径（按 Enter 跳过）", "")
    output_dir = ask("可选输出目录（按 Enter 使用 npz 原目录）", "")
    outdir = Path(output_dir) if output_dir else None
    if outdir and not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).parent / "infer_mode3.py"
    if not script.is_file():
        print(f"未找到 infer_mode3.py（期望位置: {script}），请确认文件存在")
        return

    for p in npz_paths:
        npz_p = Path(p)
        if not npz_p.is_file():
            print(f"跳过（不存在）：{npz_p}")
            continue

        cmd = [sys.executable, script.as_posix(), "--mode", mode, "--model", Path(model).as_posix(), "--npz", npz_p.as_posix()]
        if config:
            cmd += ["--config", config]
        if outdir:
            suffix = "_mode3_onnx" if mode == "onnx" else "_mode3"
            out_name = f"inference_{npz_p.stem}{suffix}.npz"
            cmd += ["--output", str(outdir / out_name)]

        print("运行:", " ".join(cmd))
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"失败: {npz_p} (code {res.returncode})")
        else:
            print(f"完成: {npz_p}")


if __name__ == "__main__":
    main()
