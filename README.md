# Hair Coverage Assistant

这是一个用于记录同一部位护理前后“可见毛发覆盖率”变化的 Streamlit 演示应用。

## 本地运行

建议使用 Python 3.12：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## 部署到 Streamlit Community Cloud

1. 将整个项目上传到 GitHub 仓库，包含 `models/deeplab_v3.tflite`。
2. 打开 [Streamlit Community Cloud](https://share.streamlit.io/)，使用 GitHub 登录。
3. 创建应用并选择仓库中的 `app.py`。
4. 在 Advanced settings 中选择 Python 3.12，然后部署。

应用仅比较照片中可见毛发的图像覆盖率变化，不用于皮肤诊断或医疗判断。
