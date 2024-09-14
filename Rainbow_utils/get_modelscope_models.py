from modelscope import snapshot_download

# 指定要下载到的目录
target_directory = './models'

# 下载模型到指定目录
model_dir = snapshot_download('qwen/Qwen2-72B', cache_dir=target_directory)

print(f'Model downloaded to: {model_dir}')
