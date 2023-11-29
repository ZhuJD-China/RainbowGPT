import tkinter as tk
import uuid
import pyperclip  # 用于复制到剪贴板

def get_mac_address():
    mac_address = get_formatted_mac()
    mac_address_label.config(text="MAC地址: " + mac_address)
    pyperclip.copy(mac_address)

def get_formatted_mac():
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
    return ":".join([mac[i:i+2] for i in range(0, 12, 2)])

# 创建主窗口
root = tk.Tk()
root.title("MAC地址获取工具")
root.geometry("300x150")  # 设置窗口大小

# 创建标签来显示MAC地址
mac_address_label = tk.Label(root, text="MAC地址: ")
mac_address_label.pack(pady=20)

# 创建获取MAC地址的按钮
get_mac_button = tk.Button(root, text="获取MAC地址并复制", command=get_mac_address)
get_mac_button.pack()

root.mainloop()
