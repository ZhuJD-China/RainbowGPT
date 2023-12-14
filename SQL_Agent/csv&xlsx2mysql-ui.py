import os
import pandas as pd
import pymysql
import tkinter as tk
from tkinter import filedialog, Label, Entry, Text, Scrollbar, Button, N, S, E, W

def pandas_type_to_sql(type_name):
    if 'int' in type_name:
        return 'INT'
    elif 'float' in type_name:
        return 'FLOAT'
    elif 'datetime' in type_name:
        return 'DATETIME'
    else:
        return 'VARCHAR(255)'

def append_text(widget, text):
    widget.configure(state='normal')
    widget.insert(tk.END, text + '\n')
    widget.configure(state='disabled')
    widget.see(tk.END)

def load_data(file_path, db_config, db_name, table_name, output_text):
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
        csv_file_path = file_path.replace('.xlsx', '.csv')
        df.to_csv(csv_file_path, index=False, encoding='utf-8')
    else:
        csv_file_path = file_path
        df = pd.read_csv(csv_file_path)

    create_table_sql = f"CREATE TABLE IF NOT EXISTS {db_name}.{table_name} ("
    for column in df.columns:
        col_type = pandas_type_to_sql(str(df[column].dtype))
        create_table_sql += f"{column} {col_type}, "
    create_table_sql = create_table_sql.strip(', ') + ');'

    create_db_sql = f"CREATE DATABASE IF NOT EXISTS {db_name}"

    try:
        connection = pymysql.connect(**db_config)
        with connection.cursor() as cursor:
            cursor.execute(create_db_sql)
            connection.commit()

        db_config['db'] = db_name
        connection = pymysql.connect(**db_config)

        with connection.cursor() as cursor:
            cursor.execute(create_table_sql)
            connection.commit()

            load_data_sql = f"""
            LOAD DATA LOCAL INFILE %s
            INTO TABLE {db_name}.{table_name}
            FIELDS TERMINATED BY ',' ENCLOSED BY '"'
            LINES TERMINATED BY '\\n'
            IGNORE 1 LINES;
            """
            cursor.execute(load_data_sql, (csv_file_path,))
            connection.commit()
            append_text(output_text, f"数据成功导入到 {db_name}.{table_name} 表。")

    except pymysql.MySQLError as e:
        append_text(output_text, f"数据库错误: {e}")
    finally:
        if connection:
            connection.close()

    if file_path.endswith('.xlsx') and os.path.exists(csv_file_path):
        os.remove(csv_file_path)

# GUI部分
def run_gui():
    root = tk.Tk()
    root.title("CSV/XLSX to MySQL")
    root.geometry("600x400")

    # 文件路径选择
    Label(root, text="文件路径:").grid(row=0, column=0, sticky=W)
    file_path_entry = Entry(root, width=50)
    file_path_entry.grid(row=0, column=1, columnspan=2, sticky=(W, E))

    def upload_action(event=None):
        filename = filedialog.askopenfilename()
        file_path_entry.delete(0, tk.END)
        file_path_entry.insert(0, filename)

    Button(root, text="选择文件", command=upload_action).grid(row=0, column=3, sticky=W)

    # 数据库配置
    Label(root, text="数据库配置").grid(row=1, column=0, columnspan=4, sticky=W)

    Label(root, text="Host:").grid(row=2, column=0, sticky=W)
    host_entry = Entry(root, width=50)
    host_entry.insert(0, "localhost")
    host_entry.grid(row=2, column=1, columnspan=3, sticky=(W, E))

    Label(root, text="User:").grid(row=3, column=0, sticky=W)
    user_entry = Entry(root, width=50)
    user_entry.insert(0, "root")
    user_entry.grid(row=3, column=1, columnspan=3, sticky=(W, E))

    Label(root, text="Password:").grid(row=4, column=0, sticky=W)
    password_entry = Entry(root, width=50)
    password_entry.insert(0, "123456")
    password_entry.grid(row=4, column=1, columnspan=3, sticky=(W, E))

    Label(root, text="Database:").grid(row=5, column=0, sticky=W)
    db_entry = Entry(root, width=50)
    db_entry.insert(0, "Environmental")
    db_entry.grid(row=5, column=1, columnspan=3, sticky=(W, E))

    Label(root, text="Table:").grid(row=6, column=0, sticky=W)
    table_entry = Entry(root, width=50)
    table_entry.insert(0, "disclosure_info")
    table_entry.grid(row=6, column=1, columnspan=3, sticky=(W, E))

    # 输出文本区
    output_text = Text(root, height=10, state='disabled')
    output_text.grid(row=7, column=0, columnspan=4, sticky=(N, S, E, W))

    # 滚动条
    scrollbar = Scrollbar(root, orient="vertical", command=output_text.yview)
    scrollbar.grid(row=7, column=4, sticky=(N, S))
    output_text['yscrollcommand'] = scrollbar.set

    # 开始导入按钮
    def start_import():
        db_config = {
            'host': host_entry.get(),
            'user': user_entry.get(),
            'password': password_entry.get(),
            'charset': 'utf8mb4',
            'local_infile': 1,
        }
        db_name = db_entry.get()
        table_name = table_entry.get()
        file_path = file_path_entry.get()
        load_data(file_path, db_config, db_name, table_name, output_text)

    start_button = Button(root, text="开始导入", command=start_import)
    start_button.grid(row=8, column=0, columnspan=4, sticky=(W, E))

    root.mainloop()

run_gui()
