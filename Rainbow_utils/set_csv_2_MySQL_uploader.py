import os
import pandas as pd
import pymysql
import gradio as gr


class CSVToMySQLUploader:
    def __init__(self):
        # Define the interface using the Blocks API for a more custom layout
        self.interface = gr.Blocks()

    @staticmethod
    def pandas_type_to_sql(type_name, precision=None, length=None):
        if 'int' in type_name:
            return 'INT' if length is None else f'INT({length})'
        elif 'float' in type_name:
            if precision is not None and length is not None:
                return f'DOUBLE({length},{precision})'
            elif precision is not None:
                return f'DOUBLE({precision})'
            else:
                return 'DOUBLE'
        elif 'datetime' in type_name:
            return 'DATETIME'
        elif 'bool' in type_name:
            return 'BOOLEAN'
        elif 'object' in type_name:
            return f'VARCHAR({length})' if length is not None else 'VARCHAR(255)'
        elif 'timedelta' in type_name:
            return 'VARCHAR(255)'  # No direct timedelta equivalent, treat as string
        elif 'category' in type_name:
            return f'VARCHAR({length})' if length is not None else 'VARCHAR(255)'
        elif 'complex' in type_name:
            return f'VARCHAR({length})' if length is not None else 'VARCHAR(255)'  # Assuming complex types should be treated as strings
        else:
            return f'VARCHAR({length})' if length is not None else 'VARCHAR(255)'  # Default for unknown types

    def load_data(self, file_path, db_config, db_name, table_name, output_callback):
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
            csv_file_path = file_path.replace('.xlsx', '.csv')
            df.to_csv(csv_file_path, index=False, encoding='utf-8')
        else:
            csv_file_path = file_path
            df = pd.read_csv(csv_file_path)

        create_table_sql = f"CREATE TABLE IF NOT EXISTS {db_name}.{table_name} ("
        for column in df.columns:
            col_type = self.pandas_type_to_sql(str(df[column].dtype))
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
                output_callback(f"Data successfully imported to {db_name}.{table_name} table.")

        except pymysql.MySQLError as e:
            output_callback(f"Database error: {e}")
        finally:
            if connection:
                connection.close()

        if file_path.endswith('.xlsx') and os.path.exists(csv_file_path):
            os.remove(csv_file_path)

    def load_data_gradio(self, file, host, user, password, db_name, table_name):
        output_text = ""

        def append_output(text):
            nonlocal output_text
            output_text += text + "\n"

        try:
            file_path = file.name
            db_config = {
                'host': host,
                'user': user,
                'password': password,
                'charset': 'utf8mb4',
                'local_infile': 1,
            }
            self.load_data(file_path, db_config, db_name, table_name, append_output)
            return output_text
        except Exception as e:
            return str(e)

    def launch(self):
        with gr.Blocks() as interface:
            gr.Markdown("# CSV/XLSX to MySQL Uploader")
            
            with gr.Row():
                # 左侧数据库配置
                with gr.Column(scale=2):
                    file_input = gr.File(label="Select CSV/XLSX File")
                    with gr.Row():
                        host_input = gr.Textbox(value="localhost", label="Host")
                        user_input = gr.Textbox(value="root", label="User")
                        password_input = gr.Textbox(value="", label="Password", type="password")
                    with gr.Row():
                        db_input = gr.Textbox(value="Environmental", label="Database")
                        table_input = gr.Textbox(value="disclosure_info", label="Table Name")
                    upload_button = gr.Button("Upload Data")
                    output_text = gr.Textbox(label="OutPut Logs", lines=10, interactive=False)
                
                # 右侧使用说明
                with gr.Column(scale=3):
                    gr.Markdown("""
                        ### 📊 数据导入工具使用指南
                        
                        #### 1️⃣ 文件准备
                        - **支持格式：** CSV文件、XLSX文件（Excel）
                        - **文件要求：**
                          - 确保文件编码为UTF-8
                          - 第一行应为列名
                          - 数据格式统一
                        
                        #### 2️⃣ 数据库配置
                        - **Host**: 数据库服务器地址（默认localhost）
                        - **User**: 数据库用户名（默认root）
                        - **Password**: 数据库密码
                        - **Database**: 目标数据库名称
                        - **Table**: 目标表名称
                        
                        #### ⚠️ 注意事项
                        - 确保数据库连接信息正确
                        - 大文件导入可能需要较长时间
                        - 表将自动创建（如不存在）
                        - 字段类型会自动推断
                        - 建议先备份重要数据
                        
                        #### 💡 使用技巧
                        - 导入前检查数据格式
                        - 观察日志了解导入进度
                        - 如遇错误，检查数据库权限
                    """)

            upload_button.click(
                self.load_data_gradio,
                inputs=[file_input, host_input, user_input, password_input, db_input, table_input],
                outputs=output_text
            )

            return interface
