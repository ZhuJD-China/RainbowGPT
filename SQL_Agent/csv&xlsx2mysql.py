import os
import pandas as pd
import pymysql


def pandas_type_to_sql(type_name):
    """
    将 pandas 数据类型转换为 SQL 数据类型。
    """
    if 'int' in type_name:
        return 'INT'
    elif 'float' in type_name:
        return 'FLOAT'
    elif 'datetime' in type_name:
        return 'DATETIME'
    else:  # 默认作为字符串处理
        return 'VARCHAR(255)'


# 数据库连接配置
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'charset': 'utf8mb4',
    'local_infile': 1,  # 启用本地文件加载
}

# 目标数据库和表的名称
db_name = 'Environmental'
table_name = 'disclosure_info'

# 文件路径
file_path = 'D:\\AIGC\\RainbowGPT\\data\\环境信息披露数据总表.xlsx'  # 可以是 .xlsx 或 .csv

# 转换文件格式，如果是 Excel 则转为 CSV
if file_path.endswith('.xlsx'):
    df = pd.read_excel(file_path)
    csv_file_path = file_path.replace('.xlsx', '.csv')
    df.to_csv(csv_file_path, index=False, encoding='utf-8')
else:
    csv_file_path = file_path
    df = pd.read_csv(csv_file_path)

# 分析数据结构并构建 SQL 创建表语句
create_table_sql = f"CREATE TABLE IF NOT EXISTS {db_name}.{table_name} ("

for column in df.columns:
    col_type = pandas_type_to_sql(str(df[column].dtype))
    create_table_sql += f"{column} {col_type}, "

create_table_sql = create_table_sql.strip(', ') + ');'

# SQL 语句：创建数据库
create_db_sql = f"CREATE DATABASE IF NOT EXISTS {db_name}"

# 连接 MySQL（不指定数据库）
connection = pymysql.connect(**db_config)

try:
    with connection.cursor() as cursor:
        # 创建数据库（如果不存在）
        cursor.execute(create_db_sql)
        connection.commit()

    # 更新数据库连接以包含数据库名
    db_config['db'] = db_name
    connection = pymysql.connect(**db_config)

    with connection.cursor() as cursor:
        # 创建表（如果不存在）
        cursor.execute(create_table_sql)
        connection.commit()

        # SQL 语句：导入 CSV 数据
        load_data_sql = f"""
        LOAD DATA LOCAL INFILE %s
        INTO TABLE {db_name}.{table_name}
        FIELDS TERMINATED BY ',' ENCLOSED BY '"'
        LINES TERMINATED BY '\\n'
        IGNORE 1 LINES;
        """

        # 执行数据导入操作
        cursor.execute(load_data_sql, (csv_file_path,))
        connection.commit()
        print(f"数据成功导入到 {db_name}.{table_name} 表。")

except pymysql.MySQLError as e:
    print(f"数据库错误: {e}")
finally:
    if connection:
        connection.close()

# 清理：删除临时 CSV 文件（如果原始文件是 Excel）
if file_path.endswith('.xlsx') and os.path.exists(csv_file_path):
    os.remove(csv_file_path)
