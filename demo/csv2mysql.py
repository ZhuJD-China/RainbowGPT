import pandas as pd
import mysql.connector
from datetime import datetime
import numpy as np

# MySQL数据库连接配置
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'movies_data'
}

# 读取CSV文件
csv_file = "C:\ProgramData\MySQL\MySQL Server 8.0\\Uploads\\Movies_dataset.csv"
data = pd.read_csv(csv_file)

# 建立数据库连接
connection = mysql.connector.connect(**db_config)
cursor = connection.cursor()

# 检查数据库表是否存在
table_name = 'movies'
table_exists_query = f"SHOW TABLES LIKE '{table_name}'"
cursor.execute(table_exists_query)
table_exists = cursor.fetchone()

if not table_exists:
    # 创建数据库表
    create_table_query = """
    CREATE TABLE movies (
        movie_index INT,
        title VARCHAR(255),
        original_language VARCHAR(50),
        release_date DATE,
        popularity FLOAT,
        vote_average FLOAT,
        vote_count INT,
        overview TEXT,
        PRIMARY KEY (movie_index)
    )
    """
    cursor.execute(create_table_query)
    print("Table created.")

# 将数据导入数据库
for index, row in data.iterrows():
    if pd.notna(row['release_date']):
        release_date = datetime.strptime(row['release_date'], '%d-%m-%Y').date()
    else:
        release_date = None

    popularity = row['popularity'] if not pd.isna(row['popularity']) else None
    vote_average = row['vote_average'] if not pd.isna(row['vote_average']) else None
    vote_count = int(row['vote_count']) if not pd.isna(row['vote_count']) else None

    # Replace NaN values with None
    overview = row['overview'] if not pd.isna(row['overview']) else None

    insert_query = """
    INSERT INTO movies (movie_index, title, original_language, release_date, popularity, vote_average, vote_count, overview)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = (
        index, row['title'], row['original_language'], release_date,
        popularity, vote_average, vote_count, overview
    )
    cursor.execute(insert_query, values)

# 提交更改
connection.commit()
print("Data imported successfully!")

# 关闭连接
connection.close()
