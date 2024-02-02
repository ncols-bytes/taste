import json
import argparse
from tqdm import tqdm
import mysql.connector
from data_process.data_processor import *

import warnings
warnings.filterwarnings('ignore')


def build_single_table(connection, cursor, table_id, pgTitle, pgEntity, secTitle, caption, headers, cells):
    table_name = 'table_' + str(table_id).replace('-', '_')

    # create table
    comment = pgTitle[:100] + "#|+=" + str(pgEntity) + "#|+=" + secTitle[:100] + "#|+=" + caption[:100]
    comment = str(comment).replace('\\', "\\\\").replace("'", "\\\'").replace('"', "\\\"")
    create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name}("

    for ci in range(len(headers)):
        column_comment = str(headers[ci])[:50].replace('\\', "\\\\").replace("'", "\\\'").replace('"', "\\\"")
        create_table_sql += f"column{str(ci)} TEXT COMMENT "
        create_table_sql += "\'"
        create_table_sql += column_comment
        create_table_sql += '\'' + ", "

    create_table_sql = create_table_sql.rstrip(', ') + ")"
    create_table_sql += "COMMENT = "
    create_table_sql += '\"'
    create_table_sql += comment
    create_table_sql += '\"'

    cursor.execute(create_table_sql)
    connection.commit()

    # preprocess content data
    table_data = []
    for ci in range(len(headers)):
        col_cell = cells[ci]
        for cj in range(len(col_cell)):
            index = col_cell[cj][0]
            cell_id = col_cell[cj][1][0]
            cell_value = col_cell[cj][1][1]

            row_idx = index[0]
            col_idx = index[1]

            existed_rows = len(table_data)
            for _ in range(row_idx + 1 - existed_rows):
                table_data.append([None for _ in range(len(headers))])

            table_data[row_idx][col_idx] = cell_value

    # check if already inserted
    query = f"SELECT COUNT(*) FROM {table_name}"
    cursor.execute(query)
    row_count = cursor.fetchone()[0]
    if row_count != len(table_data):
        # insert data
        for row_idx in range(len(table_data)):
            insert_sql = f"INSERT INTO {table_name}("
            for col_idx in range(len(table_data[row_idx])):
                insert_sql += f"column{str(col_idx)}" + ", "
            insert_sql = insert_sql.rstrip(', ') + ")"
            insert_sql += f" VALUES ("
        
            for col_idx in range(len(table_data[row_idx])):
                if table_data[row_idx][col_idx] != None:
                    insert_sql += '\"'
                    insert_sql += table_data[row_idx][col_idx][:1000].replace('\\', "\\\\").replace("'", "\\'").replace('"', '\\"')
                    insert_sql += '\"' + ", "
                else:
                    insert_sql += 'null' + ", "

            insert_sql = insert_sql.rstrip(', ') + ")"
            cursor.execute(insert_sql)
        connection.commit()

    # generate histogram
    for ci in range(len(headers)):
        column_name = 'column' + str(ci)
        generate_histogram_sql = f"ANALYZE TABLE {table_name} UPDATE HISTOGRAM ON {column_name} WITH 1024 BUCKETS;"
        cursor.execute(generate_histogram_sql)
        cursor.fetchall()
    connection.commit()


def build_tables(src_data_path, db_name, connection):
    database_name = db_name
    cursor = connection.cursor()

    # create database
    cursor = connection.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name}")
    print(f"Database '{database_name}' created successfully.")

    cursor.execute(f"USE {database_name}")
    print(f"Switched to database '{database_name}'.")

    print(f"Building tables...")
    with open(src_data_path, 'r') as fcc_file:
        fcc_data = json.load(fcc_file)
        for i in tqdm(range(len(fcc_data)), desc="Processing"):
            table_id = fcc_data[i][0]
            pgTitle = fcc_data[i][1]
            pgEntity = fcc_data[i][2]
            secTitle = fcc_data[i][3]
            caption = fcc_data[i][4]
            headers = fcc_data[i][5]
            cells = fcc_data[i][6]
            # annotations = fcc_data[i][7]

            build_single_table(connection, cursor, table_id, pgTitle, pgEntity, secTitle, caption, headers, cells)

    cursor.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mysql_host", default=None, type=str, required=True)
    parser.add_argument("--mysql_port", default=3306, type=int, required=False)
    parser.add_argument("--mysql_user", default=None, type=str, required=True)
    parser.add_argument("--mysql_password", default=None, type=str, required=True)
    parser.add_argument("--eval_database", default=None, type=str, required=True)
    parser.add_argument("--test_dataset", default=None, type=str, required=True)

    args = parser.parse_args()

    connection = mysql.connector.connect(
        host=args.mysql_host,
        port=args.mysql_port,
        user=args.mysql_user,
        password=args.mysql_password,
    )

    build_tables(args.test_dataset, args.eval_database, connection)

    connection.close()


if __name__ == "__main__":
    main()

