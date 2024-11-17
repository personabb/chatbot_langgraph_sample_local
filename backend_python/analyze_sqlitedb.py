#本コードはlangGraphのSqliteSaverで保存されたチェックポイントの中身を見るためのコードです。

import sqlite3
import pandas as pd
import os
import json
import msgpack

def decode_binary_data(df):
    """
    データフレーム内のバイナリデータをデコードして文字列に変換する。

    :param df: Pandas DataFrame
    :return: デコードされたDataFrame
    """
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: decode_if_binary(x))
    return df

def decode_if_binary(value):
    """
    値がバイナリデータの場合にデコードし、それ以外の場合はそのまま返す。

    :param value: バイナリデータまたはその他の値
    :return: デコードされた値または元の値
    """
    if isinstance(value, bytes):
        try:
            # msgpackとしてデコード
            return msgpack.unpackb(value, raw=False)
        except (msgpack.exceptions.ExtraData, msgpack.exceptions.FormatError):
            try:
                # UTF-8としてデコード
                return value.decode('utf-8')
            except UnicodeDecodeError:
                # JSONとしてデコード
                try:
                    return json.loads(value.decode('utf-8'))
                except Exception:
                    return value  # デコードできない場合はそのまま返す
    return value

def save_sqlite_tables_to_csv(db_path, output_dir):
    """
    SQLiteデータベース内の各テーブルをCSVファイルとして保存する。
    バイナリデータはデコードして保存。

    :param db_path: SQLiteデータベースファイルのパス
    :param output_dir: CSVファイルを保存するディレクトリ
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    print("=== テーブル一覧 ===")
    for table in tables:
        print(f"- {table[0]}")

    print("\n=== 各テーブルをCSVに保存 ===")
    for table in tables:
        table_name = table[0]
        print(f"テーブル名: {table_name}")

        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            df = decode_binary_data(df)
            output_path = os.path.join(output_dir, f"{table_name}.csv")
            df.to_csv(output_path, index=False, encoding="utf-8")
            print(f"テーブル '{table_name}' をCSVファイルに保存しました: {output_path}")
        except Exception as e:
            print(f"エラー: テーブル '{table_name}' の処理中に問題が発生しました: {e}")

    conn.close()

# データベースのパスを指定して実行
db_path = "./sqlite_db/asap2650.sqlite"
output_dir = "./sqlite_output"
save_sqlite_tables_to_csv(db_path, output_dir)
