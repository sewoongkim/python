import mysql.connector as mariadb
import sys

# Connect to MariaDB Platform
try:
    conn = mariadb.connect(
        user="snakemake",
        password="7ag^CbcF",
        host="3.34.107.112",
        port=3306,
        database="testdb"

    )
except mariadb.Error as e:
    print(f"Error connecting to MariaDB Platform: {e}")
    sys.exit(1)

# Get Cursor
cur = conn.cursor()

cur.execute("SELECT no, cdate FROM e_data") 

rows = cur.fetchall()
print(rows) 

conn.commit()
cur.close()
conn.close()



