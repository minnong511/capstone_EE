import sqlite3

conn = sqlite3.connect('./DB/inference_results.db')
cursor = conn.cursor()

cursor.execute("SELECT * FROM inference_results ORDER BY id DESC LIMIT 10")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()