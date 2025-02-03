from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import psycopg2

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT")
        )
        return conn
    except Exception as e:
        print("Could not connect to the database:", e)
        return None

@app.route('/api/data', methods=['GET'])
def get_data():
    conn = get_db_connection()
    if conn is None:
        return jsonify({"error": "Database connection failed"}), 500

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM your_table;")
    data = cursor.fetchall()
    cursor.close()
    conn.close()

    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)