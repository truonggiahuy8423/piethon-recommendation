from flask import Flask, request, jsonify
import pandas as pd
import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import mysql.connector
from mysql.connector import Error
from apscheduler.schedulers.background import BackgroundScheduler
import time
import logging


app = Flask(__name__)

# Đường dẫn lưu file mô hình và dữ liệu
MODEL_PATH = "lstm_model.h5"
DATA_PATH = "data.xlsx"
SCALER_X_PATH = "scaler_X.pkl"
SCALER_Y_PATH = "scaler_y.pkl"
PREVIOUS = 5
PREDICT = 5

# Hàm để build lại mô hình
def build_model():
    start_time = time.time()  # Thời gian bắt đầu
    # Lấy apiKey từ query parameters
    api_key = "your_secret_api_key"

    # Kết nối và thu thập dữ liệu từ MySQL
    # host = "127.0.0.1"
    # port = "3308"
    # database = "course"
    # username = "root"
    # password = "password"
    host = "coursedb.mysql.database.azure.com"
    port = "3306"
    database = "course"
    username = "courseuser"
    password = "123456aA@"

    try:
        connection = mysql.connector.connect(
            host=host,
            port=port,
            database=database,
            user=username,
            password=password
        )

        if connection.is_connected():
            print("Connected to MySQL database")

            # Thực hiện truy vấn
            query = "SELECT * FROM history_view ORDER BY user_id ASC, history_id ASC"
            cursor = connection.cursor()
            cursor.execute(query)

            # Lấy dữ liệu và chuyển đổi thành DataFrame
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            data = pd.DataFrame(rows, columns=columns)

            # Đếm số dòng cho mỗi user_id và lọc user_id có ít nhất 10 dòng
            user_counts = data["user_id"].value_counts()
            valid_user_ids = user_counts[user_counts >= 10].index
            filtered_df = data[data["user_id"].isin(valid_user_ids)]

            # Lấy danh sách course_id và tạo input-output
            course_ids = filtered_df["course_id"].tolist()
            window_size = PREVIOUS + PREDICT
            rows = [
                course_ids[i : i + window_size]
                for i in range(len(course_ids) - window_size + 1)
            ]

            df_output = pd.DataFrame(rows, columns=[f"col_{i+1}" for i in range(window_size)])
            df_output.columns = [f"input_{i+1}" for i in range(PREVIOUS)] + [f"output_{i+1}" for i in range(PREDICT)]

            # Tách input và output
            X = df_output.iloc[:, :PREVIOUS].values
            y = df_output.iloc[:, PREVIOUS:].values

    except Error as e:
        print(f"Database connection failed: {str(e)}")
        return
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection closed")

    # Chuẩn hóa dữ liệu
    global scaler_X, scaler_y
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y)

    # Tách dữ liệu train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Chuyển đổi thành dạng 3D cho LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Xây dựng mô hình LSTM
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.optimizers import Adam

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
        Dense(32, activation='relu'),
        Dense(y.shape[1])  # Số đầu ra bằng số cột của y
    ])

    # Biên dịch mô hình
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

    # Huấn luyện mô hình
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1, verbose=1)

    # Đánh giá mô hình
    loss, mae = model.evaluate(X_test, y_test, verbose=0)

    # Lưu mô hình và scaler
    model.save(MODEL_PATH)
    joblib.dump(scaler_X, SCALER_X_PATH)
    joblib.dump(scaler_y, SCALER_Y_PATH)

    end_time = time.time()

    logging.info("Model built successfully", "loss:", loss, "mae:", mae)
    logging.info(f"Time taken: {end_time - start_time} seconds")

# Hàm khởi động build mô hình khi Flask bắt đầu
def start_building_model():
    print("Building model on startup...")
    build_model()

# Thiết lập scheduler để tự động build mô hình sau mỗi 30 phút
scheduler = BackgroundScheduler()
scheduler.add_job(start_building_model, 'interval', minutes=30)
scheduler.start()

@app.route('/', methods=['GET'])
def index():
    logging.info("Hello world!")
    return "Hello world!"

# Endpoint để dự đoán
@app.route('/predict', methods=['POST'])
def predict():
    # Lấy apiKey từ query parameters
    api_key = request.args.get('apiKey')
    if not api_key or api_key != "your_secret_api_key":
        return jsonify({"error": "Invalid or missing apiKey"}), 403

    # Kiểm tra sự tồn tại của mô hình
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": f"Model not found at {MODEL_PATH}"}), 400

    # Kiểm tra sự tồn tại của scaler
    if not os.path.exists(SCALER_X_PATH) or not os.path.exists(SCALER_Y_PATH):
        return jsonify({"error": "Scaler files not found. Please rebuild the model first."}), 400

    # Tải scaler từ file
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)

    # Lấy dữ liệu input (10 số nguyên) từ request
    data = request.json.get('data')
    if not data or len(data) != PREVIOUS:
        return jsonify({"error": "Invalid input. Expected a list of 10 integers."}), 400
    
    # Chuyển đổi dữ liệu thành numpy array và chuẩn hóa
    input_data = np.array(data).reshape(1, -1)
    input_data = scaler_X.transform(input_data)

    # Chuyển đổi dữ liệu thành dạng 3D cho LSTM (số mẫu, số đặc trưng, 1)
    input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))

    # Tải mô hình
    model = load_model(MODEL_PATH)

    # Dự đoán kết quả
    predicted = model.predict(input_data)

    # Chuẩn hóa kết quả về lại phạm vi ban đầu
    predicted = scaler_y.inverse_transform(predicted)

    predicted = np.round(predicted).astype(int)

    # Trả kết quả dự đoán dưới dạng JSON
    return jsonify({"prediction": predicted.tolist()[0]})

if __name__ == '__main__':
    # Kiểm tra file dữ liệu tồn tại
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Excel file not found at {DATA_PATH}. Please upload it.")

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting the app...")

    start_building_model()

    app.run(host='0.0.0.0', port=8000, debug=True)
