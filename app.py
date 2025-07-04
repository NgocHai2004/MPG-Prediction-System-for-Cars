import gradio as gr
import numpy as np
import pandas as pd
import joblib

# Load mô hình và pipeline
model = joblib.load("model_rf.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Cột đầu vào cho model (không có car_name)
columns = ['displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'cylinders', 'origin']

def predict(car_name, displacement, horsepower, weight, acceleration, model_year, cylinders, origin):
    try:
        # Tạo DataFrame đúng định dạng
        input_df = pd.DataFrame([[displacement, horsepower, weight, acceleration, model_year, cylinders, origin]],
                                columns=columns)

        # Tiền xử lý
        processed_input = preprocessor.transform(input_df)

        # Dự đoán
        prediction = model.predict(processed_input)[0]

        return f"🚗 Xe: {car_name}\n✅ Dự đoán mức tiêu thụ nhiên liệu (MPG): {round(prediction, 2)}"
    except Exception as e:
        return f"❌ Lỗi: {str(e)}"

# Giao diện với thêm trường "car name"
inputs = [
    gr.Textbox(label="Tên xe (car name)", value="Toyota Corolla"),
    gr.Number(label="Dung tích động cơ (displacement)", value=150.0),
    gr.Number(label="Mã lực (horsepower)", value=95.0),
    gr.Number(label="Trọng lượng xe (weight)", value=2500.0),
    gr.Number(label="Tăng tốc (acceleration)", value=15.0),
    gr.Number(label="Năm sản xuất (model year)", value=76),
    gr.Number(label="Số xi-lanh (cylinders)", value=4),
    gr.Number(label="Xuất xứ (origin: 1-Mỹ, 2-Châu Âu, 3-Nhật)", value=1)
]

app = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs="text",
    title="🚗 Dự đoán mức tiêu thụ nhiên liệu (MPG)",
    description="Nhập thông tin xe để dự đoán mức tiêu thụ nhiên liệu. Trường 'car name' chỉ để hiển thị."
)

app.launch()
