from flask import Flask, request, jsonify, render_template_string
import numpy as np
import os
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing import image

class ImagePredictor:
    def __init__(self, model_path, class_to_calories, threshold=0.1):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.class_to_calories = class_to_calories
        self.threshold = threshold

    def load_and_preprocess_image(self, img_path, target_size=(224, 224)):
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array

    def predict_calories(self, img_path):
        img_array = self.load_and_preprocess_image(img_path)
        self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
        self.interpreter.invoke()
        predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        results = []
        for idx, pred in enumerate(predictions):
            if pred >= self.threshold:
                class_name = list(self.class_to_calories.keys())[idx]
                calories = self.class_to_calories[class_name]
                results.append({
                    "nama makanan": class_name,
                    "kalori": calories
                })

        if not results:
            max_idx = np.argmax(predictions)
            class_name = list(self.class_to_calories.keys())[max_idx]
            calories = self.class_to_calories[class_name]
            results.append({
                "nama makanan": class_name,
                "kalori": calories
            })

        return results

class_to_calories = {
    "Ayam_Crispy": 297,
    "Ayam_Kecap": 223,
    "Ayam_Serundeng": 260,
    "Bakso": 76,
    "Brownies": 379,
    "Bubur_Ayam": 155,
    "Capcay": 203,
    "Cumi_Bakar": 185,
    "Cumi_Hitam": 174,
    "Cumi_Rica": 200,
    "Dimsum_Ikan": 112,
    "Garang_Asem": 150,
    "Ikan_Bakar": 126,
    "Ikan_Goreng": 84,
    "Kentang_Balado": 102,
    "Kue_Bolu": 297,
    "Nasi_Bakar": 281,
    "Nasi_Goreng": 276,
    "Nasi_Kuning": 95,
    "Nasi_Merah": 110,
    "Nasi_Rames": 155,
    "Opor_Ayam": 163,
    "Pancake": 227,
    "Pecel": 270,
    "Pepes_Ikan": 105,
    "Perkedel_Kentang": 143,
    "Pukis": 259,
    "Rawon": 120,
    "Rendang": 193,
    "Salad_Sayur": 17,
    "Sate_Ayam": 225,
    "Sate_Kambing": 216,
    "Sayur_Asem": 29,
    "Sayur_Sop": 27,
    "Soto_Ayam": 130,
    "Telur_Balado": 202,
    "Telur_Dadar": 153,
    "Tumis_Kacang_Panjang_Tahu": 140,
    "Tumis_Kangkung": 98,
    "Tumis_Terong": 65,
    "Udang_Asam_Manis": 269,
    "Udang_Goreng_Tepung": 287
}

# URL model di GitHub
model_url = "https://github.com/Alvinnxyz/Machine-Learning-FitFood/raw/main/model3.tflite"
model_path = "model3.tflite"

def download_model(url, path):
    response = requests.get(url)
    response.raise_for_status()
    with open(path, 'wb') as f:
        f.write(response.content)

# Unduh model jika belum ada
if not os.path.exists(model_path):
    download_model(model_url, model_path)

# Inisialisasi ImagePredictor dengan model TFLite
predictor = ImagePredictor(model_path, class_to_calories)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string('''
        <!doctype html>
        <title>Image Upload</title>
        <h1>Upload an image to predict calories</h1>
        <form method=post enctype=multipart/form-data action="/predict">
          <input type=file name=file>
          <input type=submit value=Upload>
        </form>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file:
        file_path = "temp_image.jpg"
        file.save(file_path)
        
        predictions = predictor.predict_calories(file_path)
        
        os.remove(file_path)
        
        return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)


# from flask import Flask, request, jsonify, render_template_string
# import numpy as np
# import os
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image

# class ImagePredictor:
#     def __init__(self, model_path, class_to_calories, threshold=0.1):
#         # Load the TFLite model
#         self.interpreter = tf.lite.Interpreter(model_path=model_path)
#         self.interpreter.allocate_tensors()
#         self.input_details = self.interpreter.get_input_details()
#         self.output_details = self.interpreter.get_output_details()
#         self.class_to_calories = class_to_calories
#         self.threshold = threshold

#     def load_and_preprocess_image(self, img_path, target_size=(224, 224)):
#         img = image.load_img(img_path, target_size=target_size)
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array /= 255.0
#         return img_array

#     def predict_calories(self, img_path):
#         img_array = self.load_and_preprocess_image(img_path)

#         # Set the tensor to point to the input data to be inferred
#         self.interpreter.set_tensor(self.input_details[0]['index'], img_array)

#         # Run the inference
#         self.interpreter.invoke()

#         # Get the output tensor
#         predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

#         results = []
#         for idx, pred in enumerate(predictions):
#             if pred >= self.threshold:
#                 class_name = list(self.class_to_calories.keys())[idx]
#                 calories = self.class_to_calories[class_name]
#                 results.append({
#                     "nama makanan": class_name,
#                     "kalori": calories
#                 })

#         if not results:
#             max_idx = np.argmax(predictions)
#             class_name = list(self.class_to_calories.keys())[max_idx]
#             calories = self.class_to_calories[class_name]
#             results.append({
#                 "nama makanan": class_name,
#                 "kalori": calories
#             })

#         return results

# class_to_calories = {
#     "Ayam_Crispy": 297,
#     "Ayam_Kecap": 223,
#     "Ayam_Serundeng": 260,
#     "Bakso": 76,
#     "Brownies": 379,
#     "Bubur_Ayam": 155,
#     "Capcay": 203,
#     "Cumi_Bakar": 185,
#     "Cumi_Hitam": 174,
#     "Cumi_Rica": 200,
#     "Dimsum_Ikan": 112,
#     "Garang_Asem": 150,
#     "Ikan_Bakar": 126,
#     "Ikan_Goreng": 84,
#     "Kentang_Balado": 102,
#     "Kue_Bolu": 297,
#     "Nasi_Bakar": 281,
#     "Nasi_Goreng": 276,
#     "Nasi_Kuning": 95,
#     "Nasi_Merah": 110,
#     "Nasi_Rames": 155,
#     "Opor_Ayam": 163,
#     "Pancake": 227,
#     "Pecel": 270,
#     "Pepes_Ikan": 105,
#     "Perkedel_Kentang": 143,
#     "Pukis": 259,
#     "Rawon": 120,
#     "Rendang": 193,
#     "Salad_Sayur": 17,
#     "Sate_Ayam": 225,
#     "Sate_Kambing": 216,
#     "Sayur_Asem": 29,
#     "Sayur_Sop": 27,
#     "Soto_Ayam": 130,
#     "Telur_Balado": 202,
#     "Telur_Dadar": 153,
#     "Tumis_Kacang_Panjang_Tahu": 140,
#     "Tumis_Kangkung": 98,
#     "Tumis_Terong": 65,
#     "Udang_Asam_Manis": 269,
#     "Udang_Goreng_Tepung": 287
# }

# # Initialize the ImagePredictor with the TFLite model
# predictor = ImagePredictor("model3.tflite", class_to_calories)

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template_string('''
#         <!doctype html>
#         <title>Image Upload</title>
#         <h1>Upload an image to predict calories</h1>
#         <form method=post enctype=multipart/form-data action="/predict">
#           <input type=file name=file>
#           <input type=submit value=Upload>
#         </form>
#     ''')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"})
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return jsonify({"error": "No selected file"})
    
#     if file:
#         file_path = "temp_image.jpg"
#         file.save(file_path)
        
#         predictions = predictor.predict_calories(file_path)
        
#         os.remove(file_path)
        
#         return jsonify(predictions)

# if __name__ == '__main__':
#     app.run(debug=True)






# # from flask import Flask, request, jsonify
# # import tensorflow as tf
# # from tensorflow.keras.preprocessing import image
# # import numpy as np
# # import os

# # class ImagePredictor:
# #     def __init__(self, model_path, class_to_calories, threshold=0.1):
# #         self.model = tf.keras.models.load_model(model_path)
# #         self.class_to_calories = class_to_calories
# #         self.threshold = threshold
    
# #     def load_and_preprocess_image(self, img_path, target_size=(224, 224)):
# #         # Load the image
# #         img = image.load_img(img_path, target_size=target_size)
# #         img_array = image.img_to_array(img)
# #         img_array = np.expand_dims(img_array, axis=0)
# #         img_array /= 255.0
# #         return img_array
    
# #     # def predict_calories(self, img_path):
# #     #     img_array = self.load_and_preprocess_image(img_path)
# #     #     predictions = self.model.predict(img_array)[0]  # Get the first (and only) prediction
# #     #     result = {}
        
# #     #     for idx, pred in enumerate(predictions):
# #     #         if pred >= self.threshold:
# #     #             class_name = list(self.class_to_calories.keys())[idx]
# #     #             calories = self.class_to_calories[class_name]
# #     #             result[class_name] = calories
        
# #     #     if not result:
# #     #         max_idx = np.argmax(predictions)
# #     #         class_name = list(self.class_to_calories.keys())[max_idx]
# #     #         calories = self.class_to_calories[class_name]
# #     #         result = {class_name: calories}
        
# #     #     return result
# #     def predict_calories(self, img_path):
# #         img_array = self.load_and_preprocess_image(img_path)
# #         predictions = self.model.predict(img_array)[0]  # Get the first (and only) prediction
        
# #         results = []
        
# #         for idx, pred in enumerate(predictions):
# #             if pred >= self.threshold:
# #                 class_name = list(self.class_to_calories.keys())[idx]
# #                 calories = self.class_to_calories[class_name]
# #                 results.append({class_name: calories})
        
# #         if not results:
# #             max_idx = np.argmax(predictions)
# #             class_name = list(self.class_to_calories.keys())[max_idx]
# #             calories = self.class_to_calories[class_name]
# #             results.append({class_name: calories})
        
# #         return results




# # # Define the mapping from class names to calories
# # class_to_calories = {
# #     "Ayam_Crispy": 297,
# #     "Ayam_Kecap": 223,
# #     "Ayam_Serundeng": 260,
# #     "Bakso": 76,
# #     "Brownies": 379,
# #     "Bubur_Ayam": 155,
# #     "Capcay": 203,
# #     "Cumi_Bakar": 185,
# #     "Cumi_Hitam": 174,
# #     "Cumi_Rica": 200,
# #     "Dimsum_Ikan": 112,
# #     "Garang_Asem": 150,
# #     "Ikan_Bakar": 126,
# #     "Ikan_Goreng": 84,
# #     "Kentang_Balado": 102,
# #     "Kue_Bolu": 297,
# #     "Nasi_Bakar": 281,
# #     "Nasi_Goreng": 276,
# #     "Nasi_Kuning": 95,
# #     "Nasi_Merah": 110,
# #     "Nasi_Rames": 155,
# #     "Opor_Ayam": 163,
# #     "Pancake": 227,
# #     "Pecel": 270,
# #     "Pepes_Ikan": 105,
# #     "Perkedel_Kentang": 143,
# #     "Pukis": 259,
# #     "Rawon": 120,
# #     "Rendang": 193,
# #     "Salad_Sayur": 17,
# #     "Sate_Ayam": 225,
# #     "Sate_Kambing": 216,
# #     "Sayur_Asem": 29,
# #     "Sayur_Sop": 27,
# #     "Soto_Ayam": 130,
# #     "Telur_Balado": 202,
# #     "Telur_Dadar": 153,
# #     "Tumis_Kacang_Panjang_Tahu": 140,
# #     "Tumis_Kangkung": 98,
# #     "Tumis_Terong": 65,
# #     "Udang_Asam_Manis": 269,
# #     "Udang_Goreng_Tepung": 287
# # }

# # # Initialize the ImagePredictor
# # predictor = ImagePredictor("modelw3.h5", class_to_calories)

# # app = Flask(__name__)

# # @app.route('/predict', methods=['POST'])
# # def predict():
# #     if 'file' not in request.files:
# #         return jsonify({"error": "No file part"})
    
# #     file = request.files['file']
    
# #     if file.filename == '':
# #         return jsonify({"error": "No selected file"})
    
# #     if file:
# #         # Save the file temporarily
# #         file_path = "temp_image.jpg"
# #         file.save(file_path)
        
# #         # Predict the calories
# #         predictions = predictor.predict_calories(file_path)
        
# #         # Optionally remove the temporary file if needed
# #         # os.remove(file_path)
        
# #         return jsonify(predictions)

# # if __name__ == '__main__':
# #     app.run(debug=True)

# from flask import Flask, request, jsonify, render_template_string
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import os

# class ImagePredictor:
#     def __init__(self, model_path, class_to_calories, threshold=0.1):
#         self.model = tf.keras.models.load_model(model_path)
#         self.class_to_calories = class_to_calories
#         self.threshold = threshold
    
#     def load_and_preprocess_image(self, img_path, target_size=(224, 224)):
#         # Load the image
#         img = image.load_img(img_path, target_size=target_size)
#         img_array = image.img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)
#         img_array /= 255.0
#         return img_array
    
#     def predict_calories(self, img_path):
#         img_array = self.load_and_preprocess_image(img_path)
#         predictions = self.model.predict(img_array)[0]  # Get the first (and only) prediction
        
#         # results = []
        
#         # for idx, pred in enumerate(predictions):
#         #     if pred >= self.threshold:
#         #         class_name = list(self.class_to_calories.keys())[idx]
#         #         calories = self.class_to_calories[class_name]
#         #         results.append({class_name: calories})
        
#         # if not results:
#         #     max_idx = np.argmax(predictions)
#         #     class_name = list(self.class_to_calories.keys())[max_idx]
#         #     calories = self.class_to_calories[class_name]
#         #     results.append({class_name: calories})
        
#         # return results
#         results = []
        
#         for idx, pred in enumerate(predictions):
#             if pred >= self.threshold:
#                 class_name = list(self.class_to_calories.keys())[idx]
#                 calories = self.class_to_calories[class_name]
#                 results.append({
#                     "nama makanan": class_name,
#                     "kalori": calories
#                 })
        
#         if not results:
#             max_idx = np.argmax(predictions)
#             class_name = list(self.class_to_calories.keys())[max_idx]
#             calories = self.class_to_calories[class_name]
#             results.append({
#                 "nama makanan": class_name,
#                 "kalori": calories
#             })
        
#         return results

# # class ImagePredictor:
# #     def __init__(self, model_path, class_to_calories, threshold=0.1):
# #         self.model = tf.keras.models.load_model(model_path)
# #         self.class_to_calories = class_to_calories
# #         self.threshold = threshold
    
# #     def load_and_preprocess_image(self, img_path, target_size=(224, 224)):
# #         # Load the image
# #         img = image.load_img(img_path, target_size=target_size)
# #         img_array = image.img_to_array(img)
# #         img_array = np.expand_dims(img_array, axis=0)
# #         img_array /= 255.0
# #         return img_array
    
# #     def predict_calories(self, img_path):
# #         img_array = self.load_and_preprocess_image(img_path)
# #         predictions = self.model.predict(img_array)[0]  # Get the first (and only) prediction
# #         result = {}
        
# #         for idx, pred in enumerate(predictions):
# #             if pred >= self.threshold:
# #                 class_name = list(self.class_to_calories.keys())[idx]
# #                 calories = self.class_to_calories[class_name]
# #                 result[class_name] = calories
        
# #         if not result:
# #             max_idx = np.argmax(predictions)
# #             class_name = list(self.class_to_calories.keys())[max_idx]
# #             calories = self.class_to_calories[class_name]
# #             result = {class_name: calories}
        
# #         return result

# # Define the mapping from class names to calories
# class_to_calories = {
#     "Ayam_Crispy": 297,
#     "Ayam_Kecap": 223,
#     "Ayam_Serundeng": 260,
#     "Bakso": 76,
#     "Brownies": 379,
#     "Bubur_Ayam": 155,
#     "Capcay": 203,
#     "Cumi_Bakar": 185,
#     "Cumi_Hitam": 174,
#     "Cumi_Rica": 200,
#     "Dimsum_Ikan": 112,
#     "Garang_Asem": 150,
#     "Ikan_Bakar": 126,
#     "Ikan_Goreng": 84,
#     "Kentang_Balado": 102,
#     "Kue_Bolu": 297,
#     "Nasi_Bakar": 281,
#     "Nasi_Goreng": 276,
#     "Nasi_Kuning": 95,
#     "Nasi_Merah": 110,
#     "Nasi_Rames": 155,
#     "Opor_Ayam": 163,
#     "Pancake": 227,
#     "Pecel": 270,
#     "Pepes_Ikan": 105,
#     "Perkedel_Kentang": 143,
#     "Pukis": 259,
#     "Rawon": 120,
#     "Rendang": 193,
#     "Salad_Sayur": 17,
#     "Sate_Ayam": 225,
#     "Sate_Kambing": 216,
#     "Sayur_Asem": 29,
#     "Sayur_Sop": 27,
#     "Soto_Ayam": 130,
#     "Telur_Balado": 202,
#     "Telur_Dadar": 153,
#     "Tumis_Kacang_Panjang_Tahu": 140,
#     "Tumis_Kangkung": 98,
#     "Tumis_Terong": 65,
#     "Udang_Asam_Manis": 269,
#     "Udang_Goreng_Tepung": 287
# }

# # Initialize the ImagePredictor
# predictor = ImagePredictor("modelw3.h5", class_to_calories)

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template_string('''
#         <!doctype html>
#         <title>Image Upload</title>
#         <h1>Upload an image to predict calories</h1>
#         <form method=post enctype=multipart/form-data action="/predict">
#           <input type=file name=file>
#           <input type=submit value=Upload>
#         </form>
#     ''')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"})
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return jsonify({"error": "No selected file"})
    
#     if file:
#         # Save the file temporarily
#         file_path = "temp_image.jpg"
#         file.save(file_path)
        
#         # Predict the calories
#         predictions = predictor.predict_calories(file_path)
        
#         # Optionally remove the temporary file if needed
#         # os.remove(file_path)
        
#         return jsonify(predictions)

# if __name__ == '__main__':
#     app.run(debug=True)
