import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import numpy as np
import pickle
from trainer import make_predictions


# ---- Load trained model weights ----
with open("model_params.pkl", "rb") as f:
    W1, b1, W2, b2 = pickle.load(f)

# ---- MNIST-style preprocessing ----
def preprocess(img_array):
    # Ensure it's float
    img_array = np.array(img_array)
    h, w = img_array.shape
    img_array = img_array.astype(np.float32)
    factor_h = h // 28
    factor_w = w // 28

    # Reshape and average
    img_array = img_array.reshape(28, factor_h, 28, factor_w).mean(axis=(1, 3))
    
    # Normalize to [0,1]
    img_array /= 255.0
    # Invert
    img_array = 1 - img_array  
    
    # Compute center of mass manually
    total = img_array.sum()
    if total > 0:
        y, x = np.indices(img_array.shape)
        cy = (y * img_array).sum() / total
        cx = (x * img_array).sum() / total
        # Desired center
        shift_y = int(round(img_array.shape[0] / 2 - cy))
        shift_x = int(round(img_array.shape[1] / 2 - cx))
        # Shift using np.roll
        img_array = np.roll(img_array, shift_y, axis=0)
        img_array = np.roll(img_array, shift_x, axis=1)

    # Reshape for model input
    return img_array.reshape(784, 1)

# ---- GUI Application ----
class DigitApp:
    def __init__(self, root, scale):
        self.root = root
        self.root.title("Digit Recognizer")
        self.scale = scale

        # Drawing canvas
        self.canvas = tk.Canvas(root, width=28*self.scale, height=28*self.scale, bg="white")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)

        # Button panel
        btn_frame = tk.Frame(root)
        btn_frame.pack()

        tk.Button(btn_frame, text="Predict", command=self.predict).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Clear", command=self.clear).pack(side=tk.LEFT)

        # For drawing
        self.image = Image.new("L", (28*self.scale, 28*self.scale), 255)
        self.draw_image = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        r = 8  # brush size
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
        self.draw_image.ellipse([x-r, y-r, x+r, y+r], fill=0)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (28*self.scale, 28*self.scale), 255)
        self.draw_image = ImageDraw.Draw(self.image)

    def predict(self):
        img_arr = np.array(preprocess(self.image))
        prediction = make_predictions(img_arr, W1, b1, W2, b2)

        messagebox.showinfo("Prediction", f"Predicted Digit: {prediction[0]}")
        self.clear()

# ---- Run app ----
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitApp(root, scale=10)
    root.mainloop()
