import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk
import threading

class ViolenceDetectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Violence Detection Interface")
        master.geometry("800x600")

        # Load model
        try:
            self.model = load_model('violence_detection_model.h5')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            master.quit()

        # Create GUI components
        self.label = tk.Label(master, text="Select a video file for violence detection", font=("Arial", 16))
        self.label.pack(pady=20)

        self.canvas = tk.Canvas(master, width=640, height=360, bg="black")
        self.canvas.pack(pady=10)

        self.controls_frame = tk.Frame(master)
        self.controls_frame.pack()

        self.play_button = tk.Button(self.controls_frame, text="Play", command=self.play_video)
        self.play_button.pack(side="left", padx=10)

        self.pause_button = tk.Button(self.controls_frame, text="Pause", command=self.pause_video)
        self.pause_button.pack(side="left", padx=10)

        self.select_button = tk.Button(master, text="Select Video", command=self.select_video, font=("Arial", 14))
        self.select_button.pack(pady=20)

        self.result_label = tk.Label(master, text="", font=("Arial", 14), fg="blue")
        self.result_label.pack(pady=20)

        # Video playback attributes
        self.video_path = None
        self.cap = None
        self.playing = False
        self.current_frame = None

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if self.video_path:
            self.play_video()
            self.process_video()

    def play_video(self):
        if not self.video_path:
            messagebox.showwarning("Warning", "Please select a video first")
            return

        self.cap = cv2.VideoCapture(self.video_path)
        self.playing = True
        self.update_video_frame()

    def pause_video(self):
        self.playing = False

    def update_video_frame(self):
        if self.cap and self.playing:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (640, 360))
                self.current_frame = ImageTk.PhotoImage(Image.fromarray(frame))
                self.canvas.create_image(0, 0, anchor="nw", image=self.current_frame)
                self.master.after(10, self.update_video_frame)
            else:
                self.cap.release()
                self.playing = False

    def process_video(self):
        def process():
            frames = self.preprocess_video(self.video_path)
            if frames is not None:
                prediction = self.model.predict(np.expand_dims(frames, axis=0))
                self.master.after(0, self.display_result, prediction[0][0])
            else:
                self.master.after(0, self.display_result, None)
        
        threading.Thread(target=process).start()

    def preprocess_video(self, video_path, num_frames=30, target_size=(120, 120)):
        frames = []
        cap = cv2.VideoCapture(video_path)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            messagebox.showerror("Error", f"Video {video_path} has no frames")
            return None

        step = max(frame_count // num_frames, 1)
        for i in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, target_size)
            frame = frame.astype(np.float32) / 255.0
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if len(frames) == num_frames:
                break

        cap.release()

        if len(frames) < num_frames:
            frames.extend([np.zeros(target_size + (3,), dtype=np.float32)] * (num_frames - len(frames)))

        return np.array(frames)

    def display_result(self, prediction):
        if prediction is not None:
            result = "Violence Detected" if prediction > 0.5 else "No Violence Detected"
            confidence = prediction if prediction > 0.5 else 1 - prediction
            self.result_label.config(text=f"{result}\nConfidence: {confidence:.2%}")
        else:
            self.result_label.config(text="Error processing video")

def main():
    root = tk.Tk()
    app = ViolenceDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
