import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

class MiniPhotoshop:
    def __init__(self, root):
        self.root = root
        self.root.title("Mini Photoshop - Python Image Editor")
        self.root.geometry("1000x700")
        self.root.configure(bg="#2c3e50")

        # Variables
        self.original_image = None  # Keeps the original for reset
        self.processed_image = None # The image currently being edited (OpenCV BGR format)
        self.display_image = None   # The image shown on GUI (PIL format)
        self.filepath = None

        # --- Layout Layout ---
        
        # Left Toolbar Frame
        self.toolbar = tk.Frame(root, width=200, bg="#34495e", padx=10, pady=10)
        self.toolbar.pack(side=tk.LEFT, fill=tk.Y)

        # Right Image Canvas Frame
        self.canvas_frame = tk.Frame(root, bg="#2c3e50")
        self.canvas_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.image_label = tk.Label(self.canvas_frame, bg="#2c3e50", text="Load an image to start", fg="white")
        self.image_label.pack(expand=True)

        # --- Buttons ---
        self.create_buttons()

    def create_buttons(self):
        # Helper to make stylish buttons
        def btn(text, cmd, color="#ecf0f1", txt_color="#2c3e50"):
            b = tk.Button(self.toolbar, text=text, command=cmd, bg=color, fg=txt_color, 
                          font=("Arial", 10, "bold"), bd=0, padx=10, pady=5)
            b.pack(fill=tk.X, pady=5)
            return b

        # File Operations
        tk.Label(self.toolbar, text="File Ops", bg="#34495e", fg="#95a5a6").pack(pady=(0, 5))
        btn("Open Image", self.load_image, "#3498db", "white")
        btn("Save Image", self.save_image, "#27ae60", "white")
        btn("Reset", self.reset_image, "#e74c3c", "white")

        tk.Label(self.toolbar, text="Filters", bg="#34495e", fg="#95a5a6").pack(pady=(15, 5))
        btn("Grayscale", self.convert_grayscale)
        btn("Invert Colors", self.invert_image)
        btn("Gaussian Blur", self.apply_blur)
        btn("Sharpen", self.apply_sharpen)

        tk.Label(self.toolbar, text="Advanced", bg="#34495e", fg="#95a5a6").pack(pady=(15, 5))
        btn("Hist. Equalization", self.histogram_equalization)
        btn("Edge Detection", self.detect_edges)
        btn("Segmentation", self.segmentation)
        
        tk.Label(self.toolbar, text="Geometry", bg="#34495e", fg="#95a5a6").pack(pady=(15, 5))
        btn("Rotate 90°", self.rotate_image)

    # --- Core Logic ---

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        if file_path:
            self.filepath = file_path
            # Read image using OpenCV (loads as BGR)
            img = cv2.imread(file_path)
            if img is not None:
                self.original_image = img.copy()
                self.processed_image = img.copy()
                self.show_image()
            else:
                messagebox.showerror("Error", "Could not load image.")

    def save_image(self):
        if self.processed_image is None:
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", 
                                                 filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if file_path:
            cv2.imwrite(file_path, self.processed_image)
            messagebox.showinfo("Success", "Image saved successfully!")

    def show_image(self):
        """Converts OpenCV BGR image to PIL RGB and displays it on the Label."""
        if self.processed_image is None:
            return

        # Convert BGR (OpenCV) to RGB (PIL/Tkinter)
        # Handle grayscale images which only have 2 dimensions
        if len(self.processed_image.shape) == 2:
            img_rgb = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)

        # Resize to fit the window while maintaining aspect ratio
        h, w = img_rgb.shape[:2]
        display_w, display_h = 750, 650 # Max dimensions
        
        scale = min(display_w/w, display_h/h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        image_pil = Image.fromarray(img_rgb)
        image_pil = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        self.tk_image = ImageTk.PhotoImage(image_pil)
        self.image_label.config(image=self.tk_image, text="")

    def reset_image(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.show_image()

    # --- Processing Functions ---

    def convert_grayscale(self):
        if self.processed_image is not None:
            if len(self.processed_image.shape) == 3:
                self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
                self.show_image()

    def invert_image(self):
        if self.processed_image is not None:
            self.processed_image = cv2.bitwise_not(self.processed_image)
            self.show_image()

    def apply_blur(self):
        if self.processed_image is not None:
            # Gaussian Blur with a 15x15 kernel
            self.processed_image = cv2.GaussianBlur(self.processed_image, (15, 15), 0)
            self.show_image()

    def apply_sharpen(self):
        if self.processed_image is not None:
            # Sharpening kernel
            kernel = np.array([[0, -1, 0],
                               [-1, 5,-1],
                               [0, -1, 0]])
            self.processed_image = cv2.filter2D(self.processed_image, -1, kernel)
            self.show_image()

    def histogram_equalization(self):
        """ Improves contrast. Handles both Grayscale and Color images appropriately. """
        if self.processed_image is not None:
            if len(self.processed_image.shape) == 2:
                # Grayscale
                self.processed_image = cv2.equalizeHist(self.processed_image)
            else:
                # Color: Convert to YUV, equalize Y (luminance), convert back
                # This prevents weird color shifts that happen if you equalize R, G, B individually
                yuv = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2YUV)
                yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                self.processed_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            self.show_image()

    def detect_edges(self):
        """ Canny Edge Detection """
        if self.processed_image is not None:
            # Edges need grayscale primarily, but Canny handles it. 
            # We usually blur slightly before edge detection to reduce noise
            blurred = cv2.GaussianBlur(self.processed_image, (5, 5), 0)
            edges = cv2.Canny(blurred, 100, 200)
            self.processed_image = edges # This is now a binary mask (grayscale)
            self.show_image()

    def segmentation(self):
        """ Simple Segmentation using Otsu's Thresholding """
        if self.processed_image is not None:
            # Convert to grayscale first if it isn't already
            if len(self.processed_image.shape) == 3:
                gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.processed_image
            
            # Apply Otsu's thresholding
            # This separates background and foreground automatically
            thresh_val, self.processed_image = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            self.show_image()

    def rotate_image(self):
        if self.processed_image is not None:
            self.processed_image = cv2.rotate(self.processed_image, cv2.ROTATE_90_CLOCKWISE)
            self.show_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = MiniPhotoshop(root)
    root.mainloop()
