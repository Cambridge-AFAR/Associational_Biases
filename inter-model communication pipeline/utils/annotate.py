import csv
import json
import os
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from raf_utils import AGES, GENDERS, RACES

from .phase import PHASE_AGES, PHASE_GENDERS, PHASE_SKIN_TONES

# ---------------- CONFIG ----------------
# CSV_FILE = "phase_for_anno/activity_imgs.csv"  # CSV file with image paths
# ANNOTATION_FILE = "manual_annotations/phase_acts.json"  # Output annotation file

# CATEGORIES = {
#     "age": PHASE_AGES,
#     "skin tone": PHASE_SKIN_TONES,
#     "gender": PHASE_GENDERS,
# }
# CSV_FILE = "phase_for_anno/emotion_imgs.csv"  # CSV file with image paths
# ANNOTATION_FILE = "manual_annotations/phase_emos.json"  # Output annotation file

# CATEGORIES = {
#     "age": PHASE_AGES,
#     "skin tone": PHASE_SKIN_TONES,
#     "gender": PHASE_GENDERS,
# }
CSV_FILE = "collected_raf/for_manual_anno.csv"  # CSV file with image paths
ANNOTATION_FILE = "manual_annotations/raf.json"  # Output annotation file

CATEGORIES = {
    "age": AGES + ["unsure"],
    "race": RACES + ["unsure"],
    "gender": GENDERS,
}
# ----------------------------------------


class ImageAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Annotation Tool")

        self.image_entries = self.load_image_list()
        self.total_images = len(self.image_entries)

        self.annotations = self.load_annotations()

        self.label_vars = {cat: tk.StringVar() for cat in CATEGORIES}

        self._index = self.find_first_unannotated_index()

        self.setup_ui()
        self.load_image()

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self.progress_var.set(value)
        self._index = value

    def find_first_unannotated_index(self):
        for idx, (img_id, _, _) in enumerate(self.image_entries):
            if img_id not in self.annotations or not all(
                cat in self.annotations[img_id] for cat in CATEGORIES
            ):
                return idx
        return 0

    def load_image_list(self):
        entries = []
        with open(CSV_FILE, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                img_id = row["img_id"]
                source = row["source"]
                label = row.get("label", "")
                entries.append((img_id, source, label))
        return entries

    def load_annotations(self):
        if os.path.exists(ANNOTATION_FILE):
            with open(ANNOTATION_FILE, "r") as f:
                return json.load(f)
        return {}

    def save_annotations(self):
        with open(ANNOTATION_FILE, "w") as f:
            json.dump(self.annotations, f, indent=4)
        print(f"Saved annotations to {ANNOTATION_FILE}")

    def setup_ui(self):
        self.img_label = tk.Label(self.root)
        self.img_label.pack(pady=10)

        self.label_info = tk.Label(self.root, text="", font=("Arial", 12, "italic"))
        self.label_info.pack(pady=5)

        self.form_frame = tk.Frame(self.root)
        self.form_frame.pack()

        for i, (cat, options) in enumerate(CATEGORIES.items()):
            frame = tk.LabelFrame(self.form_frame, text=cat)
            frame.grid(row=0, column=i, padx=10)
            for opt in options:
                rb = ttk.Radiobutton(
                    frame, text=opt, variable=self.label_vars[cat], value=opt
                )
                rb.pack(anchor="w")

        self.progress_var = tk.IntVar()
        self.progress_var.initialize(self._index)
        self.progress = ttk.Progressbar(
            self.root,
            orient="horizontal",
            length=400,
            mode="determinate",
            variable=self.progress_var,
        )
        self.progress.pack(pady=10)
        self.progress["maximum"] = self.total_images

        self.nav_frame = tk.Frame(self.root)
        self.nav_frame.pack(pady=10)

        self.prev_btn = ttk.Button(
            self.nav_frame, text="Previous", command=self.prev_image
        )
        self.prev_btn.grid(row=0, column=0, padx=10)

        self.next_btn = ttk.Button(self.nav_frame, text="Next", command=self.next_image)
        self.next_btn.grid(row=0, column=1, padx=10)

        self.save_btn = ttk.Button(
            self.nav_frame, text="Save and Exit", command=self.save_and_exit
        )
        self.save_btn.grid(row=0, column=2, padx=10)

    def load_image(self):
        if self.index < 0 or self.index >= self.total_images:
            return

        img_id, path, label_text = self.image_entries[self.index]
        img = Image.open(path)
        img.thumbnail((500, 500))
        self.tk_img = ImageTk.PhotoImage(img)
        self.img_label.config(image=self.tk_img)
        self.label_info.config(text=f"Task Label: {label_text}")
        self.root.title(f"Annotating: {img_id} ({self.index + 1}/{self.total_images})")

        current = self.annotations.get(img_id, {})
        for cat in CATEGORIES:
            self.label_vars[cat].set(current.get(cat, ""))

    def save_current_annotation(self):
        img_id, source, _ = self.image_entries[self.index]
        annotation = {}
        for cat in CATEGORIES:
            label = self.label_vars[cat].get()
            if label:
                annotation[cat] = label

        # Extract "val" or "train" from source path
        if "val" in source:
            annotation["set"] = "val"
        elif "train" in source:
            annotation["set"] = "train"
        elif "test" in source:
            annotation["set"] = "test"
        else:
            annotation["set"] = "unknown"

        if len(annotation.keys()) > 1:
            self.annotations[img_id] = annotation
            self.save_annotations()

    def next_image(self):
        self.save_current_annotation()
        if self.index < self.total_images - 1:
            self.index += 1
            self.progress
            self.load_image()
        else:
            messagebox.showinfo("Done", "You've reached the last image.")

    def prev_image(self):
        self.save_current_annotation()
        if self.index > 0:
            self.index -= 1
            self.load_image()
        else:
            messagebox.showinfo("First Image", "You're at the first image.")

    def save_and_exit(self):
        self.save_current_annotation()
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnnotator(root)
    root.mainloop()
