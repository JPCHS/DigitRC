import tkinter as tk
import time
from PIL import Image, ImageTk,ImageDraw,ImageFont,ImageEnhance,ImageOps
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def predict_digit(image):
    image = image.convert('L')
    image = np.array(image, dtype=np.uint8)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 50, 5, padding=1)
            self.fc1 = nn.Linear(5000, 1000)
            self.fc2 = nn.Linear(1000, 200)
            self.fc3 = nn.Linear(200, 10)
        def forward(self, x):
            in_size = x.size(0)
            out = self.conv1(x)
            out = F.relu(out)
            out = F.max_pool2d(out, 2, 2)
            out = self.conv2(out)
            out = F.relu(out)
            out = out.view(in_size, -1)
            out = self.fc1(out)
            out = F.relu(out)
            out = self.fc2(out)
            out = F.relu(out)
            out = self.fc3(out)
            out = F.log_softmax(out, dim=1)
            return out
    model = ConvNet()
    model.load_state_dict(torch.load('EMNIST-65_EPOCHS.pth'))
    def preprocess_image(image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        image = image.convert('L')
        image = ImageEnhance.Contrast(image).enhance(1000)
        image = ImageOps.equalize(image)
        image = np.array(image)
        if image[0][0] > 100:
            image = 255 - image
        image = Image.fromarray(image)
        width, height = image.size
        square_size = max(width, height)
        image_ = Image.new('L', (square_size, square_size), 0)
        image_.paste(image, ((square_size - width) // 2, (square_size - height) // 2))
        image = image_
        image = image.resize([24, 24])
        image = np.array(image)
        image[image > 5] = 255
        image = Image.fromarray(image)
        image = image.transpose(Image.Transpose.TRANSPOSE)
        image_ = Image.new('L', (28, 28), 0)
        image_.paste(image, (2, 2))
        image_tensor = transform(image_).unsqueeze(0)
        return image_tensor
    def predict_image(model, image_tensor):
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            return predicted.item()
    def merge_rectangles(rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        x = min(x1, x2)
        y = min(y1, y2)
        right = max(x1 + w1, x2 + w2)
        bottom = max(y1 + h1, y2 + h2)
        w = right - x
        h = bottom - y
        return (x, y, w, h)
    def crop(image):
        _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = []
        for contour in contours:
            continue_ = False
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 or h > 10:
                for i, r in enumerate(rectangles):
                    r = r[0]
                    if (r[0] + r[2] - x > r[2] / 2 and x + w - r[0] > 0) or (
                            x + w - r[0] > w / 2 and r[0] + r[2] - x > 0):
                        a = merge_rectangles(r, (x, y, w, h))
                        rectangles[i] = (a, image[a[1]:a[1] + a[3], a[0]:a[0] + a[2]])
                        continue_ = True
                if continue_:
                    continue
                rectangles.append(((x, y, w, h), image[y:y + h, x:x + w]))
        while True:
            rectangles_ = rectangles
            rectangles = []
            merge_ = True
            for rg in rectangles_:
                continue_ = False
                x, y, w, h = rg[0]
                for i, r in enumerate(rectangles):
                    r = r[0]
                    if (r[0] + r[2] - x > r[2] / 2 and x + w - r[0] > 0) or (
                            x + w - r[0] > w / 2 and r[0] + r[2] - x > 0):
                        a = merge_rectangles(r, (x, y, w, h))
                        rectangles[i] = (a, image[a[1]:a[1] + a[3], a[0]:a[0] + a[2]])
                        continue_ = True
                        merge_ = False
                if continue_:
                    continue
                rectangles.append(((x, y, w, h), image[y:y + h, x:x + w]))
            if merge_:
                break
        sorted_rectangles = sorted(rectangles, key=lambda item: item[0][0])
        return [Image.fromarray(cv2.cvtColor(i[1], cv2.COLOR_BGR2RGB)) for i in sorted_rectangles]
    digit = ""
    for i in crop(image):
        image_tensor = preprocess_image(i)
        digit += str(predict_image(model, image_tensor.to(DEVICE)))
    return digit
def image_digit(number):
    image = Image.new('RGB', (400, 300), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("times.ttf",150)
    text_width, text_height = draw.textsize(number, font=font)
    position = ((400 - text_width) / 2, (300 - text_height) / 2)
    draw.text(position, number, fill='black', font=font)
    return image
class Main:
    last_x, last_y = None, None
    timer_id = None
    isDeletable=False
    image=None
def save_canvas(canvas):
    image = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(image)

    for item in canvas.find_all():
        coords = canvas.coords(item)
        fill = canvas.itemcget(item, "fill")
        draw.line(coords, fill=fill, width=2)
    Main.image=image
def fade_out(canvas, photo_image, after_image):
    if photo_image.size != after_image.size:
        after_image = after_image.resize(photo_image.size)
    if photo_image.mode != after_image.mode:
        after_image = after_image.convert(photo_image.mode)
    alpha = 1.0
    while alpha > 0:
        faded_image = photo_image.copy()
        faded_image.putalpha(int(255 * alpha))
        tk_image = ImageTk.PhotoImage(image=faded_image)
        canvas.create_image(200, 150, image=tk_image)
        canvas.image = tk_image
        canvas.update()
        alpha -= 0.1
        time.sleep(0.05)
    after_tk_image = ImageTk.PhotoImage(image=after_image)
    canvas.create_image(200, 150, image=after_tk_image)
    canvas.image = after_tk_image
    canvas.update()
    Main.isDeletable=True
def process_drawing(canvas):
    save_canvas(canvas)
    canvas.delete('all')
    image1 = Main.image
    image2=image_digit(predict_digit(image1))
    fade_out(canvas, image1, image2)
def start_timer(canvas):
    Main.timer_id = canvas.after(1000, lambda: process_drawing(canvas))
def restart_timer(canvas):
    if Main.timer_id is not None:
        canvas.after_cancel(Main.timer_id)
    start_timer(canvas)
def main():
    root = tk.Tk()
    root.title('DigitRC')
    root.iconphoto(False,ImageTk.PhotoImage(Image.open("DR.ico")))
    root.resizable(False, False)
    canvas = tk.Canvas(root, width=400, height=300, bg='white')
    canvas.pack()
    def on_click(event):
        if(Main.isDeletable):
            canvas.delete("all")
        Main.isDeletable=False
        canvas.bind('<B1-Motion>', draw)
    def on_release(event):
        canvas.unbind('<B1-Motion>')
        Main.last_x=None
        Main.last_y=None
        restart_timer(canvas)
    def draw(event):
        x, y = event.x, event.y
        if Main.last_x is not None and Main.last_y is not None:
            canvas.create_line(Main.last_x, Main.last_y, x, y, fill='black', width=2)
        Main.last_x, Main.last_y = x, y
        restart_timer(canvas)  # 用户继续绘制，重置计时器
    canvas.bind('<Button-1>', on_click)
    canvas.bind('<ButtonRelease-1>', on_release)
    root.mainloop()
if __name__ == "__main__":
    main()
