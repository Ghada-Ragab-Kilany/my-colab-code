!pip -q install ultralytics

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO("yolo11n.pt")

print("System Ready")

from google.colab import output
from IPython.display import Javascript, display
from base64 import b64decode

def take_photo():
  display(Javascript("""
    async function takePhoto() {
      const video = document.createElement('video');
      const button = document.createElement('button');
      
      video.style.width = '700px';
      video.autoplay = true;
      
      button.textContent = 'TAKE PHOTO';
      button.style.fontSize = '20px';
      button.style.margin = '10px';
      
      document.body.appendChild(video);
      document.body.appendChild(button);
      
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
      
      await video.play();
      
      await new Promise(resolve => {
        button.onclick = resolve;
      });
      
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      canvas.getContext('2d').drawImage(video, 0, 0);
      
      stream.getTracks().forEach(track => track.stop());
      
      video.remove();
      button.remove();
      
      return canvas.toDataURL('image/jpeg');
    }
  """))
  
  data = output.eval_js("takePhoto()")
  
  image = b64decode(data.split(",")[1])
  
  path = "/content/elevator.jpg"
  
  with open(path, "wb") as f:
    f.write(image)
    
  print("Photo captured!")
  
  return path

take_photo()
image = cv2.imread("/content/elevator.jpg")

# Detect people only
result = model(
    image,
    classes=[0],
    verbose=False
)[0]

# Number of detected students
people = len(result.boxes)

MAX_PEOPLE = 5

print("Students detected:", people)

if people < MAX_PEOPLE:
    status = "SAFE"
elif people == MAX_PEOPLE:
    status = "FULL"
else:
    status = "OVERLOAD"

print("Elevator Status:", status)

annotated = result.plot()

cv2.putText(
    annotated,
    f"People: {people} / {MAX_PEOPLE}",
    (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255,255,255),
    3
)

cv2.putText(
    annotated,
    f"STATUS: {status}",
    (30, 100),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.2,
    (255,255,255),
    3
)

plt.figure(figsize=(12,8))

plt.imshow(
    cv2.cvtColor(
        annotated,
        cv2.COLOR_BGR2RGB
    )
)

plt.axis("off")
plt.show()