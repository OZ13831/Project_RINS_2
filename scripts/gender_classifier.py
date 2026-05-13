from retinaface import RetinaFace
from deepface import DeepFace
import matplotlib.pyplot as plt

image_path = "test.jpg"
# resp = RetinaFace.detect_faces(image_path)
image = plt.imread(image_path)

fig, ax = plt.subplots(1, 1)
ax.imshow(image)

# if isinstance(resp, dict) and resp:
#     for face_id, face_info in resp.items():
#         box = face_info.get("facial_area")
#         if not box or len(box) != 4:
#             continue
#         x1, y1, x2, y2 = box
#         width = x2 - x1
#         height = y2 - y1
#         rect = plt.Rectangle(
#             (x1, y1),
#             width,
#             height,
#             fill=False,
#             edgecolor="red",
#             linewidth=2,
#         )
#         ax.add_patch(rect)
# else:
#     print("No faces detected.")



obj = DeepFace.analyze("test.jpg", actions=['age', 'gender', 'emotion'])

if isinstance(obj, list) and obj:
    for face in obj:
        bbox = face.get("region")
        if not bbox:
            continue
        x, y, w, h = bbox.get("x"), bbox.get("y"), bbox.get("w"), bbox.get("h")
        if None in (x, y, w, h):
            continue
        gender = face.get("dominant_gender")
        confidence = None
        gender_scores = face.get("gender")
        if isinstance(gender_scores, dict) and gender in gender_scores:
            confidence = gender_scores[gender]
        rect = plt.Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            edgecolor="blue",
            linewidth=2,
        )
        ax.add_patch(rect)
        if gender:
            if confidence is not None:
                label = f"{gender} ({confidence:.1f}%)"
            else:
                label = str(gender)
            ax.text(
                x,
                max(y - 8, 0),
                label,
                color="blue",
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )
else:
    print("No faces analyzed.")

  
ax.axis("off")
plt.show()