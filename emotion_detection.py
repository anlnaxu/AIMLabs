from deepface import DeepFace

result = DeepFace.analyze(
    img_path="meme.jpg",
    actions=['emotion']
)
print(result)
