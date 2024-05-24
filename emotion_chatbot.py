import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from vit_pytorch import ViT
from torchvision.models import densenet121
import google.generativeai as genai
import numpy as np
import threading

# Google API key 설정
GOOGLE_API_KEY = "AIzaSyBnJfK79vjZR-rqdp0Rz6Dhu1Bkw9lF8p4"
genai.configure(api_key=GOOGLE_API_KEY)
model_api = genai.GenerativeModel('gemini-pro')
chat = model_api.start_chat(history=[])

# 하이퍼파라미터 설정
num_classes = 4  # 감정 클래스 수

# 데이터 전처리 설정
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 모델 정의: DenseNet-121 + ViT
class HybridModel(nn.Module):
    def __init__(self, num_classes):
        super(HybridModel, self).__init__()
        self.densenet = densenet121(pretrained=False)
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()  # Fully connected layer 제거
        
        self.vit = ViT(
            image_size=224,
            patch_size=32,
            num_classes=1024,  # 중간 차원으로 1024 사용
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        
        self.fc = nn.Linear(num_ftrs + 1024, num_classes)  # 결합된 특성에 맞게 조정

    def forward(self, x):
        densenet_features = self.densenet(x)  # Shape: (batch_size, num_ftrs)
        vit_features = self.vit(x)  # Shape: (batch_size, 1024)
        combined_features = torch.cat((densenet_features, vit_features), dim=1)  # Shape: (batch_size, num_ftrs + 1024)
        out = self.fc(combined_features)
        return out

# 모델 초기화 및 가중치 로드
device = torch.device("cpu")
model = HybridModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load('best_hybrid_model_densenet.pth', map_location=device))
model.eval()

# OpenCV 웹캠 초기화
cap = cv2.VideoCapture(0)

# 감정 레이블
emotion_labels = ['anger', 'happy', 'panic', 'sadness']

# 감정 및 프레임 감지 함수
def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    emotions = []
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = transform(face_img).unsqueeze(0).to(device)
        outputs = model(face_img)
        _, predicted = torch.max(outputs, 1)
        emotion = emotion_labels[predicted.item()]
        emotions.append(emotion)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return emotions, frame

# 얼굴 감지기 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 채팅 입력을 처리하는 스레드 함수
def handle_chat_input():
    while True:
        chat_input = input("Enter your chat: ")
        if emotions:
            prompt = f"My emotion today is {', '.join(emotions)}. {chat_input} \n\n 이제 이 상황에 맞는 전문적인 심리상담가의 대화 응답을 작성해주세요."
            response = chat.send_message(prompt)
            print(response.text)

# 채팅 입력 스레드 시작
chat_thread = threading.Thread(target=handle_chat_input, daemon=True)
chat_thread.start()

# 웹캠에서 프레임 읽기
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    emotions, frame = detect_emotion(frame)
    
    # 프레임 표시
    cv2.imshow('Emotion Detection', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()