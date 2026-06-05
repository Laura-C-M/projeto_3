import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro: Não foi possível abrir a webcam.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível ler o frame da webcam.")
            break

        # Inverte o frame horizontalmente para uma visualização espelhada
        frame = cv2.flip(frame, 1)
        # Converte a imagem BGR para RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processa a imagem para detetar mãos
        results = hands.process(img_rgb)

        # Desenha os landmarks das mãos, se detetadas
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Deteccao de Maos", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()