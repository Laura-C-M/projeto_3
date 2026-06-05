import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_ttacking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

#captura de video
cap = cv2.VideoCapture(0)
#Carrega a imagem do disco
#Certifique-se de que o caminho da imagem está correto
if not cap.isOpened():
    print("Erro: Não foi possível abrir a webcam.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro: nao foi possivel ler o frame da webcam.")
            break
        #inverte o frame horizontalmente para uma visualização espelhada
        frame= cv2.flip(frame,1)
        #converte a imagem para rgb
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BG2BGR)
        
        #Processsa a imagem para detetar maos
        results = hands.process(img_rgb)

    #Desenha os landmarks das maos
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                 mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
        cv2.imshow("Deteccao de Maos", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()