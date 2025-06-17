import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def distancia(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)

def calcular_razao_olho(landmarks, p_top, p_bottom, p_left, p_right):
    abertura_vert = distancia(landmarks[p_top], landmarks[p_bottom])
    abertura_horiz = distancia(landmarks[p_left], landmarks[p_right])
    return abertura_vert / abertura_horiz if abertura_horiz != 0 else 0

def detectar_dedos(landmarks, hand_label):
    dedos = []

    if hand_label == "Right":
        polegar = landmarks[4].x < landmarks[3].x
    else:
        polegar = landmarks[4].x > landmarks[3].x
    dedos.append(1 if polegar else 0)

    pontas = [8, 12, 16, 20]
    bases = [6, 10, 14, 18]

    for ponta, base in zip(pontas, bases):
        dedos.append(1 if landmarks[ponta].y < landmarks[base].y else 0)

    return dedos

def reconhecer_gesto(dedos):
    total = sum(dedos)

    if total == 5:
        return "aberta"
    if total == 0:
        return "fechada"
    if dedos == [0, 1, 0, 0, 0]:
        return "apontando"
    if dedos == [0, 1, 1, 0, 0]:
        return "paz"
    if dedos == [1, 0, 0, 0, 0]:
        return "joinha"
    if dedos == [1, 1, 0, 0, 1]:
        return "rock"
    if dedos == [1, 0, 0, 0, 1]:
        return "hang loose"
    if dedos == [0, 1, 0, 0, 1]:
        return "spock"
    if dedos == [1, 1, 0, 0, 0]:
        return "faz o L"
    return "desconhecido"

def reconhecer_emocao(landmarks):
    boca_abertura = distancia(landmarks[13], landmarks[14])
    sobrancelha_dir_altura = abs(landmarks[brow_inner_right].y - landmarks[brow_outer_right].y)
    sobrancelha_esq_altura = abs(landmarks[brow_inner_left].y - landmarks[brow_outer_left].y)
    media_sobrancelha = (sobrancelha_dir_altura + sobrancelha_esq_altura) / 2

    if boca_abertura > 0.06:
        return "Surpreso"
    if boca_abertura < 0.02:
        return "Neutro"
    return "Feliz"

brow_inner_right = 70
brow_outer_right = 105
brow_inner_left = 300
brow_outer_left = 334

def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detector", 520, 400)

    contador_piscada = 0
    contador_boca = 0
    olho_fechado = False
    boca_aberta = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Erro na câmera")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hands_results = hands.process(rgb)
        face_results = face_mesh.process(rgb)

        h, w, _ = frame.shape

        texto_emocao = ""

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                lm = face_landmarks.landmark

                razao_olho_dir = calcular_razao_olho(lm, 159, 145, 33, 133)
                razao_olho_esq = calcular_razao_olho(lm, 386, 374, 362, 263)

                abertura_boca = distancia(lm[13], lm[14])

                pontos = [159, 145, 386, 374, 13, 14, 33, 133, 362, 263, brow_inner_right, brow_inner_left]
                for idx in pontos:
                    cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1, lineType=cv2.LINE_AA)

                media_olhos = (razao_olho_dir + razao_olho_esq) / 2
                if media_olhos < 0.25:
                    if not olho_fechado:
                        contador_piscada += 1
                        olho_fechado = True
                else:
                    olho_fechado = False

                if abertura_boca > 0.05:
                    if not boca_aberta:
                        contador_boca += 1
                        boca_aberta = True
                else:
                    boca_aberta = False

                texto_emocao = reconhecer_emocao(lm)

        gestos_maos = []

        if hands_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                hand_label = hands_results.multi_handedness[idx].classification[0].label

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
                )

                dedos = detectar_dedos(hand_landmarks.landmark, hand_label)
                gesto = reconhecer_gesto(dedos)
                gestos_maos.append(gesto)

                cv2.putText(
                    frame,
                    f"{hand_label}: {gesto}",
                    (10, 35 + idx * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        mensagem_extra = ""
        if "apontando" in gestos_maos:
            mensagem_extra = "Mito"
        elif len(gestos_maos) == 2 and gestos_maos[0] == "aberta" and gestos_maos[1] == "aberta":
            mensagem_extra = "Absolute Cinema"

        cv2.putText(frame, f"PISCADAS: {contador_piscada}", (10, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"BOCA ABERTA: {contador_boca}", (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if texto_emocao:
            cv2.putText(frame, f"EMOCAO: {texto_emocao}", (10, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if mensagem_extra:
            cv2.putText(frame, mensagem_extra, (10, h - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Detector", frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
