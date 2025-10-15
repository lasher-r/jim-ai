import torch, cv2
print('cuda?', torch.cuda.is_available())
cap = cv2.VideoCapture(0)
ok, frame = cap.read()
print('camera?', ok)
if ok:
    cv2.imwrite('frame.jpg', frame)
    print('saved: frame.jpg')
cap.release()
