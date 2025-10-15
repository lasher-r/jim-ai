import torch, cv2, numpy as np
print('cuda?', torch.cuda.is_available())
img = np.full((240,320,3), 200, dtype=np.uint8)
cv2.putText(img,'Hello from Nano',(10,120),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2,cv2.LINE_AA)
cv2.imshow('Nano', img); cv2.waitKey(1500)
