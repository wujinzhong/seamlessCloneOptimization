import cv2
from SeamlessClone import SeamlessClone
import numpy as np

def test_seamless_clone():
    seamless_clone = SeamlessClone();

    images = ["./images/airplane.jpg", "./images/sky.jpg",  ]
    center = [800, 150,]
    
    for i in range(len(images)//2):
        print(images[i*2+0])
        print(images[i*2+1])
        face = cv2.imread(images[i*2+0]);
        body = cv2.imread(images[i*2+1]);
        mask = np.full((face.shape[0], face.shape[1], 1), 255, dtype=np.uint8);

        centerX=center[i*2+0]
        centerY=center[i*2+1]
        gpu_id=1
        seamless_clone.loadMatsInSeamlessClone( face, body, mask, centerX, centerY, gpu_id );
        blendedMat = seamless_clone.seamlessClone();
        blend_file = "./output/blendedMat_{}.jpg".format(i);
        cv2.imwrite(blend_file, blendedMat);

if __name__ == "__main__":
    test_seamless_clone()

