print("Loading Modules.....")
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import neuralNetworkDetection as net
from torchvision.utils import make_grid
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import parsing as par
print("Modules Loaded")



def partition(image,calcContourImage = False,calcBoxedImage = False):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #Grayscale
    _, thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY) #Binary

    contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # Make contours

    contourImage = image.copy()

    #Merge closeby contours
    DIST_THRESHOLD = 2
    def contours_are_close(cnt1, cnt2, threshold):
        cnt1 = cnt1.reshape(-1, 2)  
        for p in cnt2.reshape(-1, 2):  
            if cv2.pointPolygonTest(cnt1, (int(p[0]), int(p[1])), True) > -threshold:
                return True
        return False
    merged_contours = [] 
    used = set()
    for i, c1 in enumerate(contours):
        if i in used:
            continue
        merged = [c1]
        for j, c2 in enumerate(contours):
            if i != j and j not in used and (contours_are_close(c1,c2,DIST_THRESHOLD) or contours_are_close(c2,c1,DIST_THRESHOLD)):
                merged.append(c2)
                used.add(j)
        merged_points = np.vstack(merged)
        hull = cv2.convexHull(merged_points)
        merged_contours.append(hull)

    for contour in merged_contours:
        color = [random.randint(0, 255) for _ in range(3)]  
        cv2.drawContours(contourImage, [contour], -1, color, 2)


    boxedImage = image.copy()

    allExtracted = []

    points = []

    #Extract Images with max and min of contours.
    for contour in merged_contours:
        x_coords = contour[:, 0, 0]
        y_coords = contour[:, 0, 1]  

        min_x = np.min(x_coords)
        max_x = np.max(x_coords)
        min_y = np.min(y_coords)
        max_y = np.max(y_coords)


        point = np.array([[max_x, max_y],
                      [max_x, min_y],
                      [min_x, min_y],
                      [min_x, max_y]])
        
        points.append([min_x, min_y, max_x, max_y])
        
        point = [point.reshape((4, 1, 2))]
        
        if(calcBoxedImage):
            color = [random.randint(0, 255) for _ in range(3)] 
            cv2.drawContours(boxedImage,point,-1,color,2)
            allExtracted.append(image[min_y:max_y+1,min_x:max_x+1])

    return allExtracted,contourImage,boxedImage,points

if __name__ == "__main__":
    image_path = "images/image2.png"

    image = cv2.imread(image_path)

    extractedImages,contourImage,boxedImage,points = partition(image,True,True)

    
    MODEL_PATH = "Model/Resnet.pth"
    DEVICE = "cuda"

    BATCH_SIZE = 9

    IMAGE_HEIGHT,IMAGE_WIDTH,COLOR_CHANNELS = 64,64,3

    data_transform = transforms.Compose([transforms.Resize((IMAGE_HEIGHT,IMAGE_WIDTH)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    classes = ['0','1','2','3','4','5','6','7','8','9','+','.','/','=','*','-','x','y','z']


    Resnet50 = net.ResNet(net.block,[2, 4, 3],3,16).to(DEVICE)
    Resnet50.load_state_dict(torch.load(MODEL_PATH,weights_only=True))

    plt.figure(figsize=(10,10))
    Resnet50.eval()

    cv2.imshow("ContourImage",contourImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow("BoxedImage",boxedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    predictions = []

    with torch.no_grad():
        for i,exImg in enumerate(extractedImages):
            exProsImg = cv2.cvtColor(exImg, cv2.COLOR_BGR2RGB)
            exProsImg = Image.fromarray(exProsImg)
            exProsImg = data_transform(exProsImg)
            prediction = classes[Resnet50(exProsImg.unsqueeze(0).to("cuda")).argmax(dim = 1)]
            predictions.append(prediction)
            plt.subplot(4,5,i+1)
            plt.imshow(exImg)
            plt.axis("off")
            plt.title(prediction)
        
        plt.show()

    symbols = []
    
    for i,point in enumerate(points):
        symbols.append({'bbox': point , 'type': predictions[i]})
    
    output = par.symbols_to_string(symbols)
    
    output = output.split('\n')

    for i in output:
        print(f'{i}{eval(i.split("=")[0])}')
