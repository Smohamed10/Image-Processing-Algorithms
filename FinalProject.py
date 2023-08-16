import random
import tkinter as tk
import cv2
import numpy as np
from tkinter import simpledialog
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
from tqdm import tqdm
import threading
import os

pepper=255
salt=0
flagforwardfourier=0
path=''
Row = []
Column=[]
originalimage=np.array((0,0))
outputimage=np.array((0,0))
imageafterforwardshift=np.array((0,0))
imageafterforwadtrans=np.array((0,0))


root = tk.Tk()
root.geometry("1920x1080")
root.title("Image Processing Final Project")
root.configure(bg='Beige')
root.attributes('-fullscreen',True)
def vertical(temp):
    

    roberts_cross_v = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, -1, 0],
                           [0, 0, 0, 0, 0]])


    x = 0
    
    temp = temp* roberts_cross_v
    x =np.sum(temp)
    return x    


def horizontal(temp):
    roberts_cross_h = np.array([
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  0, 1,  0],
    [ 0,  0,  -1,  0,  0],
    [ 0,  0,  0,  0,  0]])

    x = 0
    
    temp = temp* roberts_cross_h
    x =np.sum(temp)
    return x


def clear():
    global path,Column,originalimage,outputimage,Row
    path=''
    Row.clear()
    Column.clear()
    originalimage=np.array((0,0))
    outputimage=np.array((0,0))   


def normalize():
    global outputimage
        #normalize  
    for x in range(0,outputimage.shape[0]):
        for y in range(0,outputimage.shape[1]):  
            if (outputimage[x,y] < 0): 
             outputimage[x,y] = 0    
            if (outputimage[x,y] > 255):
             outputimage[x,y] = 255 
    return outputimage         
    

def padding():
    global originalimage
    global Row
    global Column
    for i in range(originalimage.shape[0]): # Column 3la 2d height elsora
        Column.append(0)
        #col 4mal w ymen b zeros
    for i in range(4):    
     originalimage=np.insert(originalimage,0, Column, axis=1)
     originalimage=np.append(originalimage, np.expand_dims(Column, axis=1), axis=1)
        
    for i in range(originalimage.shape[1]): # row 3la 2d width elsora
        Row.append(0)
        #row ta7t w fo2 b zeros 
    for i in range(4):
     originalimage=np.insert(originalimage,0, Row, axis=0)
     originalimage=np.append(originalimage,[Row], axis=0)    


def view():
    global originalimage
    global outputimage
    #el display 34an ykono gnb ba3d
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(originalimage, cmap='gray')
    ax[0].set_title('The Original Image')
    ax[1].imshow(outputimage, cmap='gray')
    ax[1].set_title('The Modified Image')  
    plt.show()


def Get_Blurred_Image(originalimage):
    
    blurred=originalimage.copy()
    # gaussian filter
    sigma=np.std(originalimage)
    kernel = np.zeros((5, 5))
    for i in range(5):           # w1 = h(-2, -2), w2 = h(-1, 0), .......... w25 = h(2, 2). mn el slides
        for j in range(5):
            x=i-2
            y=j-2
            kernel[i, j] = (1/(2*(sigma**2)*3.14)*(np.exp(-(x**2 + y**2) / (2 * sigma**2))))

    kernel = kernel/np.sum(kernel)
    for x in range(2,originalimage.shape[0]-2):
        for y in range(2,originalimage.shape[1]-2):  
            blurred[x,y]=np.sum(originalimage[x-2:x+3, y-2:y+3]*kernel)         
    return blurred

    
def reset():
 global originalimage,path,Row,Column
 Row.clear()
 Column.clear()
 originalimage=cv2.imread(path,0)


def choosefile():  
    clear()
    global path
    global originalimage
    path = askopenfilename(title="Select an Image File", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    originalimage=cv2.imread(path,0)
    if path=='':
     tk.messagebox.showerror("Error","You didn't select any image")
     path = askopenfilename(title="Select an Image File", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
     originalimage=cv2.imread(path,0)


def MedianFilter():
        global originalimage
        global outputimage
        outputimage=originalimage.copy()
      
        Ks=3
        for i in range (0,outputimage.shape[0]-1):
            for j in range (0,outputimage.shape[1]-1):
                pixels = outputimage[(i,j)]
                neig = outputimage[i : i + Ks ,j : j + Ks ]
                L_Median=np.median(neig)
                outputimage[(i,j)]= L_Median
        view()
        reset() 


def AdabtiveFilter():

    global originalimage
    global outputimage
    m,n=originalimage.shape
    outputimage=np.zeros((m, n))
    value = int(userinput(message="Enter The Threshold"))
    kernal_Size = 3
    MaxValue = 0

    for i in range(originalimage.shape[0]-1):
        for j in range(originalimage.shape[1]-1):
            
            pixels = originalimage[(i,j)]
            neig = originalimage[i : i + kernal_Size ,j : j + kernal_Size ]
                  
            MaxValue = np.max(neig) 

            outputimage[(i,j)] = max(MaxValue - value, 0)
    
    view()
    reset()


def averagingfilter():
   
    global originalimage
    global outputimage

    col = originalimage.shape[1] + 2
    rows = originalimage.shape[0] + 2
    pad = np.array([[1] * col] * rows)
    newimage = np.zeros(originalimage.shape, dtype=np.uint8)

    for i in range(1, originalimage.shape[0] - 2):
        for j in range(1, originalimage.shape[1] - 2):
            pad[i][j] = originalimage[i - 2][j - 2]

    for r in range(0, pad.shape[0]):
        for c in range(0, pad.shape[1]):
            # Averaging
            avg = 0
            if r + 4 < pad.shape[0] and c + 4 < pad.shape[1]:
                for i in range(r, r + 4):
                    for j in range(c, c + 4):
                        avg += pad[i][j]
                m = avg * 1 / 25
                if m < 0:
                    m = 0
                if m > 255:
                    m = 255
                newimage[r][c] = m

        # Normalize the pixel values
    outputimage= cv2.normalize(newimage, None, 0, 255, cv2.NORM_MINMAX)

    view()
    reset()  


def GaussianFilter():
    global originalimage
    global outputimage
    padding()
    outputimage=originalimage.copy()
    blurred=originalimage.copy()
    # gaussian filter
    sigma=np.std(outputimage)
    kernel = np.zeros((5, 5))
    for i in range(5):           # w1 = h(-2, -2), w2 = h(-1, 0), .......... w25 = h(2, 2). mn el slides
        for j in range(5):
            x=i
            y=j
            kernel[i, j] = (np.exp(-(x**2 + y**2) / (2 * sigma**2)))

    kernel = kernel/np.sum(kernel)
    for x in range(2,outputimage.shape[0]-2):
        for y in range(2,outputimage.shape[1]-2):  
            blurred[x,y]=np.sum(originalimage[x-2:x+3, y-2:y+3]*kernel)

    outputimage=blurred  
    outputimage=normalize()            
    view()
    reset()  

           
def Laplacian():
    global originalimage
    global outputimage
    padding()
    outputimage=originalimage.copy()
    operator=np.array([[0, 0 , -1 , 0 , 0],
                       [0, -1 , -2 , -1 , 0],
                       [-1, -2 , 16 , -2 , -1],
                       [0, -1 , -2 , -1 , 0],
                       [0, 0 , -1 , 0 , 0]])
    for x in range(2,outputimage.shape[0]-2):
        for y in range(2,outputimage.shape[1]-2):  
            outputimage[x,y]= np.sum(outputimage[x-2:x+3, y-2:y+3] * operator)
            if (outputimage[x,y] < 0): 
             outputimage[x,y] = 0    
            if (outputimage[x,y] > 255):
             outputimage[x,y] = 255 
        
                     
    view()
    reset()                  


def UnsharpMasking():
    global originalimage
    global outputimage
    padding()
    outputimage=originalimage.copy()
    blurred=Get_Blurred_Image(originalimage)
    #unsharp masking
    message="Enter Value Of K  "
    k = float(userinput(message))

    mask=(originalimage-blurred)  
    outputimage=originalimage+k*mask 

    #normalize  
    outputimage=normalize()            

    view()
    reset()      


def Roberts_Cross_Gradient_Operators():
    global originalimage
    global outputimage
    originalimage=cv2.imread(path,0).astype('float64')
    originalimage/=128.0
    outputimage=originalimage.copy()
    Gx =originalimage.copy()
    Gy=originalimage.copy()
    col=originalimage.shape[1]+4
    rows=originalimage.shape[0]+4
    pad=np.array([[0]*col]*rows)
    temp=np.array([[0]*5]*5)
    
    

    for i in range (2,originalimage.shape[0]-2):
        for j in range(2,originalimage.shape[1]-2):
         pad[i][j]=originalimage[i-1][j-1]
        
    for r in range (2 , pad.shape[0]-2):
        for c in range (2,pad.shape[1]-2):
            
            temp=pad[r-2:r+3,c-2:c+3]
            y= vertical(temp)
            z=horizontal(temp)
            Gx[r-2][c-2]=y
            Gy[r-2][c-2]=z
                    

    outputimage= np.sqrt( np.square(Gy) + np.square(Gx))

    view()
    reset()


def Sobel_Operation():
    global originalimage
    global outputimage
    m,n=originalimage.shape
    outputimage=np.zeros((m, n))
    kernal_Size = 3

    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    for i in range(originalimage.shape[0] - 2):
        for j in range(originalimage.shape[1] - 2): 
            neig = originalimage[i:i + kernal_Size, j:j + kernal_Size]

            gx = np.sum(sx * neig)
            gy = np.sum(sy * neig)
            gradient = np.sqrt(gx ** 2 + gy ** 2)
            outputimage[i+1 , j+1 ] = gradient


    view()
    reset()


def add_salt_and_pepper_noise():
    global salt
    global pepper
    global outputimage
    global originalimage

    m,n=originalimage.shape
    outputimage=np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if np.random.random() < 0.5:
                outputimage[i, j] = salt
            else:
                outputimage[i, j] = pepper

    outputimage = outputimage + originalimage
                
    view()
    reset()


def gaussian_noise():
    global originalimage
    global outputimage
    Gstandard = 50
    mean = 40
    
    noise = np.zeros_like(originalimage, dtype=np.float32)

    for i in range(originalimage.shape[0]):
        for j in range(originalimage.shape[1]):
            z = int(random.uniform(mean-2*Gstandard, mean+2*Gstandard))
            noise[i][j] = z

    noisy_image = originalimage.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    outputimage = noisy_image.astype(np.uint8)

    view()
    reset()


def uniform_noise():
    global originalimage
    global outputimage
    a=35
    b=100 
   
    noise = np.zeros_like(originalimage, dtype=np.float32)

    for i in range(originalimage.shape[0]):
        for j in range(originalimage.shape[1]):
            z =int(random.uniform(0, 255))

            if a <= z <= b:
                noise[i, j] =1/a-b
            else:
                noise[i, j] = 0

    noisy_image = originalimage.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    outputimage = noisy_image.astype(np.uint8)
   
    view()
    reset()


def Histogram_Equalization():
    global originalimage
    global outputimage
    intensityNum = np.zeros(256,dtype=np.int)
    intensityProb = np.zeros(256,dtype=np.float64)
    intensityDis = np.zeros(256,dtype=np.float64)
    EqualizedValues = np.zeros(256,dtype=np.float64)

    m,n=originalimage.shape
    outputimage=np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            intensityVal = originalimage[i, j]
            intensityNum[intensityVal] += 1
    for i in range (len(intensityNum)):
        intensityProb[i]= intensityNum[i]/(m*n)

    draw_Normalized_Histogram(intensityProb)

    for i in tqdm(range (len(intensityProb))):
        intensityDis[i]= 255* np.sum(intensityProb[0:i])
        intensityDis[i]=np.round(intensityDis[i])



        #print(str(intensityDis[i])+"= 255 * "+str(np.round(np.sum(intensityProb[0:i]))))
        #print('#') 
    draw_Transformation_Histogram(intensityDis)

    for i in range (len(intensityDis)):
        EqualizedValues[int(intensityDis[i])]+=intensityProb[i]
        #print('#') 
    # Display the graph
    draw_Equalized_Histogram(intensityDis,EqualizedValues)

    for i in tqdm(range(256)):
        for k in range(m):
            for l in range(n):
                if  originalimage[k,l]==i:
                        outputimage[k,l]=intensityDis[i]
    view()
    reset()


def Histogram_Specification():
    global originalimage
    global outputimage
    global originalimage
    min=9999999999
    Specifiedpath = askopenfilename(title="Select an Image File", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    Specified=cv2.imread(Specifiedpath,0)
    intensityNum = np.zeros(256,dtype=np.int)
    intensityProb = np.zeros(256,dtype=np.float64)
    intensityDis = np.zeros(256,dtype=np.float64)
    EqualizedValues = np.zeros(256,dtype=np.float64)

    intensityNum_Specified = np.zeros(256,dtype=np.int)
    intensityProb_Specified = np.zeros(256,dtype=np.float64)
    intensityDis_Specified = np.zeros(256,dtype=np.float64)
    EqualizedValues_Specified = np.zeros(256,dtype=np.float64)

    m,n=originalimage.shape
    m2,n2=Specified.shape
    outputimage=np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            intensityVal = originalimage[i, j]
            intensityNum[intensityVal] += 1
    for i in range (len(intensityNum)):
        intensityProb[i]= intensityNum[i]/(m*n)


    for i in tqdm(range (len(intensityProb))):
        intensityDis[i]= 255* np.sum(intensityProb[0:i])
        intensityDis[i]=np.round(intensityDis[i])

    for i in range (len(intensityDis)):
        EqualizedValues[int(intensityDis[i])]+=intensityProb[i]


    draw_Normalized_Histogram(intensityProb)
    draw_Transformation_Histogram(intensityDis)
    draw_Equalized_Histogram(intensityDis,EqualizedValues)
######################################################################### FOR SPECIFIED ############################################################################################

    for i in range(m2):
        for j in range(n2):
            intensityVal = Specified[i, j]
            intensityNum_Specified[intensityVal] += 1
    for i in range (len(intensityNum_Specified)):
        intensityProb_Specified[i]= intensityNum_Specified[i]/(m2*n2)


    for i in tqdm(range (len(intensityProb_Specified))):
        intensityDis_Specified[i]= 255* np.sum(intensityProb_Specified[0:i])
        intensityDis_Specified[i]=np.round(intensityDis_Specified[i])

    for i in range (len(intensityDis_Specified)):
        EqualizedValues_Specified[int(intensityDis_Specified[i])]+=intensityProb_Specified[i]

    draw_Normalized_Histogram(intensityProb_Specified)
    draw_Transformation_Histogram(intensityDis_Specified)
    draw_Equalized_Histogram(intensityDis_Specified,EqualizedValues_Specified)

    for i in range(len(intensityDis)):
        min=9999999
        for k in range(len(intensityDis_Specified)):
                diff = abs(intensityDis[i] - intensityDis_Specified[k])
                if diff <= min:
                    min = diff
                    val=k
        intensityDis[i]=val


    for i in tqdm(range(256)):
        for k in range(m):
            for l in range(n):
                if  originalimage[k,l]==i:
                        outputimage[k,l]=intensityDis[i]

    draw_Transformation_Histogram(intensityDis)

    view()
    reset()


def ForwardFouriertransform():
    global originalimage
    global outputimage
    global imageafterforwardshift
    global imageafterforwadtrans
    global flagforwardfourier
    m, n = originalimage.shape
    outputimage = np.zeros((m, n))
    imageafterforwardshift=np.zeros((m, n))
    mnew = int(m / 6)
    
    def loop(start, end):
        for u in tqdm(range(start, end)):
            for v in range(n):
                x = np.arange(m)
                y = np.arange(n)
                u_x = (u * x) / m
                v_y = (v * y) / n
                exponential = np.exp(-1j * 2 * np.pi * (u_x[:, None] + v_y[None, :]))
                sum_val = np.sum(originalimage * exponential)
                outputimage[u, v] = sum_val
    
    thread1 = threading.Thread(target=loop, args=(0, mnew))
    thread2 = threading.Thread(target=loop, args=(mnew, mnew * 2))
    thread3 = threading.Thread(target=loop, args=(mnew * 2, mnew * 3))
    thread4 = threading.Thread(target=loop, args=(mnew * 3, mnew * 4))
    thread5 = threading.Thread(target=loop, args=(mnew * 4, mnew * 5))
    thread6 = threading.Thread(target=loop, args=(mnew * 5, m))
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()
    thread6.start()
    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    thread5.join()
    thread6.join()
    
    image_after_shif = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            image_after_shif[i, j] = outputimage[(i + m // 2) % m, (j + n // 2) % n]
    flagforwardfourier=1
    imageafterforwardshift=image_after_shif
    outputimage = np.log(10 + np.abs(image_after_shif))
    imageafterforwadtrans=outputimage
    current_path = os.path.dirname(os.path.abspath(__file__))
    image_filename = "Image after Forward Transformation.png"
    save_path = os.path.join(current_path, image_filename)
    plt.imsave(save_path, outputimage, cmap='gray')
    view()
    reset()
                      

def InverseFouriertransform():
    global originalimage
    global outputimage
    global imageafterforwardshift
    global imageafterforwadtrans
    global flagforwardfourier

    if(flagforwardfourier==1):
            m, n = originalimage.shape
            originalimage=imageafterforwadtrans
            
            outputimage = np.zeros((m, n))
            mnew = int(m / 6)


            image_before_shif = np.zeros((m, n))
            for i in range(m):
                for j in range(n):
                    image_before_shif[i, j] = imageafterforwardshift[(i - m // 2) % m, (j - n // 2) % n]
                    
            imagebeforeforwardshift=image_before_shif

            def loop(start, end):
                for u in tqdm(range(start, end)):
                    for v in range(n):
                        x = np.arange(m)
                        y = np.arange(n)
                        u_x = (u * x) / m
                        v_y = (v * y) / n
                        exponential = np.exp(1j * 2 * np.pi * (u_x[:, None] + v_y[None, :]))
                        sum_val = np.sum(imagebeforeforwardshift * exponential)
                        outputimage[u, v] = sum_val/(m*n)

            thread1 = threading.Thread(target=loop, args=(0, mnew))
            thread2 = threading.Thread(target=loop, args=(mnew, mnew * 2))
            thread3 = threading.Thread(target=loop, args=(mnew * 2, mnew * 3))
            thread4 = threading.Thread(target=loop, args=(mnew * 3, mnew * 4))
            thread5 = threading.Thread(target=loop, args=(mnew * 4, mnew * 5))
            thread6 = threading.Thread(target=loop, args=(mnew * 5, m))

            thread1.start()
            thread2.start()
            thread3.start()
            thread4.start()
            thread5.start()
            thread6.start()

            thread1.join()
            thread2.join()
            thread3.join()
            thread4.join()
            thread5.join()
            thread6.join()


            current_path = os.path.dirname(os.path.abspath(__file__))
            image_filename = "Image_after_Inverse_Transformation.png"
            save_path = os.path.join(current_path, image_filename)
            plt.imsave(save_path, outputimage, cmap='gray')
            view()
            reset()
            flagforwardfourier=0
    else:
       tk.messagebox.showerror("Error","Please Perform Forward Transform First")
       reset() 


def NearestNeighborInterpolation():
    global originalimage
    global outputimage

    scale_factor = float(userinput(message="Enter The Scale Factor"))

    width, height = originalimage.shape
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    outputimage = np.zeros((new_width, new_height), dtype=np.uint8)

    for i in range(new_width):
        for j in range(new_height):
            x = int(i / scale_factor)
            y = int(j / scale_factor)
            outputimage[i,j] = originalimage[x,y]

    cv2.imshow('Original Image',originalimage)
    cv2.imshow('Modified Image',outputimage)
    reset()   


def draw_Transformation_Histogram(intensityDis):
    
    plt.figure(figsize=(10, 6))
    
    # Calculate the step function values
    x = range(256)
    y = intensityDis 
    
    # Plot the step function
    plt.plot(x, y)
    
    plt.xlabel('R(k)')
    plt.ylabel('S(k)')
    plt.xticks(range(0, 256, 32), fontsize=6)
    plt.yticks(range(0, 256, 32), fontsize=8)
    plt.title('Transformation Histogram', fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()


def draw_Normalized_Histogram(intensityProb):
    # Set the figure size
    plt.figure(figsize=(10, 6))
    plt.plot(range(256), intensityProb)
    plt.xlabel('Intensity Value')
    plt.ylabel('Probability')
    plt.xticks(range(256), fontsize=6)
    plt.yticks(fontsize=8)
    plt.title('Normalized Histogram', fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()


def draw_Equalized_Histogram(intensityDis,EqualizedValues):
    # Set the figure size
    plt.figure(figsize=(10, 6))
    plt.plot(intensityDis,EqualizedValues)
    plt.xlabel('Intensity Value')
    plt.ylabel('Probability')
    plt.xticks(range(256),fontsize=6)
    plt.yticks(fontsize=8)
    plt.title('Equalized Histogram', fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()


button1 = tk.Button(root, text="Select Photo",command=choosefile, width=10, height=2,bg="Black",fg="white")
button1.pack(pady=20)


button2 = tk.Button(root, text="Apply Median Filter", command=MedianFilter, width=40, height=2,bg="Dark Red",fg="white")
button3 = tk.Button(root, text="Apply Adaptive Filter", command=AdabtiveFilter, width=40, height=2,bg="Dark Red",fg="white")
button4 = tk.Button(root, text="Apply Averaging Filter", command=averagingfilter, width=40, height=2,bg="Dark Red",fg="white")
button5 = tk.Button(root, text="Apply Gaussian Filter", command=GaussianFilter, width=40, height=2,bg="Dark Red",fg="white")
button6 = tk.Button(root, text="Apply Laplacian Operator", command=Laplacian, width=40, height=2,bg="Dark Red",fg="white")
button7 = tk.Button(root, text="Apply Unsharp Masking", command=UnsharpMasking, width=40, height=2,bg="Dark Red",fg="white")
button8 = tk.Button(root, text="Apply Roberts' Operators Filter", command=Roberts_Cross_Gradient_Operators, width=40, height=2,bg="Dark Red",fg="white")
button9 = tk.Button(root, text="Apply Sobel Operators Filter", command=Sobel_Operation, width=40, height=2,bg="Dark Red",fg="white")
button10 = tk.Button(root, text="Apply Salt and Pepper Noise", command=add_salt_and_pepper_noise, width=40, height=2,bg="Dark Red",fg="white")
button11 = tk.Button(root, text="Apply Gaussian Noise", command=gaussian_noise, width=40, height=2,bg="Dark Red",fg="white")
button12 = tk.Button(root, text="Apply Uniform Noise", command=uniform_noise, width=40, height=2,bg="Dark Red",fg="white")
button13 = tk.Button(root, text="Apply Forward Fourier Transform", command=ForwardFouriertransform, width=40, height=2,bg="Dark Red",fg="white")
button14 = tk.Button(root, text="Apply Inverse Fourier Transform", command=InverseFouriertransform, width=40, height=2,bg="Dark Red",fg="white")
button16 = tk.Button(root, text="Apply Histogram Equalization", command=Histogram_Equalization, width=40, height=2,bg="Dark Red",fg="white")
button15 = tk.Button(root, text="Apply Histogram Specification", command=Histogram_Specification, width=40, height=2,bg="Dark Red",fg="white")
button17 = tk.Button(root, text="Apply Nearest Neighbor Interpolation", command=NearestNeighborInterpolation, width=40, height=2,bg="Dark Red",fg="white")

root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)
root.grid_rowconfigure(3, weight=1)
root.grid_rowconfigure(4, weight=1)
root.grid_rowconfigure(5, weight=1)
root.grid_rowconfigure(6, weight=1)
root.grid_rowconfigure(7, weight=1)
root.grid_rowconfigure(8, weight=1)
root.grid_rowconfigure(9, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)
root.grid_rowconfigure(3, weight=1)
root.grid_rowconfigure(4, weight=1)
root.grid_rowconfigure(5, weight=1)
root.grid_rowconfigure(6, weight=1)
root.grid_rowconfigure(7, weight=1)
root.grid_rowconfigure(8, weight=1)
root.grid_rowconfigure(9, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

button1.grid(row=0, column=0, columnspan=2, pady=5)

button2.grid(row=1, column=0, padx=(300, 0), pady=(5, 5))
button3.grid(row=2, column=0, padx=(300, 0), pady=5)
button4.grid(row=3, column=0, padx=(300, 0), pady=5)
button5.grid(row=4, column=0, padx=(300, 0), pady=5)
button6.grid(row=5, column=0, padx=(300, 0), pady=5)
button7.grid(row=6, column=0, padx=(300, 0), pady=5)
button8.grid(row=7, column=0, padx=(300, 0), pady=5)
button17.grid(row=8, column=0,padx=(300, 0), pady=(5, 5))

button9.grid(row=1, column=1, padx=(0, 300), pady=(5, 5))
button10.grid(row=2, column=1, padx=(0, 300), pady=5)
button11.grid(row=3, column=1, padx=(0, 300), pady=5)
button12.grid(row=4, column=1, padx=(0, 300), pady=5)
button13.grid(row=5, column=1, padx=(0, 300), pady=5)
button14.grid(row=6, column=1, padx=(0, 300), pady=5)
button15.grid(row=7, column=1, padx=(0, 300), pady=5)
button16.grid(row=8, column=1, padx=(0, 300), pady=(5, 5))




def userinput(message):
    window = tk.Tk()
    window.geometry("{}x{}+{}+{}".format(300, 150, 700, 300))
    prompt_label = tk.Label(window, text=message)
    prompt_label.pack()
    input_box = tk.Entry(window)
    input_box.pack(pady=20)
    submit_button = tk.Button(window, text="Ok", command=window.quit,width=15,height=2)
    submit_button.pack(pady=20)
    window.mainloop()
    user_input = input_box.get()
    window.destroy()
    return user_input
root.mainloop()