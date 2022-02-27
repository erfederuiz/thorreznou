
# resize list of images to same sizes / dimensions
def resize_image(directory_path,
                pixels_1,
                pixels_2,
                bw = False,
                img_name_list= None,
                zip_file = False):
    '''   
    Resize images in .jpg format. Its purpose is to have all the image set levelled off 
    to use it in a convolutional neural net. Returns a list of the resized images.
    
    Keywords arguments:
    directory_path -- directory containing the set of original images
    pixels_1 -- value for the horizontal axis (pixels)
    pixels_2 -- value for the vertical axis (pixels)
    img_name_list -- in case there is a list of images names to use (default: None)
        
    '''


    from cv2 import imread , resize , IMREAD_GRAYSCALE
    from os import remove , walk
    from os.path import splitext
    import zipfile
    from pathlib import PurePath
    
    
    #List to append the resized images
    re_image_list = []
    
    #If images in zip_file, extract and delete original zip
    if zip_file:
        
        with zipfile.ZipFile(directory_path , 'r') as z:
            
            for name in z.namelist():
                
                if (img_name_list != None):
                    z.extract(name, '.\images_unzip')
                else:
                    z.extract(name, '.\images_unzip')
                    
        #Delete zip file
        remove(directory_path)
        
        #Search paths in directories and subdirectories for all the images
        img_path_list = []
        for path , folders , files in walk('.\images_unzip'):
            
            #If not a predefined list of images, take all paths
            if img_name_list == None:
                
                for name in files: 
                    file_ext = splitext(name)[1]
                    
                    if file_ext.lower() == '.jpg':
                        img_path_list.append(PurePath(path, name))
                    else:
                        continue  
                    
            #If a predefined list of images, take only the ones in list
            if img_name_list != None:
                
                for name in files: 
                    
                    if name in img_name_list:
                        file_ext = splitext(name)[1]
                        
                        if file_ext.lower() == '.jpg':
                            img_path_list.append(PurePath(path, name))
                        else:
                            continue

        #Resize images and append to list
        print('Nº images:' ,
              len(img_path_list), '------- Final shape: %sx%s pixels' % (pixels_1 , pixels_2))
        
        for image_path in img_path_list:
            if bw:
                img = imread(str(image_path) , flags=IMREAD_GRAYSCALE)
            else:
                img = imread(str(image_path))
            image = resize(img, (pixels_1, pixels_2))
            re_image_list.append(image)     
        
    #If not zip_file
    if not zip_file:
        
        img_path_list = []
        for path , folders , files in walk('.\images_unzip'):
            
            #If not a predefined list of images, take all paths
            if img_name_list == None:
                
                for name in files: 
                    file_ext = splitext(name)[1]
                    
                    if file_ext.lower() == '.jpg':
                        img_path_list.append(PurePath(path, name))
                    else:
                        continue  
                    
            #If a predefined list of images, take only the ones in list
            if img_name_list != None:
                
                for name in files: 
                    
                    if name in img_name_list:
                        file_ext = splitext(name)[1]
                        
                        if file_ext.lower() == '.jpg':
                            img_path_list.append(PurePath(path, name))
                        else:
                            continue
                        
        #Resize images and append to list            
        print('Nº images:' , 
              len(img_name_list), '------- Final shape: %sx%s pixels' % (pixels_1 , pixels_2))
        
        for image_path in img_path_list:
            if bw:
                img = imread(str(image_path) , flags=IMREAD_GRAYSCALE)
            else:
                img = imread(str(image_path))
            image = resize(img, (pixels_1, pixels_2))
            re_image_list.append(image)
        
    return re_image_list


# standarize list of numbers into a format which makes possible to convert to float / integers
def standarize_numbers(numbers_list, 
                        type):
    '''
    Convert a list of numbers in string format to float format in case of 
    incompatibilities. Returns a list of float numbers (for "decimal" and "double") 
    or a list of integers (for "integer").
    
    Keywords arguments:
    numbers_list -- list of elements to convert
    type -- number type to convert. 3 values:
        - "decimal": decimal number denoted by "," instead of ".". Ej.: 6,67
        - "integer": integer number with thousand, millions, billions, etc. denoted by '.'. Ej.: 12.000.000
        - "double": number written in both previous forms. Ej.: 6.000.000,255 
    '''
    
    #Make transformations by 'type' argument value
    if type == 'decimal':
        float_list = list(map(lambda num: float(num.replace(',' , '.')) , numbers_list))
    elif type == 'integer':
        float_list = list(map(lambda num: int(num.replace('.' , '')) , numbers_list))
    elif type == 'double':
        float_list = list(map(lambda num: float(num.replace('.' , '').replace(',' , '.')) ,
                              numbers_list))
    else:
        print('Type not valid')

    return float_list


# transform str decimal coordenates into float decimal coordenates
def coordinates(list_lat,
                list_lng,
                tuple_format= False):
    '''
    Function to transform mixed coordinates (integers + cardinal point) to decimal 
    coordenates. Be careful, the two firts numbers will be taken as the integer part 
    of the decimal coordinate. Return 2 options: separated lists for latitude and 
    longitude or list of tuples representing geographic points.
    
    Keyword arguments:
    list_lat -- list of latitude values. Ej.: 400782N , 031213s 
    list_lng -- list of longitud values. Ej.: 239872e , 328769W (also accepts Spanish version "O")
    tuple_format -- If False, returns separated lists; if True, returns a list of tuples (default False)
    '''
    
    #List to append new latitude values and count to show position in case of error
    lat = []
    contador = 0
    
    #Loop to take all values in original latitude list
    for coor in list_lat:
        #Check if the original value is correctly written, if not, adds '0' before value to avoid errors
        if len(coor) < 7:
            coor = (7-len(coor))*'0' + coor
            
        #North values
        if (coor[-1] == 'N') or (coor[-1] == 'n'):
            valor = coor[0 : 2] + '.' + coor[2 : -1]
            lat.append(float(valor))
            #South values
        elif (coor[-1] =='S') or (coor[-1] =='s'):
            valor = coor[0 : 2] + '.' + coor[2 : -1]
            lat.append(-float(valor))
        #Anything else
        else:
            print('Latitudinal coordenate (%s) not valid in position (%s)' % (coor , contador))
            break
        
        contador += 1
        
    #List to append new longitude values and count to show position in case of error
    lng = []
    contador = 0
    #Loop to take all values in original longitude list
    for coor in list_lng:
        #Check if the original value is correctly written, if not, adds '0' before value to avoid errors
        if len(coor) < 7:
            coor = (7-len(coor))*'0' + coor
            
        #West values
        if (coor[-1] == 'W') or (coor[-1] == 'w') or (coor[-1] == 'O') or (coor[-1] == 'o'):
            valor = coor[0 : 2] + '.' + coor[2 : -1]
            lng.append(-float(valor))
        #East values
        elif (coor[-1] == 'E') or (coor[-1] == 'e'):
            valor = coor[0 : 2] + '.' + coor[2 : -1]
            lng.append(float(valor))
        else:
            print('Longitudinal coordenate (%s) not valid in position (%s)' % (coor , contador))
            break
        
        contador += 1
    
    #If taple_format, return list of coordenate tuples with zip(). If not, return separated lists
    if tuple_format:
        coor = list(zip(lat , lng))
        
        return coor
    
    else:
        
        return lat , lng


# reduce image size maintaining the original proportions
def reduce_img(image, height):
    '''
    Reduce the image dimensions, maintaining the ratio between original width 
    and height values. Returns the reduced image.
    
    Keyword arguments:
    image -- image to reduce dimensions
    height -- new height
    '''
    
    
    from cv2 import resize , INTER_AREA
    
    
    #Set dimensions ratio
    ratio = image.shape[0]/image.shape[1]
    #Calculate new width
    width = int(height/ratio)
     
    return resize(image, (width, height), interpolation=INTER_AREA)      


# process color image to use in the 'reduce_color_palette' function
def process_color(channel_value, bins):
    '''
    This function is used to map the values of RGB in a pixel that
    go from 0 to 255 to a simple version with X values that lead to
    a palette of Y colors (Xx Xx Xx = Y).
    Eg. X = 4, then y = 4x4x4 = 64 colors
    
    It is used in 'reduce_col_palette' function
    
    Keywords arguments:
    channel_value -- value of color intensity for each pixel in the BGR layers
    bins -- number of colours to reduce each layer (total colors = bins^3)
    '''
    
    
    from numpy import floor
    
    
    ######################################
    if channel_value >= 255: 
        processed_value = 255
    else:
        preprocessed_value = floor((channel_value*bins)/255)
        processed_value = abs(int(preprocessed_value*(255/(bins-1))))
    
    return processed_value


# reduce the number of colours of a image
def reduce_col_palette(image, bins, info=False):
    '''
    This function iterate through every pixel of an image to map
    each rgb channel color value to a reduced palette.
    
    Keywords arguments:
    image -- image to reduce color palette
    bins -- number of colours to reduce each layer (total colors = bins^3). Used in 'process_color' function
    info -- if True, prints a message with the number of color of the new image
    '''


    from numpy import reshape
    
    
    # Capture image dimensions
    img = image.flatten()
       
    # Iterate the array to transform the value of the pixels
    for px in range(len(img)):
        
        if img[px] == 255: img[px] = 255
        else: img[px] = process_color(img[px], bins)
    
    # Restore image shape
    img = reshape(img, image.shape)
        
    # Inform user if 'info' is True about the total number of colors in new image
    if info:
        print(f'Palette reduced to {bins**3} colors.')
            
    return img

