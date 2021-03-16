import json 
import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.measure import find_contours

'''
Задача.
Имеются фото и соответствующие им json файлы, в которых указаны координаты прямоугольников найденных на фото объектов. Необходимо объекты собрать в группы по принадлежности к полке, как на примере. 
Координаты: первые два числа - это XY левого верхнего угла, вторые – XY правого нижнего угла.
Скрипт требуется написать на python, можно использовать сторонние модули (OpenCV, sklearn и т.д.).

Алгоритм решения.
1. По очереди открываем пары файлов: разметку+изображение.
2. Разметку укрупняем: собираем в один близкие и перекрывающиеся объекты (модуль shelf_from_img()), помещаем координаты углов в датафрейм df.
3. В df добавляем новые признаки объектов разметки: координаты нижних и верхних границ, координаты центра, ширину, высоту, площадь, её логарифм, пропорции(=aspect). Что-то из этого в итоге не понадобилось.
4. Проводим несколько кластеризаций данных с разными парамтрами: сначала получаем черновую разметку кластеров по координате Y, затем по координате X, затем повторно по Y с уже имеющейся разметкой по Х.
5. Результаты помещаем в массив res_clusters, размером, идентичным фотографии, в виде номеров кластеров (=номеров полок).
6. Наложив на этот массив имеющиеся в файлах разметки исходные bboxы, получаем номер полки для каждого объекта.
7. Наносим полученные данные на изображение, указываем на нем номер полки и сохраняем в папку results.

Для работы достаточно поместить данный скрипт в папку, содержащую файлы изображения и разметки, и запустить его.
'''

def shelf_from_img(df, pad_x=50, pad_y=50):
    ''' 
    Объединяет близко расположенные или накладывающиеся bboxы в один с параметрами pad_x, pad_y
    Возвращает датафрейм с новыми координатами bboxов и маску с ними размером, эквивалентным размеру снимка.     
    '''
    mask_x = np.zeros((img.shape[0], img.shape[1])).astype('bool')
    padding_x = pad_x        
    for i in range(len(df)):
        xmin,ymin,xmax,ymax = df.loc[i].values
        xmin = max(xmin-padding_x, 0)            
        xmax = min(xmax+padding_x, img.shape[1])            
        mask_x[ymin:ymax,xmin:xmax] = 1
    
    cx = find_contours(mask_x)        
    mask_x = np.zeros_like(mask_x)
    for i in range(len(cx)):
        xmax,ymax = np.max(cx[i], axis=0).astype('int')
        xmin,ymin = np.min(cx[i], axis=0).astype('int')
        mask_x[xmin:xmax,ymin:ymax]=1
        
    
    mask_y= np.zeros((img.shape[0], img.shape[1])).astype('bool')        
    padding_y = pad_y
    for i in range(len(df)):
        xmin,ymin,xmax,ymax = df.loc[i].values            
        ymin = max(ymin-padding_y, 0)                
        ymax = min(ymax+padding_y, img.shape[0])        
        mask_y[ymin:ymax,xmin:xmax] = 1        
    
    cy = find_contours(mask_y)        
    mask_y = np.zeros_like(mask_y)
    for i in range(len(cy)):
        xmax,ymax = np.max(cy[i], axis=0).astype('int')
        xmin,ymin = np.min(cy[i], axis=0).astype('int')
        mask_y[xmin:xmax,ymin:ymax]=1
        
    mask_x = mask_x.astype('bool')
    mask_y = mask_y.astype('bool')
    
    mask = mask_x+mask_y          
    
    c = find_contours(mask)          
    res_df = pd.DataFrame(index=range(len(c)), columns=df.columns)
    mask = np.zeros_like(mask)
    for i in range(len(c)):
        ymax,xmax = np.max(c[i], axis=0).astype('int')
        ymin,xmin = np.min(c[i], axis=0).astype('int')
        mask[ymin:ymax,xmin:xmax]=1
        
        res_df.xmin[i] = xmin
        res_df.xmax[i] = xmax
        res_df.ymax[i] = ymax
        res_df.ymin[i] = ymin   
        
    return mask, res_df

def clusterize(df, quantile=0.15, bin_seeding=True):
    ''' Кластеризует данные во входном датафрейме, возвращает столбец с присвоенными номерами кластеров
    '''
    res = df.values
    bandwidth = estimate_bandwidth(res, quantile=quantile)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding)
    ms.fit(res) 
    return ms.labels_

def get_colors(n_colors):
    '''Генерирует случайную цветовую палитру, возвращает первые n_colors цветов
    '''
    palette = np.arange(0, 2048, dtype=np.uint8).reshape(1, 2048, 1)
    palette = cv2.applyColorMap(palette, cv2.COLORMAP_JET).squeeze(0)
    np.random.shuffle(palette)
    return palette[:n_colors].tolist()

# create output folder
if not os.path.exists(Path(os.getcwd(),'results')): os.mkdir('results')

for fname in os.listdir(): 
    if '.json' in fname: # for each file

        # read image, put bboxes to df
        f = open(fname) 
        data = json.load(f)
        f.close() 
        img = cv2.imread(fname.replace('.json',''))
        df = pd.DataFrame(data = data['boxes'], index=range(len(data['boxes'])), columns=['xmin', 'ymin', 'xmax', 'ymax']).astype('int')        
        df['cat']=0
        bboxes = df.values
        
        # join close bboxes
        mask, df = shelf_from_img(df.iloc[:,:4], pad_x=5, pad_y=15)            
            
        # add new features    
        df['centers_x'] = (df.xmax+df.xmin)/2
        df['centers_y'] = (df.ymax+df.ymin)/2
        df['bottoms_x'] = (df.xmax+df.xmin)/2
        df['bottoms_y'] = df.ymax 
        df['tops_x'] = (df.xmax+df.xmin)/2
        df['tops_y'] = df.ymin 
        df['area'] = (df.xmax-df.xmin)*(df.ymax-df.ymin)
        df['aspect'] = (df.xmax-df.xmin)/(df.ymax-df.ymin)
        df['height'] = (df.ymax-df.ymin)
        df['log_area'] = df.area.apply(lambda x: np.log(x))
        df['width'] = df.xmax-df.xmin

        # clustering and generate y_cat, x_cat features
        df['y_cat'] = clusterize(df[['bottoms_y', 'tops_y', 'centers_y', 'height','aspect','bottoms_x', 'width', 'log_area']], 0.15)        
        df['x_cat'] = clusterize(df[['y_cat', 'bottoms_y', 'centers_y', 'bottoms_x']], 0.357)
        df['y_cat'] = clusterize(df[['x_cat', 'bottoms_y']], 0.15)
                
        # create final object shelf labels
        df['cat'] = df.y_cat.apply(str)+df.x_cat.apply(str)        
        df.cat = df.cat.map(dict(zip(df.cat.unique().tolist(), range(1, 1+len(df.cat.unique())))))
        
        # create resulting array with objects labels
        res_clusters = np.zeros_like(mask).astype('int')        
        for i in df.index:
            res_clusters[df.ymin[i]:df.ymax[i],df.xmin[i]:df.xmax[i]] = df.cat[i]

        # get coloring dictionary
        colors = dict(zip(range(1,1+len(df.cat.unique())), get_colors(len(df.cat.unique()))))
        
        # set label font params
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1.5
        color = (0, 255, 0) 
        thickness = 5
        
        for i in range(bboxes.shape[0]): # for each initial bbox:
            # get corners
            xmin,ymin,xmax,ymax = bboxes[i,:4]
            # get resulting label
            cat = res_clusters[int((ymax+ymin)/2), int((xmax+xmin)/2)]
            # crop bbox from image
            crop = img[ymin+1:ymax-1, xmin+1:xmax-1]
            # generate rectangle of category color
            rect = np.ones(crop.shape, dtype=np.uint8)
            rect[:,:,:] = colors[cat]
            # mixup image crop with rectangle
            res = cv2.addWeighted(crop, 0.5, rect, 0.5, 1.0)    
            # place mix to image
            img[ymin+1:ymax-1, xmin+1:xmax-1] = res    
            # find bbox center
            org = (int((xmax+xmin)/2), int((ymax+ymin)/2))
            # put label text to image over rectangle 
            img = cv2.putText(img, str(cat), org, font, fontScale, color, thickness, cv2.LINE_AA)
        # save image
        cv2.imwrite(str(Path('results','shelf_clustered_' + fname.replace('.json',''))), img)
        print('Saved:','shelf_clustered_' + fname.replace('.json',''))

print('Completed.')