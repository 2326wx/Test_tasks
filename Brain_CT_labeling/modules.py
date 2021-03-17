'''
Contains

- Translation functions from Euler angles to rotation matrix and back. Taken from: https://learnopencv.com/rotation-matrix-to-euler-angles/
- Function for reading transform txt files
- Image preprocessing loader: it loads transform data for image, setup and verify transform, then transform segmentation file.
- Batch generator
- Load and save Pickle utilities
- Dice loss function
- Metrics Class
- Dataset creation function

'''

import math
import numpy as np
import os
import pickle
import SimpleITK as sitk
from pathlib import Path
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K
from sklearn.metrics import multilabel_confusion_matrix




# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6





# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :
    
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
                                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                    
    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R




# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])





# get transform file data for .nii img from txt file

def get_transform_data(fpath):
    '''
    Input: full filename with path, only txt files accepted.
    Output: dict with 'Transform', 'Parameters', 'FixedParameters' data from txt file
    '''
    assert fpath.split('.')[-1]=='txt', 'File is not supported: ' + fpath    
    
    output={}
    with open(fpath) as f:
        contents = f.readlines()
    for line in contents:
        if line.startswith('Transform'):
            output['Transform'] = (line.split(':')[1]).strip()
            assert output['Transform']=='MatrixOffsetTransformBase_double_3_3', 'This transform type is not supported: '+ output['Transform']
        if line.startswith('Parameters'):
            params = np.array((line.split(':')[1]).strip().split(), dtype='float64')
            assert len(params)==12, 'This transform type is not supported: %i parameters.' % len(params)
            output['Parameters'] = params.reshape(4,3)
        if line.startswith('FixedParameters'):
            output['FixedParameters']= np.array((line.split(':')[1]).strip().split(' '), dtype='float64')
            
    assert len(output.keys())==3, 'Can not read the file ' + fpath
    
    return output




def load_and_process_images(full_path, name_prefix, transform_type='X', silent=True, return_3D_arrays=True):
    '''
    Load X and y images, then reversely transform y image according to transformation txt file.
    Returns tuple of SimpleITK.SimpleITK.Image objects as (img_x, img_y)
    - full_path - path to files
    - name_prefix - common prefix for filnames
    - transform_type - 'X', 'y' or None: transforms X or y image respectively
    - silent - if True, no messages printed
    - return_3D_arrays - if False, then returns SimpleITK.SimpleITK.Image objects, else returns data 3D image arrays
    '''
    name_prefix = str(name_prefix)
    
    if not transform_type: transform_type='not'
    
    if not silent: print('Load images "%s.*" from %s...' % (name_prefix, full_path) )

    transform_data = get_transform_data(str(Path(full_path, name_prefix+'.txt')))['Parameters']
    rotation_matrix = np.ravel(transform_data[:3,:])
    offsets = np.ravel(transform_data[3:,:])
    
    if not silent:
        print('Rotation matrix:\n', transform_data[:3,:])
        print('Offset:\n', transform_data[3:,:])
        
    assert isRotationMatrix(rotation_matrix.reshape(3,3)), name_prefix+'.txt : rotation matrix is false.'

    point = [0,0,0]
    forward_transform = sitk.Euler3DTransform()
    forward_transform.SetMatrix(rotation_matrix)
    forward_transform.SetTranslation(offsets)
    reverse_transform = forward_transform.GetInverse()
    forward_transform.TransformPoint(point)
    assert np.sum(np.round(reverse_transform.TransformPoint(forward_transform.TransformPoint(point)),6))==0, 'Error in transform data'
    if not silent: print ('Transform Ok')
        
    img_x = sitk.ReadImage(str(Path(full_path, name_prefix+'.nii')))
    img_y = sitk.ReadImage(str(Path(full_path, name_prefix+'-seg.nii')))
           
    assert isinstance(img_x, sitk.SimpleITK.Image), name_prefix+'.nii is broken.'
    assert isinstance(img_y, sitk.SimpleITK.Image), name_prefix+'-seg.nii is broken.'
    assert img_x.GetSize()==img_y.GetSize(), 'Images X and Y are of different sizes.'
            
    if transform_type.lower()=='y': img_y = sitk.Resample(img_y, reverse_transform, sitk.sitkNearestNeighbor)
    if transform_type.lower()=='x': img_x = sitk.Resample(img_x, forward_transform, sitk.sitkNearestNeighbor)
    
    if not return_3D_arrays: return img_x, img_y
    
    img_x_data = sitk.GetArrayFromImage(img_x)
    img_y_data = sitk.GetArrayFromImage(img_y)
    img_x_data = np.moveaxis(img_x_data, 0, -1)
    img_y_data = np.moveaxis(img_y_data, 0, -1)
                           
    return np.array(img_x_data), np.array(img_y_data)





# generate batches
class BatchGenerator(Sequence):
    
    def __init__(self, X_input, y_input, list_IDs, batch_size=1, shuffle=True):
        
        self.Xdim = X_input.shape[1:]
        self.ydim = y_input.shape[1:]
        self.batch_size = batch_size        
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()
        self.X_input = X_input
        self.y_input = y_input
        self.dtype_x = X_input.dtype
        self.dtype_y = y_input.dtype

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
       
        X = np.empty((self.batch_size, *self.Xdim), dtype = self.dtype_x)
        y = np.empty((self.batch_size, *self.ydim), dtype = self.dtype_y)
        
        for i, ID in enumerate(list_IDs_temp):            
            X[i,] = self.X_input[ID,]
            y[i,] = self.y_input[ID,]

        return X, y
    
    
    
def save_pkl(data, filepath):
    '''
    Saves any object as pickle file. Filename shall contain full path.
    '''
    with open(filepath, 'wb') as output:
        pickle.dump(data, output)       
        
def load_pkl(filepath):
    '''
    Loads object from pickle file. Filename shall contain full path.
    '''
    with open(filepath, 'rb') as input:
        return pickle.load(input)
    
    
    
    
def tversky_loss(y_true, y_pred):
    #https://github.com/keras-team/keras/issues/9395#issuecomment-379228094
    
    y_true = K.cast(y_true,'float32')
    
    alpha = 0.90
    beta  = 0.10
    
    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true
    
    num = K.sum(p0*g0, (0,1,2))
    den = num + alpha*K.sum(p0*g1,(0,1,2)) + beta*K.sum(p1*g0,(0,1,2))
    
    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T




class Metrics:
    '''
    Class instanse calculates IoU, IoU by class, FP, FN, TP, TN on init.
    Works only with int arrays with int class labels (1,2,...,n)!
    '''


    def __init__(self, true, pred, tol=1e-16):


        targets = (np.array(true).astype('int16')).flatten()
        inputs = (np.array(pred).astype('int16')).flatten()
        
        _, counts = np.unique(targets, return_counts=True)
        self.weights = (1-(counts/np.sum(counts)))/np.sum(1-(counts/np.sum(counts)))

        assert targets.shape==inputs.shape, 'Arrays are of different sizes, can not create instance.'

        tmax = max( targets.max(), inputs.max()) 

        tb = np.zeros((targets.size,tmax+1),dtype='bool')
        tb[np.arange(targets.size),targets] = 1

        ib = np.zeros((inputs.size, tmax+1),dtype='bool')
        ib[np.arange(inputs.size),inputs] = 1

        intersection = np.logical_and(tb,ib)
        union = np.logical_or(tb,ib)
        
        self.classIoU = ((np.sum(intersection,axis=0)+tol) / (np.sum(union,axis=0)+tol))[1:]
        self.IoU = (np.sum(np.logical_and(tb[:,1:],ib[:,1:]))+tol) / (np.sum(np.logical_or(tb[:,1:],ib[:,1:]))+tol)
        
        self.cm = multilabel_confusion_matrix(tb, ib)
        self.weighted_cm = self.cm*self.weights.reshape((self.weights.shape[0],)+(1,1,))
        self.misclass_rate = (self.cm[:,1,0]+self.cm[:,0,1])/(self.cm[:,0,0]+self.cm[:,1,1]+self.cm[:,1,0]+self.cm[:,0,1])
         
        self.TN = self.cm[:,0,0]
        self.FN = self.cm[:,1,0]

        self.TP = self.cm[:,1,1]
        self.FP = self.cm[:,0,1]
        
        assert np.mean(self.TP+self.TN+self.FN+self.FP)==len(targets), "Metrics failed!!!"
        
        
        
def create_dataset(plane, trainval_files, dataset_folder, img_size=(512,512,116), z_dim=128, xydim=512, n_input_channels=1, n_output_channels=20):
    # Create X and y arrays. All trainval data fits memory. 
    files = len(trainval_files)

    if plane not in('XY', 'XZ', 'YZ'): raise Exception('Plane is missed')

    if plane!='XY':
        X = np.zeros((files*img_size[0], img_size[0], z_dim, n_input_channels), dtype='float32')
        y = np.zeros((files*img_size[0], img_size[0], z_dim, n_output_channels), dtype='bool')
    else:
        X = np.zeros((files*z_dim, img_size[0], img_size[1], n_input_channels), dtype='float32')
        y = np.zeros((files*z_dim, img_size[0], img_size[1], n_output_channels), dtype='bool')

    crops_placed = 0

    for i in trainval_files:

        print('Processing files group for %i.nii..' % i)

        img_x, img_y = load_and_process_images(os.path.join(dataset_folder), i, 'X')

        img_x=img_x/np.max(img_x)
        
        if img_x.shape[2]!=z_dim:          # if image Z shape is less than required:  padding Z coordinate with reflection
            delta = z_dim-img_x.shape[2]
            pad_left = int(delta/2)
            pad_right = delta-pad_left        
            img_x = np.pad(img_x, ((0,0),(0,0),(pad_left, pad_right)), mode='symmetric')
            img_y = np.pad(img_y, ((0,0),(0,0),(pad_left, pad_right)), mode='symmetric')

        if plane=='YZ':
            for crop in range(xydim):
                X[crops_placed, :, :, :]=np.expand_dims(img_x[crop,:,:], 2)
                for category in range(0, n_output_channels):
                    y[crops_placed, :, :, category] = (img_y==category+1)[crop,:,:]
                crops_placed+=1

        if plane=='XZ':
            for crop in range(xydim):
                X[crops_placed, :, :, :]=np.expand_dims(img_x[:,crop,:], 2)
                for category in range(0, n_output_channels):
                    y[crops_placed, :, :, category] = (img_y==category+1)[:,crop,:]
                crops_placed+=1

        if plane=='XY':
            for crop in range(z_dim):
                X[crops_placed, :, :, :]=np.expand_dims(img_x[:,:,crop], 2)
                for category in range(0, n_output_channels):
                    y[crops_placed, :, :, category] = (img_y==category+1)[:,:,crop]
                crops_placed+=1

    print('Saving dataset...')
    save_pkl(X, os.path.join(dataset_folder, plane+'_X.pkl'))
    save_pkl(y[:(y.shape[0]//2),...], os.path.join(dataset_folder, plane+'_y1.pkl'))
    save_pkl(y[(y.shape[0]//2):,...], os.path.join(dataset_folder, plane+'_y2.pkl'))

    print('Saved:\nX: %s of %s, y: %s of %s.' % (X.shape, X.dtype, y.shape, y.dtype))