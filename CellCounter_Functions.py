import pylab 
import os
import sys
import fnmatch
import imread
import cv2
import numpy as np
import mahotas as mh 
import pandas as pd
import holoviews as hv
from contextlib import contextmanager
hv.notebook_extension('bokeh')



#############################################################################################################################


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


#############################################################################################################################


def getdirinfo(dirinfo):

    #Define subdirectories
    dirinfo['ch1'] = os.path.join(os.path.normpath(dirinfo['main']), "Ch1")
    dirinfo['ch2'] = os.path.join(os.path.normpath(dirinfo['main']), "Ch2")
    dirinfo['ROI'] = os.path.join(os.path.normpath(dirinfo['main']), "ROI")
    dirinfo['output'] = os.path.join(os.path.normpath(dirinfo['main']), "SavedOutput")

    #Get filenames and create output subdirectories based upon usage
    try:
        os.mkdir(dirinfo['output'])
    except FileExistsError:
        pass
    if os.path.isdir(dirinfo['ch1']):
        dirinfo['ch1_fnames'] = sorted(os.listdir(dirinfo['ch1']))
        dirinfo['ch1_fnames'] = fnmatch.filter(dirinfo['ch1_fnames'], '*.tif') #restrict files to .tif images
        dirinfo['output_ch1'] = os.path.join(os.path.normpath(dirinfo['output']), "Ch1")
        try:
            os.mkdir( dirinfo['output_ch1'])
        except FileExistsError: 
            pass
    if os.path.isdir(dirinfo['ch2']):
        dirinfo['ch2_fnames'] = sorted(os.listdir(dirinfo['ch2']))
        dirinfo['ch2_fnames'] = fnmatch.filter(dirinfo['ch2_fnames'], '*.tif') #restrict files to .tif images
        dirinfo['output_ch2'] = os.path.join(os.path.normpath(dirinfo['output']), "Ch2")
        try:
            os.mkdir( dirinfo['output_ch2'])
        except FileExistsError: 
            pass  
    if os.path.isdir(dirinfo['ROI']):
        dirinfo['roi_fnames'] = sorted(os.listdir(dirinfo['ROI']))
        dirinfo['roi_fnames'] = fnmatch.filter(dirinfo['roi_fnames'], '*.tif') #restrict files to .tif images
    if os.path.isdir(dirinfo['ch1']) and os.path.isdir(dirinfo['ch2']):
        dirinfo['output_merge'] = os.path.join(os.path.normpath((dirinfo['output'])), "Merge")
        try:
            os.mkdir(dirinfo['output_merge']) 
        except FileExistsError:
            pass
    return dirinfo



#############################################################################################################################


def optim_getdirinfo(dirinfo):
    
    #Define existing subdirectories
    dirinfo['composite'] = os.path.join(os.path.normpath(dirinfo['main']), "Composite")
    dirinfo['manual'] = os.path.join(os.path.normpath(dirinfo['main']), "ManualCounts")
    dirinfo['output'] = os.path.join(os.path.normpath(dirinfo['main']), "SavedOutput")
    
    #Get filenames and create output subdirectories based upon usage
    try:
        os.mkdir(dirinfo['output'])
    except FileExistsError:
        pass

    dirinfo['composite_fnames'] = sorted(os.listdir(dirinfo['composite']))
    dirinfo['composite_fnames'] = fnmatch.filter(dirinfo['composite_fnames'], '*.tif') #restrict files in folder to .tif files
    dirinfo['manual_fnames'] = sorted(os.listdir(dirinfo['manual']))
    dirinfo['manual_fnames'] = fnmatch.filter(dirinfo['manual_fnames'], '*.tif') #restrict files in folder to .tif files

    return dirinfo


#############################################################################################################################


def optim_getimages(dirinfo,params):
    images = {
        'manual' : mh.imread(os.path.join(os.path.normpath(dirinfo['manual']), 
                                         dirinfo['manual_fnames'][0]), as_grey=True),
        'composite' : mh.imread(os.path.join(os.path.normpath(dirinfo['composite']), 
                                     dirinfo['composite_fnames'][0]), as_grey=True)
    }
    images['bg'] = cv2.GaussianBlur(images['composite'].astype('float'),(0,0),params['diam']*3)
    images['bg'] = images['composite'] - images['bg'] 
    images['bg'][images['bg']<0] = 0
    images['gauss'] = cv2.GaussianBlur(images['bg'],(0,0),params['diam']/12)
    params['counts'] = (images['manual']>0).sum()
    params['otsu'] = mh.otsu(images['gauss'].astype('uint8'))
    params['thresh'] = params['otsu']
    count_out = Count(0,"Optim",params,dirinfo,UseROI=False,UseWatershed=params['UseWatershed'])  
    images['otsu'] = count_out['thresh']
     
    i_comp = hv.Image((np.arange(images['composite'].shape[1]), 
                       np.arange(images['composite'].shape[0]), 
                       images['composite'])).opts(
               invert_yaxis=True,cmap='gray',toolbar='below',
               title="Composite Image") 
    i_gauss = hv.Image((np.arange(images['gauss'].shape[1]), 
                        np.arange(images['gauss'].shape[0]), 
                       images['gauss'])).opts(
               invert_yaxis=True,cmap='gray',toolbar='below',
               title="Preprocessed Image")   
    i_otsu = hv.Image((np.arange(images['otsu'].shape[1]), 
                       np.arange(images['otsu'].shape[0]), 
                       images['otsu']*255)).opts(
                invert_yaxis=True,cmap='gray',toolbar='below',
                title="Otsu Thresholded Image")
    i_cells = hv.Image((np.arange(count_out['cells'].shape[1]), 
                       np.arange(count_out['cells'].shape[0]), 
                       count_out['cells']*(255//count_out['cells'].max()))).opts(
                invert_yaxis=True,cmap='jet',toolbar='below',
                title="Cells Counted Using Otsu")   
    
    display = i_comp + i_gauss + i_otsu + i_cells

    return images, params, display


#############################################################################################################################


def optim_iterate(images,dirinfo,params):

    Channel = 'Optim'
    file = 0 #Should always be zero because only one composite image

    #Initialize Arrays to Store Data In
    List_AutoCounts = []
    List_Hits = []
    List_CellAreas = []
    List_Acc_HitsOverAutoCounts = []
    List_Acc_HitsOverManualCounts = []

    #Define maximum threshold value and create series of thresholds to cycle through
    TMin = 0#params['otsu']
    TMax = int(images['gauss'].max()//1) #Get maximum value in array.  Threshold can't go beyond this
    List_ThreshValues = list(range(TMin,TMax))

    for thresh in List_ThreshValues:

        #print("Counting Threshold " + str(thresh))
        params['thresh']=thresh  
        with suppress_stdout():
            count_out = Count(file,Channel,params,dirinfo,UseROI=False,UseWatershed=params['UseWatershed'])    
        List_AutoCounts.append(count_out['nr_nuclei'])

        #Determine Avg Cell Size in Pixel Units
        if count_out['nr_nuclei'] > 0:
            Cell_Area = count_out['cells'] > 0
            Cell_Area = Cell_Area.sum() / count_out['nr_nuclei']
        elif count_out['nr_nuclei'] == 0:
            Cell_Area = float('nan')
        List_CellAreas.append(Cell_Area)    

        #Determine Number of Automatically Counted Cells Whose Location Corresponds to a Single Manually Counted Cell
        Hits = 0 
        if count_out['nr_nuclei'] > 0:
            for cell in range (1,count_out['nr_nuclei']+1): #for each automatically counted cell
                HitMap = images['manual'][count_out['cells']==cell]
                if HitMap.sum()==255:
                    Hits += 1
        List_Hits.append(Hits)

        #Calculate Accuracies
        try:
            List_Acc_HitsOverAutoCounts.append(Hits/count_out['nr_nuclei'])
        except: #if divide by zero occurs, set value to nan
            List_Acc_HitsOverAutoCounts.append(np.nan)

        List_Acc_HitsOverManualCounts.append(Hits/params['counts'])

    #Create Dataframe
    DataFrame = pd.DataFrame(
        {'AutoCount_Thresh': List_ThreshValues,
         'OTSU_Thresh': np.ones(len(List_ThreshValues))*params['otsu'],
         'Manual_CellDiam': np.ones(len(List_ThreshValues))*params['diam'],
         'Manual_Counts': np.ones(len(List_ThreshValues))*params['counts'],
         'AutoCount_UseWatershed': np.ones(len(List_ThreshValues))*params['UseWatershed'],
         'AutoCount_Counts': List_AutoCounts,
         'AutoCount_Hits': List_Hits,
         'AutoCount_AvgCellArea': List_CellAreas,
         'Acc_HitsOverManualCounts': List_Acc_HitsOverManualCounts,
         'Acc_HitsOverAutoCounts': List_Acc_HitsOverAutoCounts,
         'Acc_HitsOverManualCounts': List_Acc_HitsOverManualCounts,
         'Acc_Avg' : [(List_Acc_HitsOverAutoCounts[x]+List_Acc_HitsOverManualCounts[x])/2 
                      for x in range(len(List_Acc_HitsOverAutoCounts))]
        }) 
    
    return DataFrame

#############################################################################################################################


#Function to Count Cells
def Count(file,Channel,params,dirinfo,UseROI=False,UseWatershed=False):
       
    #Set function parameters in accordance with channel to be counted
    if Channel == "Ch1":
        CellDiam = params['ch1_diam']
        Thresh = params['ch1_thresh']
        Directory_Current = dirinfo['ch1']
        FileNames_Current = dirinfo['ch1_fnames']
    elif Channel == "Ch2":
        CellDiam = params['ch2_diam']
        Thresh = params['ch2_thresh']
        Directory_Current = dirinfo['ch2']
        FileNames_Current = dirinfo['ch2_fnames']
    elif Channel == "Optim":
        CellDiam = params['diam']
        Thresh = params['thresh']
        Directory_Current = dirinfo['composite']
        FileNames_Current = dirinfo['composite_fnames']
 
    #Load file
    Image_Current_File = os.path.join(os.path.normpath(Directory_Current), FileNames_Current[file]) #Set directory location
    Image_Current_Gray = mh.imread(Image_Current_File,as_grey=True) #Load File as greyscale image
    print("Processing: " + FileNames_Current[file])

    #Substract Background
    Image_Current_BG = cv2.GaussianBlur(Image_Current_Gray.astype('float'),(0,0),CellDiam*3)
    Image_Current_BG = Image_Current_Gray - Image_Current_BG #Subtract background from orginal image
    Image_Current_BG[Image_Current_BG<0] = 0 #Set negative values = 0
    #Apply Gaussian Filter to Image
    Image_Current_Gaussian = cv2.GaussianBlur(Image_Current_BG.astype('float'),(0,0),CellDiam/6)
    #Threshold image 
    Image_Current_T = Image_Current_Gaussian > Thresh #Threshold image
    #Erase any particles that are below the minimum particle size
    labeled,nr_objects = mh.label(Image_Current_T) #label particles in Image_Current_T
    sizes = mh.labeled.labeled_size(labeled) #get list of particle sizes in Image_Current_T
    too_small = np.where(sizes < (CellDiam*CellDiam*params['particle_min'])) #get list of particle sizes that are too small
    labeled = mh.labeled.remove_regions(labeled, too_small) #remove particle sizes that are too small 
    Image_Current_T = labeled != 0 #reconstitute Image_Current_T with particles removed

    #Get ROI and apply to thresholded image
    if UseROI:
        ROI_Current_File = os.path.join(os.path.normpath(dirinfo['ROI']), dirinfo['roi_fnames'][file]) #Set directory 
        ROI_Current = mh.imread(ROI_Current_File,as_grey=True) #Load File
        Image_Current_T[ROI_Current==0]=0 #Set values of thresholded image outside of ROI to 0
        roi_size = np.count_nonzero(ROI_Current)
    else:
        roi_size = Image_Current_Gray.size

    if UseWatershed == True:
        Image_Current_Cells, nr_nuclei = watershed(Image_Current_T,CellDiam)
    else:
        Image_Current_Cells, nr_nuclei = mh.label(Image_Current_T)
    
    count_output = {
        'cells' : Image_Current_Cells,
        'nr_nuclei' : nr_nuclei,
        'roi_size' : roi_size,
        'image' : Image_Current_Gray,
        'gauss' : Image_Current_Gaussian,
        'thresh' : Image_Current_T
    }  
    print('Cells: {x}'.format(x=count_output['nr_nuclei']))
    return count_output

#############################################################################################################################    

def Count_folder(dirinfo,params,Channel,UseROI=False,UseWatershed=False):

    #Set some info in accordance with channel to be counted
    if Channel == "Ch1":
        fnames = dirinfo['ch1_fnames']
        output = dirinfo['output_ch1']
        diam = params['ch1_diam']
        thresh = params['ch1_thresh']    
    elif Channel == "Ch2":
        fnames = dirinfo['ch2_fnames']
        output = dirinfo['output_ch2']
        diam = params['ch2_diam']
        thresh = params['ch2_thresh']  
    
    #Initialize arrays to store data in
    COUNTS = []
    ROI_SIZE = []

    #Loop through images and count cells
    for file in range (len(fnames)):
        #call function to count cells
        count_out = Count(file,Channel,params,dirinfo,UseROI=UseROI,UseWatershed=UseWatershed)
        #store summary data
        COUNTS.append(count_out['nr_nuclei'])
        ROI_SIZE.append(count_out['roi_size'])
        #save image of cell locations
        mh.imsave(
            filename = os.path.splitext(
                os.path.join(os.path.normpath(output),fnames[file])
            )[0] + '_Counts.tif',
            array = count_out['cells'].astype(np.uint8)
        )

    #Create DattaFrame
    DataFrame = pd.DataFrame(
    {'FileNames': fnames,
     'Channel' : len(fnames)*[Channel],
     'Thresh' : np.ones(len(fnames))*thresh,
     'UseROI' : np.ones(len(fnames))*UseROI,
     'AvgCellDiam' : np.ones(len(fnames))*diam,
     'ParticleMin' : np.ones(len(fnames))*params['particle_min'],
     'Ch1_Counts': COUNTS,
     'Ch1_ROI_Size': ROI_SIZE
    })
    return DataFrame


#############################################################################################################################    
    
    
def watershed(Image_Current_T,CellDiam):
    
    #If there are pixels above the threshold proceed to watershed
    if Image_Current_T.max() == True:

        #Create distance transform from thresholded image to help identify cell seeds
        Image_Current_Tdist = mh.distance(Image_Current_T)
        Image_Current_Tdist[Image_Current_Tdist<CellDiam*0.3]=0

        #Define Sure Background for watershed
        #Background is dilated proportional to cell diam.  Allows final cell sizes to be a bit larger at end.  
        #Will not affect cell number but can influence overlap
        #See https://docs.opencv.org/3.4/d3/db4/tutorial_py_watershed.html for tutorial that helps explain this
        Dilate_Iterations = int(CellDiam//2) 
        Dilate_bc = np.ones((3,3)) #Use square structuring element instead of cross
        Image_Current_SureBackground = Image_Current_T
        for j in range (Dilate_Iterations): 
            Image_Current_SureBackground = mh.dilate(Image_Current_SureBackground,Bc=Dilate_bc)

        #Create seeds/foreground for watershed
        #See https://docs.opencv.org/3.4/d3/db4/tutorial_py_watershed.html for tutorial that helps explain this
        Regmax_bc = np.ones((CellDiam,CellDiam)) #Define structure element regional maximum function.  Currently uses diamater
        Image_Current_Seeds = mh.locmax(Image_Current_Tdist,Bc=Regmax_bc) #Find local maxima of distance transform
        Image_Current_Seeds[Image_Current_Tdist==0]=False #remove locmax corresponding to 0 values
        Image_Current_Seeds = mh.dilate(Image_Current_Seeds,np.ones((3,3)))
        #Image_Current_Seeds = mh.erode(Image_Current_Seeds,np.ones((3,3)))
        seeds,nr_nuclei = mh.label(Image_Current_Seeds,Bc=np.ones((3,3)))

        #Define unknown region between sure foreground (the seeds) and sure background 
        Image_Current_Unknown = Image_Current_SureBackground.astype(int) - Image_Current_Seeds.astype(int)

        #Modify seeds to differentiate between background and unknown regions
        seeds+=1
        seeds[Image_Current_Unknown==1]=0

        #Perform watershed
        Image_Current_Watershed = mh.cwatershed(surface=Image_Current_SureBackground,markers=seeds)
        Image_Current_Watershed -= 1 #Done so that background value is equal to 0.
        Image_Current_Cells = Image_Current_Watershed

    #If there are no pixels above the threshold watershed procedure has issues.  Set cell count to 0.
    elif Image_Current_T.max() == False:
        Image_Current_Cells = Image_Current_T.astype(int)
        nr_nuclei = 0
    
    #return Image_Current_Seeds, nr_nuclei
    return Image_Current_Cells,nr_nuclei

#############################################################################################################################


#Function to count merged cells
def Merge(Cells_Ch1,Cells_Ch2,params): 
    
    if params['ch1_diam'] < params['ch2_diam']:
        SmallCellDiam = params['ch1_diam']
    else:
        SmallCellDiam = params['ch2_diam']
    size_req = SmallCellDiam*SmallCellDiam*params['overlap']
    merge = np.zeros(Cells_Ch1.shape)
    nr_nuclei = 0
    
    for c1_cell in range(1,Cells_Ch1.max()+1):
        for c2_cell in range(1,Cells_Ch2.max()+1):
            overlap_area=sum(Cells_Ch2[Cells_Ch1==c1_cell]==c2_cell)
            if overlap_area > size_req:
                nr_nuclei+=1
                merge[Cells_Ch1==c1_cell]=nr_nuclei
    
    merge_output = {
        'cells' : merge,
        'nr_nuclei' : nr_nuclei,
    }  
    
    return merge_output
    

#############################################################################################################################


def Merge_folder(dirinfo,params):
    
    if len(os.listdir(dirinfo['output_ch1']))>0 and len(os.listdir(dirinfo['output_ch2']))>0:

        #Get list of Files to operate on 
        dirinfo['output_ch1_fnames'] = sorted(os.listdir(dirinfo['output_ch1']))
        dirinfo['output_ch1_fnames'] = fnmatch.filter(dirinfo['output_ch1_fnames'], '*.tif')
        dirinfo['output_ch2_fnames'] = sorted(os.listdir(dirinfo['output_ch1']))
        dirinfo['output_ch2_fnames'] = fnmatch.filter(dirinfo['output_ch2_fnames'], '*.tif')

        #Define smaller of two cells
        if params['ch1_diam'] < params['ch2_diam']:
            SmallCellDiam = params['ch1_diam']
        else:
            SmallCellDiam = params['ch2_diam']

        #Check to make sure number of files for Ch1 and Ch2 match
        if len(dirinfo['output_ch1_fnames']) != len(dirinfo['output_ch2_fnames']):
            print('Different number of images detected for Ch1 and Ch2.  Aborting Count.')

        else:      
            #Initialize arrays to store data in
            COUNTS = []
            #Loop through files to identify overlapping cells
            for file in range (len(dirinfo['output_ch1_fnames'])):
                    
                #Load Threshold Images of Cell Locations  
                print('Processing: {x}'.format(x=dirinfo['ch1_fnames'][file])) 
                Cells_Ch1 = mh.imread(os.path.join(os.path.normpath(dirinfo['output_ch1']), 
                                                   dirinfo['output_ch1_fnames'][file]), as_grey=True)
                Cells_Ch2 = mh.imread(os.path.join(os.path.normpath(dirinfo['output_ch2']), 
                                                   dirinfo['output_ch2_fnames'][file]), as_grey=True)
                merge_out = Merge(Cells_Ch1,Cells_Ch2,params)
                COUNTS.append(merge_out['nr_nuclei'])    
                mh.imsave(
                    filename = os.path.join(os.path.normpath(dirinfo['output_merge']), 
                                    dirinfo['output_ch1_fnames'][file] + "_merge.tif"),
                    array = merge_out['cells'].astype(np.uint8)
                )

            #Count summary that can be saved to disk
            DataFrame = pd.DataFrame(
                {'FileNames_Ch1': dirinfo['output_ch1_fnames'],
                 'FileNames_Ch2': dirinfo['output_ch2_fnames'],
                 'Merge_Counts': COUNTS
                })
            return DataFrame,Cells_Ch1

    else:
        if len(os.listdir(dirinfo['output_ch1']))==0:
            print('Ch1 must be counted before attempting to examine cell overlap')
        if len(os.listdir(dirinfo['output_ch2']))==0:
            print('Ch2 must be counted before attempting to examine cell overlap')



#############################################################################################################################


def display_count(count_out):  
    i_gray = hv.Image((np.arange(count_out['image'].shape[1]), 
                       np.arange(count_out['image'].shape[0]), 
                       count_out['image'])).opts(
               invert_yaxis=True,cmap='gray',toolbar='below',
               title="Original Image")
    i_gauss = hv.Image((np.arange(count_out['gauss'].shape[1]), 
                        np.arange(count_out['gauss'].shape[0]), 
                        count_out['gauss'])).opts(
               invert_yaxis=True,cmap='gray',toolbar='below',
               title="Preprocessed Image")
    i_thresh = hv.Image((np.arange(count_out['thresh'].shape[1]), 
                         np.arange(count_out['thresh'].shape[0]), 
                         count_out['thresh']*255)).opts(
               invert_yaxis=True,cmap='gray',toolbar='below',
               title="Thresholded Image")
    i_cells = hv.Image((np.arange(count_out['cells'].shape[1]), 
                        np.arange(count_out['cells'].shape[0]), 
                        count_out['cells']*(255//count_out['cells'].max()))).opts(
               invert_yaxis=True,cmap='jet',toolbar='below',
               title="Defined Cells")
    display = i_gray + i_gauss + i_thresh + i_cells
    return display
    
    
#############################################################################################################################


def display_merge(count_out1,count_out2,merge_out):  
    i_gray1 = hv.Image((np.arange(count_out1['image'].shape[1]), 
                       np.arange(count_out1['image'].shape[0]), 
                       count_out1['image'])).opts(
               invert_yaxis=True,cmap='gray',toolbar='below',
               title="Ch1 Original Image")
    
    i_gray2 = hv.Image((np.arange(count_out2['image'].shape[1]), 
                       np.arange(count_out2['image'].shape[0]), 
                       count_out2['image'])).opts(
               invert_yaxis=True,cmap='gray',toolbar='below',
               title="Ch2 Original Image")
    
    i_cells1 = hv.Image((np.arange(count_out1['cells'].shape[1]), 
                        np.arange(count_out1['cells'].shape[0]), 
                        count_out1['cells']*(255//count_out1['cells'].max()))).opts(
               invert_yaxis=True,cmap='jet',toolbar='below',
               title="Ch1 Defined Cells")
    
    i_cells2 = hv.Image((np.arange(count_out2['cells'].shape[1]), 
                    np.arange(count_out2['cells'].shape[0]), 
                    count_out2['cells']*(255//count_out2['cells'].max()))).opts(
           invert_yaxis=True,cmap='jet',toolbar='below',
           title="Ch2 Defined Cells")
    i_merge = hv.Image((np.arange(merge_out['cells'].shape[1]), 
                np.arange(merge_out['cells'].shape[0]), 
                merge_out['cells']*(255//merge_out['cells'].max()))).opts(
       invert_yaxis=True,cmap='jet',toolbar='below',
       title="Overlapping Cells")  
    display = i_gray1 + i_gray2 + i_cells1 + i_cells2 + i_merge
    return display
    
    
    
    