import pylab
import os
import sys
import fnmatch
import cv2
import numpy as np
import mahotas as mh
import pandas as pd
import holoviews as hv
from holoviews import streams
from holoviews.streams import Stream, param
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
    if not os.path.exists(dirinfo['output']): os.mkdir(dirinfo['output'])
    if os.path.isdir(dirinfo['ch1']):
        dirinfo['ch1_fnames'] = sorted(os.listdir(dirinfo['ch1']))
        dirinfo['ch1_fnames'] = fnmatch.filter(dirinfo['ch1_fnames'], '*.tif')
        dirinfo['output_ch1'] = os.path.join(os.path.normpath(dirinfo['output']), "Ch1")
        if not os.path.isdir(dirinfo['output_ch1']): os.mkdir(dirinfo['output_ch1'])
    if os.path.isdir(dirinfo['ch2']):
        dirinfo['ch2_fnames'] = sorted(os.listdir(dirinfo['ch2']))
        dirinfo['ch2_fnames'] = fnmatch.filter(dirinfo['ch2_fnames'], '*.tif')
        dirinfo['output_ch2'] = os.path.join(os.path.normpath(dirinfo['output']), "Ch2")
        if not os.path.isdir(dirinfo['output_ch2']): os.mkdir(dirinfo['output_ch2'])
    if os.path.isdir(dirinfo['ROI']):
        dirinfo['roi_fnames'] = sorted(os.listdir(dirinfo['ROI']))
        dirinfo['roi_fnames'] = fnmatch.filter(dirinfo['roi_fnames'], '*.tif')
    if os.path.isdir(dirinfo['ch1']) and os.path.isdir(dirinfo['ch2']):
        dirinfo['output_merge'] = os.path.join(os.path.normpath((dirinfo['output'])), "Merge")
        if not os.path.isdir(dirinfo['output_merge']): os.mkdir(dirinfo['output_merge'])

    return dirinfo


#############################################################################################################################


def optim_getdirinfo(dirinfo):

    dirinfo['composite'] = os.path.join(os.path.normpath(dirinfo['main']), "Composite")
    dirinfo['manual'] = os.path.join(os.path.normpath(dirinfo['main']), "ManualCounts")
    dirinfo['output'] = os.path.join(os.path.normpath(dirinfo['main']), "SavedOutput")
    if not os.path.isdir(dirinfo['output']): os.mkdir(dirinfo['output'])
    dirinfo['composite_fnames'] = sorted(os.listdir(dirinfo['composite']))
    dirinfo['composite_fnames'] = fnmatch.filter(dirinfo['composite_fnames'], '*.tif')
    dirinfo['manual_fnames'] = sorted(os.listdir(dirinfo['manual']))
    dirinfo['manual_fnames'] = fnmatch.filter(dirinfo['manual_fnames'], '*.tif')

    return dirinfo


#############################################################################################################################


def optim_getimages(dirinfo,params):

    images = {
        'manual' : cv2.imread(os.path.join(os.path.normpath(dirinfo['manual']), dirinfo['manual_fnames'][0]),
                              cv2.IMREAD_GRAYSCALE),
        'composite' : cv2.imread(os.path.join(os.path.normpath(dirinfo['composite']), dirinfo['composite_fnames'][0]),
                                cv2.IMREAD_GRAYSCALE)
    }

    images['median'] = medianFilter(images['composite'], ksize = params['diam']//2)
    images['bg'] = subtractbg(images['median'], ksize = params['diam']*3)
    images['gauss'] = cv2.GaussianBlur(images['bg'],(0,0),params['diam']/6)
    params['counts'] = (images['manual']>0).sum()
    params['otsu'], _ = cv2.threshold(images['gauss'].astype('uint64'),0,255,
                                      cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    params['thresh'] = params['otsu']
    images['otsu'] = images['gauss'] > params['thresh']
    count_out = Count(0,"Optim",params,dirinfo,UseROI=False,UseWatershed=params['UseWatershed'])

    i_comp = mkimage(images['composite'], title="Composite Image")
    i_gauss = mkimage(images['gauss'], title="Preprocessed Image")
    i_otsu = mkimage(images['otsu']*255, title="Otsu Thresholded Image")
    i_cells = mkimage(count_out['cells']*(255//count_out['cells'].max()),
                      title="Cells Counted Using Otsu").opts(cmap='jet')
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
            for cell in range (1,count_out['nr_nuclei']+1):
                HitMap = images['manual'][count_out['cells']==cell]
                if HitMap.sum()==255:
                    Hits += 1
        List_Hits.append(Hits)

        #Calculate Accuracies
        hoac = Hits/count_out['nr_nuclei'] if count_out['nr_nuclei'] > 0 else np.nan
        List_Acc_HitsOverAutoCounts.append(hoac)
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
    return DataFrame, images['manual']


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
    Image_Current_File = os.path.join(os.path.normpath(Directory_Current), FileNames_Current[file])
    Image_Current_Gray = cv2.imread(Image_Current_File,cv2.IMREAD_GRAYSCALE)
    print("Processing: " + FileNames_Current[file])

    #Process file
    Image_Current_Median = medianFilter(Image_Current_Gray, ksize = CellDiam//2)
    Image_Current_BG = subtractbg(Image_Current_Gray, ksize = CellDiam*3)
    Image_Current_Gaussian = cv2.GaussianBlur(Image_Current_BG.astype('float'),(0,0),CellDiam/6)
    Image_Current_T = rm_smallparts(Image_Current_Gaussian > Thresh, CellDiam, params['particle_min'])

    if UseROI:
        ROI_Current_File = os.path.join(os.path.normpath(dirinfo['ROI']), dirinfo['roi_fnames'][file])
        ROI_Current = cv2.imread(ROI_Current_File,cv2.IMREAD_GRAYSCALE)
        Image_Current_T[ROI_Current==0]=0
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
        count_out = Count(file,Channel,params,dirinfo,UseROI=UseROI,UseWatershed=UseWatershed)
        COUNTS.append(count_out['nr_nuclei'])
        ROI_SIZE.append(count_out['roi_size'])
        cv2.imwrite(
            filename = os.path.splitext(
                os.path.join(os.path.normpath(output),
                             fnames[file]))[0] + '_Counts.tif',
            img = count_out['cells'].astype(np.uint16)
        )

    #Create DattaFrame
    if Channel == "Ch1":
        DataFrame = pd.DataFrame(
        {'Ch1_FileNames': fnames,
         'Ch1_Thresh' : np.ones(len(fnames))*thresh,
         'Ch1_UseROI' : np.ones(len(fnames))*UseROI,
         'Ch1_AvgCellDiam' : np.ones(len(fnames))*diam,
         'Ch1_ParticleMin' : np.ones(len(fnames))*params['particle_min'],
         'Ch1_Counts': COUNTS,
         'Ch1_ROIsize': ROI_SIZE
        })
        return DataFrame
    if Channel == "Ch2":
        DataFrame = pd.DataFrame(
        {'Ch2_FileNames': fnames,
         'Ch2_Thresh' : np.ones(len(fnames))*thresh,
         'Ch2_UseROI' : np.ones(len(fnames))*UseROI,
         'Ch2_AvgCellDiam' : np.ones(len(fnames))*diam,
         'Ch2_ParticleMin' : np.ones(len(fnames))*params['particle_min'],
         'Ch2_Counts': COUNTS,
         'Ch2_ROIsize': ROI_SIZE
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
        Regmax_bc = np.ones((CellDiam,CellDiam))
        Image_Current_Seeds = mh.locmax(Image_Current_Tdist,Bc=Regmax_bc)
        Image_Current_Seeds[Image_Current_Tdist==0]=False
        Image_Current_Seeds = mh.dilate(Image_Current_Seeds,np.ones((3,3)))
        seeds,nr_nuclei = mh.label(Image_Current_Seeds,Bc=np.ones((3,3)))
        Image_Current_Unknown = Image_Current_SureBackground.astype(int) - Image_Current_Seeds.astype(int)
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

    SmallCellDiam = params['ch1_diam'] if params['ch1_diam'] < params['ch2_diam'] else params['ch2_diam']
    size_req = SmallCellDiam*SmallCellDiam*params['overlap']
    merge = np.zeros(Cells_Ch1.shape)

    nr_nuclei = 0
    for c1_cell in range(1,Cells_Ch1.max()+1):
        for c2_cell in range(1,Cells_Ch2.max()+1):
            overlap_area=sum(Cells_Ch2[Cells_Ch1==c1_cell]==c2_cell)
            if overlap_area > size_req:
                nr_nuclei+=1
                merge[Cells_Ch1==c1_cell]=nr_nuclei

    print(nr_nuclei)

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
        dirinfo['output_ch2_fnames'] = sorted(os.listdir(dirinfo['output_ch2']))
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

            COUNTS = []
            for file in range (len(dirinfo['output_ch1_fnames'])):
                print('Processing: {x}'.format(x=dirinfo['ch1_fnames'][file]))
                Cells_Ch1 = cv2.imread(os.path.join(os.path.normpath(dirinfo['output_ch1']),
                                                    dirinfo['output_ch1_fnames'][file]),
                                       cv2.IMREAD_ANYDEPTH)
                Cells_Ch2 = cv2.imread(os.path.join(os.path.normpath(dirinfo['output_ch2']),
                                                    dirinfo['output_ch2_fnames'][file]),
                                       cv2.IMREAD_ANYDEPTH)
                merge_out = Merge(Cells_Ch1,Cells_Ch2,params)
                COUNTS.append(merge_out['nr_nuclei'])
                cv2.imwrite(
                    filename = os.path.join(os.path.normpath(dirinfo['output_merge']),
                                    dirinfo['output_ch1_fnames'][file] + "_merge.tif"),
                    img = merge_out['cells'].astype(np.uint16)
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


def ROI_plot(directory,fnames,file,region_names=None):

    #Get image
    try:
        Image_Current_File = os.path.join(os.path.normpath(directory), fnames[file])
        img = cv2.imread(Image_Current_File,cv2.IMREAD_GRAYSCALE)
        print(Image_Current_File)
        print('file: {}'.format(file))
    except IndexError:
        print('Max file index exceeded. All images in folder drawn.')
        return None,None,None

    #get number of objects to be drawn
    nobjects = len(region_names) if region_names else 0

    #Make reference image the base image on which to draw
    image_title = "No Regions to Draw" if nobjects == 0 else "Draw Regions: "+', '.join(region_names)
    image = hv.Image((np.arange(img.shape[1]), np.arange(img.shape[0]), img))
    image.opts(width=int(img.shape[1]),
               height=int(img.shape[0]),
              invert_yaxis=True,cmap='gray',
              colorbar=True,
               toolbar='below',
              title=image_title)

    #Create polygon element on which to draw and connect via stream to PolyDraw drawing tool
    poly = hv.Polygons([])
    poly_stream = streams.PolyDraw(source=poly, drag=True, num_objects=nobjects, show_vertices=True)
    poly.opts(fill_alpha=0.3, active_tools=['poly_draw'])

    def centers(data):
        try:
            x_ls, y_ls = data['xs'], data['ys']
        except TypeError:
            x_ls, y_ls = [], []
        xs = [np.mean(x) for x in x_ls]
        ys = [np.mean(y) for y in y_ls]
        rois = region_names[:len(xs)]
        return hv.Labels((xs, ys, rois))

    if nobjects > 0:
        dmap = hv.DynamicMap(centers, streams=[poly_stream])
        return (image * poly * dmap), poly_stream, img.shape
    else:
        return (image), None, img.shape


#############################################################################################################################


def ROI_mkMasks(directory,fnames,file,region_names,ROI_stream,img_shape):

    ROI_masks = {}
    for poly in range(len(ROI_stream.data['xs'])):
        x = np.array(ROI_stream.data['xs'][poly]) #x coordinates
        y = np.array(ROI_stream.data['ys'][poly]) #y coordinates
        xy = np.column_stack((x,y)).astype('uint64') #xy coordinate pairs
        mask = np.zeros(img_shape) # create empty mask
        cv2.fillPoly(mask, pts =[xy], color=255) #fill polygon
        ROI_masks[region_names[poly]] = mask #save to ROI masks as boolean

        outname = "{cfile}_mask_{region}.tif".format(cfile=os.path.splitext(fnames[file])[0],
                                                  region=region_names[poly])

        cv2.imwrite(
            filename = os.path.join(os.path.normpath(directory), outname),
            img = mask
        )

    return ROI_masks


#############################################################################################################################


def display_count(count_out):

    i_gray = mkimage(count_out['image'], title = "Original Image")
    i_gauss = mkimage(count_out['gauss'], title = "Preprocessed Image")
    i_thresh = mkimage(count_out['thresh']*255, title = "Thresholded Image")
    i_cells = mkimage((count_out['cells']/count_out['cells'].max())*255,
                         title = "Defined Cells").opts(cmap="jet")
    #i_cells = mkimage(count_out['cells']*(255//count_out['cells'].max()),
    #                  title = "Defined Cells").opts(cmap="jet")
    display = i_gray + i_gauss + i_thresh + i_cells
    return display


#############################################################################################################################


def display_merge(count_out1,count_out2,merge_out):

    i_gray1 = mkimage(count_out1['image'], title = "Ch1 Original Image")
    i_gray2 = mkimage(count_out2['image'], title = "Ch2 Original Image")
    i_cells1 = mkimage(count_out1['cells']*(255//count_out1['cells'].max()),
                       title = "Ch1 Defined Cells")
    i_cells2 = mkimage(count_out2['cells']*(255//count_out1['cells'].max()),
                       title = "Ch2 Defined Cells")
    i_merge = mkimage(merge_out['cells']*(255//merge_out['cells'].max()),
                     title = "Overlapping Cells")
    display = i_gray1 + i_gray2 + i_cells1 + i_cells2 + i_merge
    return display


#############################################################################################################################


def medianFilter(image, ksize):

    ksize = (ksize-1) if (ksize%2 == 0) else ksize
    image = cv2.medianBlur(image, ksize)
    return image


#############################################################################################################################


def subtractbg(image, ksize):

    image = image.astype('float')
    bg = cv2.GaussianBlur(image,
                         (0,0),
                         ksize)
    new_image = image - bg
    new_image[new_image<0] = 0
    return new_image


#############################################################################################################################


def mkimage(image, title=""):

    image = hv.Image((np.arange(image.shape[1]),
                   np.arange(image.shape[0]),
                   image)).opts(
           invert_yaxis=True,cmap='gray',toolbar='below',
           title= title)

    return image


#############################################################################################################################


def rm_smallparts (image, celldiam, pmin):
    labeled,nr_objects = mh.label(image)
    sizes = mh.labeled.labeled_size(labeled)
    too_small = np.where(sizes < (celldiam*celldiam*pmin))
    labeled = mh.labeled.remove_regions(labeled, too_small)
    Image_Current_T = labeled != 0
    return Image_Current_T


#############################################################################################################################


def split_channels(dirinfo):

    dirinfo['fnames'] = sorted(os.listdir(dirinfo['main']))
    dirinfo['fnames'] = fnmatch.filter(dirinfo['fnames'], 
                                       '.'.join(['*',dirinfo['fext']]))

    for channel in dirinfo['cnames']:
        dirinfo[channel] = os.path.join(os.path.normpath(dirinfo['main']), channel)
        if not os.path.exists(dirinfo[channel]): os.mkdir(dirinfo[channel])

    for file in dirinfo['fnames']:
        print(file)
        image = cv2.imread(os.path.join(os.path.normpath(dirinfo['main']), file), cv2.IMREAD_COLOR)
        depth = image.dtype
        for index, channel in enumerate(dirinfo['cnames']): 
            cv2.imwrite(
                filename = os.path.join(
                    os.path.normpath(dirinfo[channel]),
                    '.'.join(['_'.join([file,channel]) , dirinfo['fext']])),
                img = image[:,:,index].astype(depth))
