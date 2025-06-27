
import os, math
import numpy as np

def grids_nearest_or_within(src_grids={}, masked_pts={}, remask_approach = 'nearest', \
                     unlimit_xmin=False, unlimit_xmax=False, unlimit_ymin=False, unlimit_ymax=False,\
                     out2d=False, reorder_src=False, keep_duplicated=False):
    '''
    re-masking ELM domain grids, by providing centroids which also may be masked
        src_grids      - list of source grids, with required paired of 'xc' and 'yc', 
                         and/or its vertices 'xv', 'yv' for within-grid search
        masked_pts     - list of np arrays of 'xc', 'yc', 'mask' of user-provided grid-centroids or points
        remask_approach- 'nearest', or points in polygon if 'xv'/'yv' known
        unlimit_x(y)min(max) - x/y axis bounds are from source domain, when True; otherwise, from masked_pts
        out2d          - output data in 2D or flatten (so-called unstructured)
        reorder_src    - trunked source grids will re-order following that in masked_pts 
        keep_duplicated- don't sort or unique source grids. 
    '''

    import geopandas as geopd
    from shapely.geometry import Point
    from geopandas.tools import sjoin, sjoin_nearest
    from shapely.geometry import MultiLineString, LineString
    from shapely.ops import polygonize

    if len(masked_pts['xc'])<1 or len(masked_pts['yc'])<1: 
        print('Error: no masked paired xc/yc centroids provided! ')
        return    
    if len(src_grids['xc'])<1 or len(src_grids['yc'])<1: 
        print('Error: no source grids paired xc/yc centroids provided! ')
        return    

    # masked points, paired xc/yc (NOT blocked or box of [[xc],[yc]])
    mask_xc= masked_pts['xc']
    mask_yc= masked_pts['yc']
    LONGXY360=True
    if np.any(mask_xc<0.0): LONGXY360=False
    mask_id = np.indices(mask_xc.flatten().shape)[0]
        
    # 
    # Open the source domain or surface nc file, 
    # directly from an ELM domain nc file
    # TIP: in this case, both domain and land surface grid-systems are EXACTLY matching with each other
    src_yc = src_grids['yc']
    src_xc = src_grids['xc']
    nv = 4 # by default
    if LONGXY360: 
        src_xc[src_xc<0.0]=360+src_xc[src_xc<0.0]
    else:
        src_xc[src_xc>180.0]=-360+src_xc[src_xc>180.0]
    if 'xv' in src_grids.keys() \
        and 'yv' in src_grids.keys():
        src_xv = src_grids['xv']
        if LONGXY360: 
            src_xv[src_xv<0.0]=360+src_xv[src_xv<0.0]
        else:
            src_xv[src_xv>180.0]=-360+src_xv[src_xv>180.0]
            # need to re-adj for cross -180/180 line's grid
            
        src_yv = src_grids['yv']
        nv = src_xv.shape[-1]
    
    # 
    # 0.5 times of grid-size edges 
    # to allows one-grid edges of range of latitude/longitude boxes
    x_edge = 0.0; y_edge = 0.0
    if (1 in src_xc.shape):
        #probably unstructed grids
        if 'xv' in src_grids.keys() and \
            'yv' in src_grids.keys():
            xdiff = np.squeeze(abs(src_xv[...,0]-src_xv[...,1]))
            xdiff = xdiff[np.where(xdiff<=180.0)] # removal those cross line 0/360 or -180/180
            x_edge = np.nanmean(xdiff)*0.5
            y_edge = np.nanmean(np.squeeze(abs(src_yv[...,0]-src_yv[...,2])))*0.5        
    else:
        xdiff = np.diff(src_xc[0,:])
        xdiff = xdiff[np.where(abs(xdiff)<=180.0)] # removal those cross line 0/360 or -180/180
        x_edge = abs(np.nanmean(xdiff)*0.5)
        y_edge = abs(np.nanmean(np.diff(src_yc[:,0]))*0.5)
    # [mask_xc,mask_yc] box
    x_min = min(mask_xc)-x_edge
    x_max = max(mask_xc)+x_edge
    y_min = min(mask_yc)-y_edge
    y_max = max(mask_yc)+y_edge
    # but may not want to do trunking
    if (1 in src_xc.shape):
        if unlimit_xmin: x_min = min(src_xc[0,:])
        if unlimit_ymin: y_min = min(src_yc[0,:])
        if unlimit_xmax: x_max = max(src_xc[0,:])
        if unlimit_ymax: y_max = max(src_yc[0,:])
    else:
        if unlimit_xmin: x_min = min(src_xc[0,:])
        if unlimit_ymin: y_min = min(src_yc[:,0])
        if unlimit_xmax: x_max = max(src_xc[0,:])
        if unlimit_ymax: y_max = max(src_yc[:,0])
    
    boxed_idx = \
        np.where((src_yc>=y_min) &  \
                 (src_yc<=y_max) &  \
                 (src_xc>=x_min) &  \
                 (src_xc<=x_max))
    boxed_xc = src_xc[boxed_idx]
    boxed_yc = src_yc[boxed_idx]
    poly_id = np.indices(src_xc.flatten().shape)[0]
    poly_id = poly_id.reshape(src_xc.shape)[boxed_idx]
    # this is the indices of trunked but in original paired [yc,xc] of source  

    if 'xv' in src_grids.keys() and \
        'yv' in src_grids.keys():    
        boxed_yv1 = src_yv[...,0]; boxed_xv1 = src_xv[...,0]
        boxed_yv2 = src_yv[...,1]; boxed_xv2 = src_xv[...,1]
        boxed_yv3 = src_yv[...,2]; boxed_xv3 = src_xv[...,2]
        if nv==4: boxed_yv4 = src_yv[...,3]; boxed_xv4 = src_xv[...,3]        
        boxed_yv1=boxed_yv1[boxed_idx]; boxed_xv1=boxed_xv1[boxed_idx]
        boxed_yv2=boxed_yv2[boxed_idx]; boxed_xv2=boxed_xv2[boxed_idx]
        boxed_yv3=boxed_yv3[boxed_idx]; boxed_xv3=boxed_xv3[boxed_idx]
        if nv==4: boxed_yv4=boxed_yv4[boxed_idx]; boxed_xv4=boxed_xv4[boxed_idx]
    
    # trunked source domain's xc,yc, i.e. centroids of xv/yv which formed polygons above, 
    # to be marked in X_axis, Y_axis, and re-indexing
    # (TIPs: this is import for visualizing data in map or transfrom btw 1D and 2D)
    X_axis = boxed_xc
    Y_axis = boxed_yc     
    [X_axis, i] = np.unique(X_axis, return_inverse=True)
    [Y_axis, j] = np.unique(Y_axis, return_inverse=True)
    YY, XX = np.meshgrid(Y_axis, X_axis, indexing='ij')
    xid = np.indices(XX.shape)[1]
    yid = np.indices(XX.shape)[0]
    xyid = np.indices(XX.flatten().shape)[0]
    xyid = np.reshape(xyid, xid.shape)

    if (1 not in src_xc.shape):
        # structured grids yc[nj,ni], xc[nj,ni]
        
        # new indices of paired [yc,xc] in grid-mesh YY/XX
        # note: here don't override xc,yc, so xc=X_axis[i], yc=Y_axis[j]
        yxc = np.asarray([(y,x) for y,x in zip(boxed_yc,boxed_xc)])
        yxc_remeshed = np.asarray([(y,x) for y,x in zip(YY.flatten(),XX.flatten())])
        set1 = dict((k,i) for i,k in enumerate({tuple(row) for row in yxc}))
        set2 = dict((k,i) for i,k in enumerate({tuple(row) for row in yxc_remeshed}))
        inter_set = set(set2).intersection(set(set1))
        ji = np.asarray([set2[yx] for yx in inter_set])
        ji = np.unravel_index(ji, xyid.shape)
        sub_xid = xid[ji]     
        sub_yid = yid[ji]    
        sub_xyid= xyid[ji]  

    else:
        # unstructured  yc[1,ni], xc[1,ni]

        # new indices of paired [yc,xc] in grid-mesh YY/XX
        sub_xid = xid[j,i]     
        sub_yid = yid[j,i] 
        sub_xyid= xyid[j,i]        
    #

    # masked points
    points = geopd.GeoDataFrame({"pts_indices":mask_id, \
                                 "x":mask_xc,"y":mask_yc})
    points['geometry'] = points.apply(lambda p: Point(p.x, p.y), axis=1)

    #
    #remask_approach = 'nearest'
    if remask_approach=='nearest':
        print('nearest point searching')

        # re-meshed XX/YY centroids
        grid_centroids = geopd.GeoDataFrame({'pid':poly_id,"xyid":sub_xyid,"xid":sub_xid,"yid":sub_yid, \
                                             "xc":boxed_xc,"yc":boxed_yc})
        grid_centroids['geometry'] = grid_centroids.apply(lambda p: Point(p.xc, p.yc), axis=1)

        pointsInGrids = sjoin_nearest(points, \
                                      grid_centroids[['pid','xyid','xid','yid','geometry']], how='inner')
        
        remask_pid = pointsInGrids.pid  # the original (untrunked) grid/polygon flatten id
        remask_xyid = pointsInGrids.xyid  # the grid/polygon flatten id in new XX/YY mesh (re-ordered)
        remask_maskid = pointsInGrids.pts_indices # the orginal (masking) point id

    
    elif 'xv' in src_grids.keys() \
        and 'yv' in src_grids.keys():
        print('points in polygon searching')
        
        # box trunked source domain's xv,yv to form polygons
        vpts1=[(x,y) for x,y in zip(boxed_xv1,boxed_yv1)]
        vpts2=[(x,y) for x,y in zip(boxed_xv2,boxed_yv2)]
        vpts3=[(x,y) for x,y in zip(boxed_xv3,boxed_yv3)]
        lines = [(vpts1[i], vpts2[i], vpts3[i], vpts1[i]) for i in range(len(vpts1))]
        if nv==4: 
            vpts4=[(x,y) for x,y in zip(boxed_xv4,boxed_yv4)]
            lines = [(vpts1[i], vpts2[i], vpts3[i], vpts4[i], vpts1[i]) for i in range(len(vpts1))]
        
        lines_str = [LineString(lines[i]) for i in range(len(lines))]
        polygons = list(polygonize(MultiLineString(lines_str)))
        # if regular x/y mesh may be do like following: 
        #x = np.linspace(src_xv[0,0], xv[0,-1], len(src_xc[1]))         
        #y = np.linspace(src_yv[0,0], yv[0,-1], len(src_yc[0]))    
        #hlines = [((x1, yi), (x2, yi)) for x1, x2 in zip(x[:-1], x[1:]) for yi in y]
        #vlines = [((xi, y1), (xi, y2)) for y1, y2 in zip(y[:-1], y[1:]) for xi in x]
        #lines_str = vlines+hlines
        #polygons = list(polygonize(MultiLineString(lines_str)))
        
        grids = geopd.GeoDataFrame({'pid':poly_id,"xyid":sub_xyid,"xid":sub_xid,"yid":sub_yid,"geometry":polygons})
    
        pointsInPolys = sjoin(points, grids[['pid','xyid','xid','yid','geometry']], how='inner')
        remask_pid = pointsInPolys.pid.values  # the original (untrunked) grid/polygon flatten id
        remask_xyid = pointsInPolys.xyid.values  # the grid/polygon flatten id in new XX/YY mesh (re-ordered)        
        remask_maskid = pointsInPolys.pts_indices.values # the orginal (masking) point id
        
    #
        
    # need to obtain all original masked_pts id for each srcpts_unique_pid
    # so that, could do some maths within a srcpts_unique_pid
    srcpolys_unique_pid, pts_startidx, pts_count  = np.unique(remask_pid, \
                            return_index=True, return_counts=True)
    sorted_ptsingrids = np.column_stack((remask_pid, remask_maskid))
    sorted_ptsingrids = sorted_ptsingrids[sorted_ptsingrids[:,0].argsort()]      
    srcpolys_contained_maskid = {}
    for i in range(len(srcpolys_unique_pid)):
        # sorted unique grid id as key, for which masked_pid contained can be saved
        srcpolys_contained_maskid[srcpolys_unique_pid[i]] = \
            sorted_ptsingrids[pts_startidx[i]:pts_startidx[i]+pts_count[i],1]

    # re-masking grids, which actually conaining masked_xc/_yc/_id in new XX/YY mesh
    sub_remasked = np.isin(sub_xyid, remask_xyid)

    if keep_duplicated or reorder_src:
        
        sorted_ptsidx = remask_xyid # not really sorted here
        if keep_duplicated:
            boxed_xc = boxed_xc[sorted_ptsidx]
            boxed_yc = boxed_yc[sorted_ptsidx]
        elif reorder_src:
            # shuffle indices or data array by data order in mask
            sorted_ptsingrids = np.column_stack((remask_maskid, remask_pid, remask_xyid))
            sorted_ptsingrids = sorted_ptsingrids[sorted_ptsingrids[:,0].argsort()]
            sorted_ptsidx = sorted_ptsingrids[:,2] # i.e. sorted 'remask_xyid' by 'remask_maskid'

            # using mask_pts x/y, but still with src_xv/yv below
            boxed_xc = mask_xc[remask_maskid]
            boxed_yc = mask_yc[remask_maskid]
                    
        if 'xv' in src_grids.keys() and \
            'yv' in src_grids.keys():    
            boxed_yv1=boxed_yv1[sorted_ptsidx]; boxed_xv1=boxed_xv1[sorted_ptsidx]
            boxed_yv2=boxed_yv2[sorted_ptsidx]; boxed_xv2=boxed_xv2[sorted_ptsidx]
            boxed_yv3=boxed_yv3[sorted_ptsidx]; boxed_xv3=boxed_xv3[sorted_ptsidx]
            if nv==4: boxed_yv4=boxed_yv4[sorted_ptsidx]; boxed_xv4=boxed_xv4[sorted_ptsidx]
        
        # re-do indices of [yc,xc] in [YY,XX] meshgrid
        boxed_idx = (boxed_idx[0][sorted_ptsidx], boxed_idx[1][sorted_ptsidx])
        sub_xyid = sub_xyid[sorted_ptsidx]
        sub_xid= sub_xid[sorted_ptsidx]
        sub_yid= sub_yid[sorted_ptsidx]
        sub_remasked = sub_remasked[sorted_ptsidx]
        

    if out2d and (keep_duplicated or reorder_src):
        # to 2D but masked    
        grid_id_arr = np.reshape(sub_xyid,xyid.shape)
        grid_xid_arr = np.reshape(sub_xid,xyid.shape)
        grid_yid_arr = np.reshape(sub_yid,xyid.shape)
        grid_remasked_arr = np.reshape(sub_remasked,xyid.shape)
        
        xc_arr = np.reshape(boxed_xc,xyid.shape)
        yc_arr = np.reshape(boxed_yc,xyid.shape)
        xv_arr = np.stack((np.reshape(boxed_xv1,xyid.shape), \
                           np.reshape(boxed_xv2,xyid.shape), \
                           np.reshape(boxed_xv3,xyid.shape), \
                           np.reshape(boxed_xv4,xyid.shape)), axis=2)
        yv_arr = np.stack((np.reshape(boxed_yv1,xyid.shape), \
                           np.reshape(boxed_yv2,xyid.shape), \
                           np.reshape(boxed_yv3,xyid.shape), \
                           np.reshape(boxed_yv4,xyid.shape)), axis=2)

    else:
    
        grid_id_arr = sub_xyid
        grid_xid_arr = sub_xid
        grid_yid_arr = sub_yid
        grid_remasked_arr = sub_remasked
        xc_arr = boxed_xc
        yc_arr = boxed_yc
        xv_arr = np.stack((boxed_xv1,boxed_xv2,boxed_xv3), axis=1)
        yv_arr = np.stack((boxed_yv1,boxed_yv2,boxed_yv3), axis=1)
        if nv==4:
            xv_arr = np.stack((boxed_xv1,boxed_xv2,boxed_xv3,boxed_xv4), axis=1)
            yv_arr = np.stack((boxed_yv1,boxed_yv2,boxed_yv3,boxed_yv4), axis=1)
 
    # 
    # output arrays
    subdomain = {}
    subdomain['X_axis'] = X_axis
    subdomain['Y_axis'] = Y_axis
    subdomain['XX'] = XX
    subdomain['YY'] = YY
            
    subdomain['gridID']  = grid_id_arr
    subdomain['gridXID'] = grid_xid_arr
    subdomain['gridYID'] = grid_yid_arr
    subdomain['xc'] = xc_arr
    subdomain['yc'] = yc_arr
    subdomain['xv'] = xv_arr
    subdomain['yv'] = yv_arr
    subdomain['remasked'] = grid_remasked_arr
        
    return subdomain, boxed_idx, srcpolys_unique_pid, srcpolys_contained_maskid
#
