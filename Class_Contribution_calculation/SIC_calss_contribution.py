import numpy as np
import os
import matplotlib.pyplot as plt
import numpy.ma as ma
import scipy.stats as stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath



def Statistical_sig_test(dist1,dist2):
    """
    Determine if there is a statistically significant difference between two distributions.
    Parameters:
    dist1 : numpy.ndarray
        An array of values representing the first distribution.
    dist2 : numpy.ndarray
        An array of values representing the second distribution.
    
    Returns:
    is_significant : bool
        True if there is a statistically significant difference (p < 0.05), False otherwise.
    """
    levene_stat, levene_p = stats.levene(dist1, dist2)
    if levene_p > 0.05:
        t_stat, p_value = stats.ttest_ind(dist1, dist2, equal_var=True)
    else:
        t_stat, p_value = stats.ttest_ind(dist1, dist2, equal_var=False)
    if p_value <0.05:
        is_significant = 1
    else:
        is_significant = 0
    return is_significant



def seasonal_id(MONTH):
    """
    Get the month of the year corresponding to the daily data points and return the seasonal point indices.
    Parameters:
    MONTH : numpy.ndarray
        An array of integers representing the month of each daily data point, with values ranging from 1 (January) to 12 (December).
    Returns:
    tuple
        A tuple containing four arrays, each with the indices of the data points corresponding to a particular season:
        (winter_indices, spring_indices, summer_indices, fall_indices)
    """
    win1= np.where(MONTH==12)[0]
    win2= np.where(MONTH==1)[0]
    win3= np.where(MONTH==2)[0]
    win = np.concatenate(( win1,  win2), axis=0)
    win = np.concatenate(( win,  win3), axis=0)
    spr1= np.where(MONTH==3)[0]
    spr2= np.where(MONTH==4)[0]
    spr3= np.where(MONTH==5)[0]
    spr = np.concatenate(( spr1,  spr2), axis=0)
    spr = np.concatenate(( spr,  spr3), axis=0)
    sum1= np.where(MONTH==6)[0]
    sum2= np.where(MONTH==7)[0]
    sum3= np.where(MONTH==8)[0]
    sum = np.concatenate(( sum1,  sum2), axis=0)
    sum = np.concatenate(( sum,  sum3), axis=0)
    fal1= np.where(MONTH==9)[0]
    fal2= np.where(MONTH==10)[0]
    fal3= np.where(MONTH==11)[0]
    fal = np.concatenate(( fal1,  fal2), axis=0)
    fal = np.concatenate(( fal,  fal3), axis=0)
    return win, spr, sum, fal


    
def Seasonal_Anomaly(Sea_Ice_C, Sea_Ice_E, Month):
    """
    Calculate seasonal anomalies and determine statistical significance.
    Parameters:
    Sea_Ice_C : numpy.ndarray
        Daily SIC data points for each day in the Control run, expected shape (days, projection x, projection y).
    Sea_Ice_E : numpy.ndarray
        Daily SIC data points for each day in the Experiment run, expected shape (days, projection x, projection y).
    Month : numpy.ndarray
        Month ID for each data point in Sea_Ice_C and Sea_Ice_E, expected length equal to the number of days.
    Returns:
    tuple
        - Sea_Ice_seasonal_Anomaly: numpy.ndarray
            An array consisting of seasonal anomaly values (E-C), shape (4 [Winter, Spring, Summer, Fall], projection x, projection y).
        - Sea_Ice_Significant_HW: numpy.ma.masked_array
            A mask that indicates whether the anomaly is statistically significant (1 if not significant, 0 if significant) for each pixel, shape (4 [Winter, Spring, Summer, Fall], projection x, projection y).
    """
    Season_id = [-1]*4
    Season_id[0], Season_id[1], Season_id[2], Season_id[3]= seasonal_id(Month[:-8])   # datapoints ids for different season (Winter[0], Spring[1], Summer[2], Autumn[3])
    Sea_Ice_seasonal_Anomaly = np.zeros((4,Sea_Ice_C.shape[1],Sea_Ice_C.shape[2]))   # SIC seasonal anomaly (E-C)
    for k in range(4):
        Sea_Ice_seasonal_Anomaly[k] = Sea_Ice_E[Season_id[k],:,:].mean(axis=(0))  -  Sea_Ice_C[Season_id[k],:,:].mean(axis=(0))
    # see whether they are statistically significant
    Sea_Ice_Significant_HW=  np.ones((4,Sea_Ice_C.shape[1],Sea_Ice_C.shape[2]))     # a mask for pixels that the corresponded anomaly is not statistically signifcant
    for k in range(4):
        for i in range(Sea_Ice_C.shape[1]):
            for j in range(Sea_Ice_C.shape[2]):
                Sea_Ice_Significant_HW[k,i,j] = not(Statistical_sig_test(Sea_Ice_C[Season_id[k],i,j],Sea_Ice_E[Season_id[k],i,j]))
    Sea_Ice_Significant_HW =  ma.masked_array(Sea_Ice_Significant_HW, mask = Sea_Ice_Significant_HW)
    return Sea_Ice_seasonal_Anomaly, Sea_Ice_Significant_HW



def ploting_seasonal_anomaly(Sea_Ice_seasonal_Anomaly, Sea_Ice_Significant_HW, season):
    """
    Plot seasonal anomalies for sea ice concentration and highlight where anomalies are statistically significant.
    Parameters:
    Sea_Ice_seasonal_Anomaly : numpy.ndarray
        An array containing seasonal anomaly values (E-C), with shape (4 [Winter, Spring, Summer, Fall], projection x, projection y).
    Sea_Ice_Significant_HW : numpy.ndarray
        A mask that indicates where anomalies are statistically significant (0 if significant, 1 if not), shape (4 [Winter, Spring, Summer, Fall], projection x, projection y).
    season : str
        The name of the season to plot. Must be one of ['Winter', 'Spring', 'Summer', 'Fall'].
    Outputs:
    A plot showing the sea ice concentration anomalies for the specified season, with areas of statistical significance highlighted.
    """
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    # clreating the projection
    projection = ccrs.LambertAzimuthalEqualArea(central_latitude=90, central_longitude=0)
    # transfer lat an lon to projection axis
    xy = projection.transform_points(ccrs.PlateCarree(), grid_lon, grid_lat)
    x = xy[:,:,0]
    y = xy[:,:,1]
    Season_name = ['Winter','Spring','Summer','Fall']
    max_Sea_Ice = max([(-1)*Sea_Ice_seasonal_Anomaly.min(),Sea_Ice_seasonal_Anomaly.max()])
    i = Season_name.index(season)
    fig, ax = plt.subplots(subplot_kw={'projection': projection})
    im = ax.imshow(Sea_Ice_seasonal_Anomaly[i], vmax = max_Sea_Ice, vmin = (-1)*max_Sea_Ice, cmap='seismic',  extent=(x.min(), x.max(), y.min(), y.max()), origin='upper')
    ax.quiver(x, y, Sea_Ice_Significant_HW[i,:,:]+1, Sea_Ice_Significant_HW[i,:,:]+1, pivot='middle')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.set_extent([0, 360, 40, 90], ccrs.PlateCarree())
    ax.set_title("whole domain mean anomaly sea ice for "+Season_name[i])
    ax.set_boundary(circle, transform=ax.transAxes)
    plt.colorbar(im, ax=ax, orientation='vertical', label='SIC anomaly')
    plt.show()
    plt.close()





def Class_contribution(Sea_Ice_C, Sea_Ice_E, Month, predict_label_C, predict_label_E):
    """
    This function computes the class contribution for a SIC (could be any other parameter), 
    organized into different clusters (here 6 clusters) based on their spatiotemporal and associated
    PSMET patterns (clustering based on MCAE latent space, this could be any grouping of data points). 
    The anomaly is defined as the difference between the seasonal mean of SIC in the Experiment run data 
    and the Control run data, normalized by the number of data points in each cluster for each season.
    #
    The function returns the fields of WCVC and FSDC for each cluster and season, providing insights into how 
    different clusters contribute to the overall seasonal variability of the parameter X (here SIC).
    Parameters:
    - Sea_Ice_C : numpy.ndarray
        Daily SIC data points for each day in the Control run, expected shape (days, projection x, projection y).
    - Sea_Ice_E : numpy.ndarray
        Daily SIC data points for each day in the Experiment run, expected shape (days, projection x, projection y).
    - Month : numpy.ndarray
        Month ID for each data point in Sea_Ice_C and Sea_Ice_E, expected length equal to the number of days.
    - predict_label_C : numpy.ndarray: 
        Cluster labels for each data point in the Control run.
    - predict_label_E : numpy.ndarray: 
        Cluster labels for each data point in the Experiment run.
    #
    Returns:
    tuple
        - K^s_i(X) [here Sea_Ice_season_class]: numpy.ndarray
            The Total class contribution of of cluster i to the observes seasonal anomaly for season s means between 
            the experimental and control data sets for each cluster, normalized by the number of experimental data
            points in the cluster.
        - WCVC [here Sea_Ice_class_WCVC]: numpy.ndarray
            Within-Cluster Variability Contribution, which quantifies the variability within a cluster due to differences 
            in the mean seasonal values of SIC between the experiment and control, weighted by the number of data points
            in that cluster.
        - FSDC [here Sea_Ice_class_FSDC]: numpy.ndarray
            Frequency-weighted Seasonal Deviation Contribution, incorporating both the frequency of data points in each 
            cluster and their deviations from the seasonal mean squared.
    """
    Season_id = [-1]*4
    Season_id[0], Season_id[1], Season_id[2], Season_id[3]= seasonal_id(Month[:-8])   # datapoints ids for different season (Winter[0], Spring[1], Summer[2], Autumn[3])
    # devide the data point based on the occurrecne seasons
    Sea_Ice_seasonal_C = [-1]*4         
    Sea_Ice_seasonal_E = [-1]*4
    for i in range(4):
        Sea_Ice_seasonal_C[i] = Sea_Ice_C[Season_id[i],:,:]
        Sea_Ice_seasonal_E[i] = Sea_Ice_E[Season_id[i],:,:]
    # seasonal mean calculation for the Control run
    Sea_Ice_X_bar_C_s = np.zeros((4,83,83))    # the seasonal mean of SIC for the Control run
    for k in range(4):
        Sea_Ice_X_bar_C_s[k,:,:] = Sea_Ice_seasonal_C[k].mean(axis=0)
    # The total number of data points in each season for each simulation
    n_s = [Season_id[0].shape[0],Season_id[1].shape[0],Season_id[2].shape[0],Season_id[3].shape[0]] # is the number of data points belonging to season s in the 30-year simulation period
    # seperating the seasonal data into the different clusters (basically devide the data into differenct class and different seasons)
    Sea_Ice_season_class_C = [-1]*4   #Sea_Ice_season_class_C[seasin id][class id][number of daily data in the detrmined class and season, projection x, projection y]
    Sea_Ice_season_class_E = [-1]*4
    Class_Month_C = [-1]*4
    Class_Month_E = [-1]*4
    no_data_C = []
    no_data_E = []
    for i in range(4):
        Class_Month_C[i] = predict_label_C[Season_id[i]]
        Class_Month_E[i] = predict_label_E[Season_id[i]]
        Sea_Ice_season_class_C [i] = [-1]*6
        Sea_Ice_season_class_E [i] = [-1]*6
        for j in range(6):
            class_id_data_C = np.where(Class_Month_C[i]==j)[0]
            class_id_data_E = np.where(Class_Month_E[i]==j)[0]
            if class_id_data_C.shape[0]>0:
                Sea_Ice_season_class_C [i][j] = Sea_Ice_seasonal_C[i][class_id_data_C,:,:]
            else:
                Sea_Ice_season_class_C [i][j] = np.zeros((1,83,83))
                no_data_C.append([i,j])
            if class_id_data_E.shape[0]>0:
                Sea_Ice_season_class_E [i][j] = Sea_Ice_seasonal_E[i][class_id_data_E,:,:]
            else:
                Sea_Ice_season_class_E [i][j] = np.zeros((1,83,83))
                no_data_E.append([i,j])
    N_C_i_s = np.zeros((4,6))                                # The numbers of data points for season s that have been classified as cluster i in the Control run
    N_E_i_s = np.zeros((4,6))                                # The numbers of data points for season s that have been classified as cluster i in the Experiment run
    Sea_Ice_X_bar_C_i_s = np.zeros(( 4, 6, 83, 83))          # Mean field of SIC for the data points in season s and categorized as claster i in the Control run
    Sea_Ice_X_bar_E_i_s = np.zeros(( 4, 6, 83, 83))          # Mean field of SIC for the data points in season s and categorized as claster i in the Experiment run
    for i in range(4):
        for j in range(6):
            N_E_i_s[i,j] = Sea_Ice_season_class_E[i][j].shape[0]
            N_C_i_s[i,j] = Sea_Ice_season_class_C[i][j].shape[0]
            Sea_Ice_X_bar_C_i_s[i,j,:,:] = Sea_Ice_season_class_C[i][j].mean(axis=0)
            Sea_Ice_X_bar_E_i_s[i,j,:,:] = Sea_Ice_season_class_E[i][j].mean(axis=0)
    for i in range(len(no_data_C)):
        N_C_i_s[no_data_C[i][0],no_data_C[i][1]] = 0
    for i in range(len(no_data_E)):
        N_E_i_s[no_data_E[i][0],no_data_E[i][1]] = 0
    n_i_s = N_E_i_s - N_C_i_s   # change in the seasonal occurrence frequency of a cluster in the Experiment run compared to the Control run 
    Sea_Ice_class_WCVC = np.zeros(( 4, 6, 83, 83))    # The Within-Cluster Variability Contribution (WCVC) omponents of the class contribution
    Sea_Ice_class_FSDC = np.zeros(( 4, 6, 83, 83))    # The Frequency-weighted Seasonal Deviation Contribution (FSDC) components of the class contribution
    for i in range(4):
        for j in range(6):
            Sea_Ice_class_WCVC[i,j,:,:] = ( N_C_i_s[i,j]  * ( Sea_Ice_X_bar_E_i_s[i,j,:,:] - Sea_Ice_X_bar_C_i_s[i,j,:,:] ) ) / n_s[i]
            Sea_Ice_class_FSDC[i,j,:,:] = ( n_i_s[i,j] * ( Sea_Ice_X_bar_E_i_s[i,j,:,:] -  Sea_Ice_X_bar_C_s[i]  ) ) / n_s[i]
    Sea_Ice_season_class = Sea_Ice_class_WCVC + Sea_Ice_class_FSDC   # Total class contribution to the anomaly
    return Sea_Ice_season_class, Sea_Ice_class_WCVC, Sea_Ice_class_FSDC



def ploting_the_class_contribution(Sea_Ice_season_class, Sea_Ice_class_WCVC, Sea_Ice_class_FSDC, season, class_id):
    """
    Plot the total class contribution (color shading) and its components with arrows for sea ice concentration.
    The horizontal component of the arrows corresponds to the normalized value of WCVC, and the vertical component
    corresponds to the normalized value of FSDC.
    #
    Parameters:
    sea_ice_season_class : numpy.ndarray
        An array containing the total class contribution.
    sea_ice_class_WCVC : numpy.ndarray
        An array containing the WCVC component of the class contribution.
    sea_ice_class_FSDC : numpy.ndarray
        An array containing the FSDC component of the class contribution.
    season : str
        The name of the season to plot. Must be one of ['Winter', 'Spring', 'Summer', 'Fall'].
    class_id : int
        An integer determining the cluster id. Must be in the range 1-6.
    Outputs:
    A plot showing the class contribution (color shading) to the seasonal anomaly for the specified season,
    with arrows indicating the relative roles of WCVC and FSDC.
    """
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    # clreating the projection
    projection = ccrs.LambertAzimuthalEqualArea(central_latitude=90, central_longitude=0)
    # transfer lat an lon to projection axis
    xy = projection.transform_points(ccrs.PlateCarree(), grid_lon, grid_lat)
    x = xy[:,:,0]
    y = xy[:,:,1]
    Season_name = ['Winter','Spring','Summer','Fall']
    i = Season_name.index(season)
    # calculating max/min for the class contributions in seasons
    Max_SI_cont = np.zeros((4))
    for i in range(4):
        A = Sea_Ice_season_class[i,:,:,:].max()
        B = Sea_Ice_season_class[i,:,:,:].min()
        Max_SI_cont[i] = max([ A, (-1)*B])
        del A
        del B
    # sparcing the WCVC and FSDC as well as normalizing them (for the vectors in the plot)
    def sparcing_2d(dd):
        dd_new = np.zeros(( 41, 41))
        for i1 in range(41):
            for i2 in range(41):
                dd_new[i1, i2] = dd[2*(i1)+1:2*(i1)+3, 2*(i2)+1:2*(i2)+3].mean()
        return dd_new
    def sparcing_4d(dd):
        dd_new = np.zeros((4, 6, 41, 41))
        for i1 in range(4):
            for i2 in range(6):
                for i3 in range(41):
                    for i4 in range(41):
                        dd_new[i1, i2, i3, i4] = dd[i1, i2, 2*(i3)+1:2*(i3)+3, 2*(i4)+1:2*(i4)+3].mean()
        return dd_new
    xx = sparcing_2d(x)
    yy = sparcing_2d(y)
    Sea_Ice_class_WCVC = sparcing_4d(Sea_Ice_class_WCVC)
    Sea_Ice_class_FSDC = sparcing_4d(Sea_Ice_class_FSDC)
    # calculating the normalized WCVC and FSDC
    Sea_Ice_magnitude = np.sqrt(Sea_Ice_class_WCVC**2 + Sea_Ice_class_FSDC**2)
    Sea_Ice_class_WCVC = (Sea_Ice_class_WCVC/ Sea_Ice_magnitude)
    Sea_Ice_class_FSDC = (Sea_Ice_class_FSDC/ Sea_Ice_magnitude)
    Class_names = ['C1','C2','C3','C4','C5','C6']
    j = class_id - 1 
    # ploting the class contributions and their WCVC and 
    fig, ax = plt.subplots(subplot_kw={'projection': projection})
    im = ax.imshow(Sea_Ice_season_class[i,j,:,:], cmap='twilight_shifted',vmin = (-1)*Max_SI_cont[i], vmax = Max_SI_cont[i],  extent=(x.min(), x.max(), y.min(), y.max()), origin='upper')
    ax.quiver(xx, yy, Sea_Ice_class_WCVC[i,j,:,:], Sea_Ice_class_FSDC[i,j,:,:], pivot='middle', scale=40, alpha=0.6)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.set_extent([0, 360, 40, 90], ccrs.PlateCarree())
    ax.set_title("SI_Class_contribution "+Season_name[i]+" "+Class_names[j])
    ax.set_boundary(circle, transform=ax.transAxes)
    plt.colorbar(im, ax=ax, orientation='vertical', label='SIC')
    plt.show()
    plt.close()


#-----------------------------------------------------------------------------------------------------

# loading the Sea Ice Concentration (SIC) map for the whole data point in the Control (C) and Experiment (E) runs
# they have been transferred to the Lambert azimuthal equal-area projection
loaded = np.load('Sea_Ice_Concentartion_C.npz')

Sea_Ice_C = loaded['Sea_Ice_C']      # SIC for data points (daily) in the Control run, shape (10949 [days from the first day of the simulation], 83 [projection x], 83 [projection y])
Month = loaded['Month']              # month of the year for each datapoint, shape (10949 [data point id])
Day = loaded['Day']                  # day of the year for each datapoint
Year = loaded['Year']                # the simulation year for each datapoint

# Projection latitude and longitude for each pixel
grid_lat = loaded['grid_lat']
grid_lon = loaded['grid_lon']

Mask = loaded['Mask']               # mask for data southward of latitude 30 deg North



loaded = np.load('Sea_Ice_Concentartion_E.npz')
Sea_Ice_E = loaded['Sea_Ice_E']      # SIC for data points (daily) in the Experiment run, shape (10949 [days from the first day of the simulation], 83 [projection x], 83 [projection y])


# loading the clastering labels
# for each daily datapoint the class labels determine whcih cluster the data point is assigned to
loaded = np.load('MCAE_clusterig_labels.npz')

predict_label_C = loaded['predict_label_C']   # labels for data ponts in C [range form 0 to 5]
predict_label_E = loaded['predict_label_E']   # labels for data ponts in E [range form 0 to 5]



# creat seasonal anomaly (E-C), and detrmine where they are statistically significant
Sea_Ice_seasonal_Anomaly, Sea_Ice_Significant_HW = Seasonal_Anomaly(Sea_Ice_C, Sea_Ice_E, Month)

# ploting the seasonal anomalis
ploting_seasonal_anomaly(Sea_Ice_seasonal_Anomaly, Sea_Ice_Significant_HW, 'Winter')
ploting_seasonal_anomaly(Sea_Ice_seasonal_Anomaly, Sea_Ice_Significant_HW, 'Spring')
ploting_seasonal_anomaly(Sea_Ice_seasonal_Anomaly, Sea_Ice_Significant_HW, 'Summer')
ploting_seasonal_anomaly(Sea_Ice_seasonal_Anomaly, Sea_Ice_Significant_HW, 'Fall')

# calculate the class contribution and its components (WCVC, FSDC)
Sea_Ice_season_class, Sea_Ice_class_WCVC, Sea_Ice_class_FSDC = Class_contribution(Sea_Ice_C, Sea_Ice_E, Month, predict_label_C, predict_label_E)

# ploting the class contribution, with normalized WCVC and FSDC for class 3 (C3), and season Fall (plot 14 of the paper)

ploting_the_class_contribution(Sea_Ice_season_class, Sea_Ice_class_WCVC, Sea_Ice_class_FSDC, 'Fall', 3)






##-------------------------------------------------------------------------------------------------------
# recalculate the seasonal anomaly using the class contributions to test whether
# the sum of class contribution will be equal to the seasonal anomaly

Sea_Ice_anomaly_Class_contribution = np.zeros((4,83,83))
for i in range(4):
    count = 0
    A = np.zeros((83,83))
    for j in range(6):
        A = A + Sea_Ice_season_class [i,j,:,:]
    Sea_Ice_anomaly_Class_contribution[i] = A
    del A



##test wether it is same as the anomaly that we calculated directly
test_difference = Sea_Ice_seasonal_Anomaly - Sea_Ice_anomaly_Class_contribution 
test_difference.max()   # should return very smal values  
test_difference.min()  # should return very smal values


####-------------------------------------------------------------------------------------------------------







