#!/usr/bin/env python
# coding: utf-8

# In[1]:


import multiprocessing
import datetime
import ephem
import math
import numpy as np
import time as T
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap


YYYY='2016'
MM=['01']
DD=['01']
HH=['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24',]
MIN=['00','10','20','30','40','50']


def do_solar(para):
    return calc_sunpos(para[0],para[1],para[2])
def calc_sunpos(dtime,col,row):
    sun = ephem.Sun()
    obs = ephem.Observer()
    obs.date = dtime
    obs.lat = latgridrad[row]
    obs.long = longridrad[col]
    sun.compute(obs)
    return np.degrees(sun.az),90.0-np.degrees(sun.alt)

for k in range(len(MM)):
    for m in range(len(DD)):
        for i in range(len(HH)):
            for j in range(len(MIN)):
                date=YYYY+MM[k]+DD[m]+HH[i]+MIN[j] 
                Solar_zM = np.zeros((3000,3000))
                Solar_aM = np.zeros((3000,3000))
                
                dtime = datetime.datetime(int(YYYY),int(MM[k]),int(DD[m]),int(HH[i]),int(MIN[j]))
                dellon = 0.01
                dellat = 0.01

                longrid = np.linspace(120+dellon/2, 150-dellon/2, 3000) #degrees
                latgrid = np.linspace(20-dellat/2, 50+dellat/2, 3000) #degrees
                latgridrad = latgrid*math.pi/180.0 #radians
                longridrad = longrid*math.pi/180.0 #radians
                if __name__ == '__main__':
    
                    para_solar= []
                    for q in range(3000):
                        for w in range(3000):
                            t=(dtime,w,q)
                            para_solar.append(t)
                    start_time=T.time()
                    p=multiprocessing.Pool(32)
                    b=p.map(do_solar,para_solar)
                    p.close()
                    p.join()
                    Solar_aM = [i[0] for i in b]
                    Solar_zM = [i[1] for i in b]
                    end_time=T.time()
                    #cost=end_time-start_time
                    #print(cost)
                    
                    Solar_aM=np.array(Solar_aM).reshape(3000,3000)
                    Solar_zM=np.array(Solar_zM).reshape(3000,3000)
                    
                    plt.title('Himawari-8 Solar azimuth angle \n {dt}'.format(dt=date),fontsize='large')
                    plt.imshow(Solar_aM,cmap='terrain_r',origin='upper',vmax=360,vmin=0)
                    v=np.linspace(0,360,13)
                    plt.colorbar(ticks=v)
                    plt.savefig('Solar_azimuth_angle_pic/solar_aM_{d}.jpg'.format(d=date),dpi=6000)
                    plt.clf()
                    plt.close()
                    
                    
                    plt.title('Himawari-8 Solar zenith angle \n {dt}'.format(dt=date),fontsize='large')
                    plt.imshow(Solar_zM,cmap='terrain_r',origin='upper',vmax=90,vmin=0)
                    v=np.linspace(0,90,10)
                    cb=plt.colorbar(ticks=v)
                    plt.savefig('Solar_zenith_angle_pic/solar_zM_{d}.jpg'.format(d=date),dpi=6000)
                    plt.clf()
                    plt.close()
                    
                    datfile_solar_aM_test=open('Solar_azimuth_angle/solar_aM_{d}.dat'.format(d=date),'wb')
                    Solar_aM.tofile(datfile_solar_aM_test)
                    datfile_solar_aM_test.close()

                    datfile_solar_zM_test=open('Solar_zenith_angle/solar_zM_{d}.dat'.format(d=date),'wb')
                    Solar_zM.tofile(datfile_solar_zM_test)
                    datfile_solar_zM_test.close()


# In[ ]:




