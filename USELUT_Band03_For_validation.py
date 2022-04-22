#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from Py6S import *
import time as T
from joblib import Parallel, delayed
from scipy.interpolate import griddata,interpn,RegularGridInterpolator
import math
import os
import datetime
import cv2
from ftplib import FTP
import xarray as xr
import multiprocessing
import paramiko
from scp import SCPClient
import ephem
from decimal import Decimal, ROUND_HALF_UP


# In[ ]:





# In[1]:


def FindPixel(x,y):
    lat = int((50 - x)/0.005)
    lon = int((y - 120)/0.005)
    return lat,lon

def dms2deg(dms):
    h = dms[0]
    m = dms[1]
    s = dms[2]
    deg = Decimal(str(h + (m / 60) + (s / 3600))).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
    return deg


# In[7]:


# Before using this, you need to set some input path.
# set your account at def download_AOT
# Set input Viewer zenith angle data path
# Set input Solar zenith angle data path
# Set input LUT path
# Set input Atmosphere data path
# Set date and target(save path)
# Atmospheric correction block is used for calculate SR
# Angle date and Atmosphere data will be upload to server later




#Set date and path
YYYY = '2019'
MM = ['05']
DD = ['07']
MIN = ['20']
HH = ['06']

target ='/media/liwei/Data/AHI_AC_RESULT/'
SZA_path = '/media/liwei/Data/AHI_Angle/Solar_zenith_angle/'
SAZ_path = '/media/liwei/Data/AHI_Angle/Solar_azimuth_angle/'
VZA_path = '/media/liwei/Data/AHI_Angle/Viewer_angle/view_zM_JAPAN_05.dat'
VAZ_path = '/media/liwei/Data/AHI_Angle/Viewer_angle/view_aM_JAPAN_05.dat'
LUT_path = '/media/liwei/Data/LUT/'
ATMOS_path = '/media/liwei/Data/CAMS/'

site_name = 'TKY'
# site_name = ['TKY']


Validation_site = int((50 - 36.1462)/0.005),int((137.4231 - 120)/0.005)


# In[3]:


sza = np.linspace(0,80,17)
vza = np.linspace(0,80,17)
water = np.linspace(0,7,8)
ozone = np.linspace(0.2,0.4,5)
AOT = np.array([0.01,0.05,0.1,0.15,0.2,0.3,0.4,0.6,0.8,1.0,1.5,2.0]) 
raa = np.linspace(0,180,19)




   
class read_H8data:

    def __init__(self, band, band_number):
        self.band = band
        self.band_number = band_number

    def get_path(self, date):
        return '/mnt/nas01G/geo01/H8AHI/download/org/192.168.1.5/gridded/FD/V20190123/' + date[0:6] + '/' +                self.band.upper() + '/'

    def get_filename(self):
        return self.band + "." + self.band_number + ".fld.geoss.bz2"


def file_path(band_num, date):
    return read_H8data.get_path(BAND[band_num - 1], date) + date + "." + read_H8data.get_filename(BAND[band_num - 1])


def Hi8_band():
    b01 = read_H8data('vis', '01');
    b02 = read_H8data('vis', '02');
    b03 = read_H8data('ext', '01');
    b04 = read_H8data('vis', '03')
    b05 = read_H8data('sir', '01');
    b06 = read_H8data('sir', '02');
    b07 = read_H8data('tir', '05');
    b08 = read_H8data('tir', '06')
    b09 = read_H8data('tir', '07');
    b10 = read_H8data('tir', '08');
    b11 = read_H8data('tir', '09');
    b12 = read_H8data('tir', '10')
    b13 = read_H8data('tir', '01');
    b14 = read_H8data('tir', '02');
    b15 = read_H8data('tir', '03');
    b16 = read_H8data('tir', '04')
    BAND = [b01, b02, b03, b04, b05, b06, b07, b08, b09, b10, b11, b12, b13, b14, b15, b16]
    return BAND

def download_H8data(date):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname='10.4.200.105', port=22, username='liwei', password='liwei000')
    scp = SCPClient(client.get_transport())
    sftp = client.open_sftp()
    
    try :
        sftp.stat(file_path(3, date))
        
    except FileNotFoundError:
        print("File Not Found")
        pass
    
    else:
        scp.get(file_path(3, date), folder_original+'/')
        os.system('lbzip2 -d {}{}'.format(folder_original+'/',file_path(3, date)[-33:]))
    
def remove_original_file(path):
    os.system('rm -rf {}'.format(path))
    
    
def download_AOT(YYYY,MM,DD,HH,folder):
    ftp_addr = 'ftp.ptree.jaxa.jp'
    f=FTP(ftp_addr)
    f.login('liwei1997_chiba-u.jp','SP+wari8')
    remote_filepath = '/pub/model/ARP/MS/bet/'+YYYY+MM+'/'+DD+'/'
    f.cwd(remote_filepath)
    list=f.nlst()
    bufsize=1024
    for name in list:
        if name[13:17]==HH+'00':
            data=open(folder+'/'+name,'wb')
            filename='RETR '+name
            f.retrbinary(filename,data.write,bufsize)
    f.quit()
    
def mkdir(path):
    folder = os.path.exists(path) 
    if not folder:
        os.makedirs(path)
        
def DN2tbb(dn):         
    LUT=np.loadtxt('/media/liwei/Data/count2tbb_v102/ext.01')
    return LUT[dn,1]


def get_point():
    A=[]
    for i in range(len(vza)):
        for j in range(len(sza)):
            A.append(vza[i])
            A.append(sza[j])
    point=np.array(A).reshape(17*17,2)
    return point

def griddata_inter(X1,X2,X3,a,b,c,d):
    X1_new_inter=[]
    X2_new_inter=[]
    X3_new_inter=[]
    
    X1_inter=X1[a,b,c,:,:,d].reshape(17*17,1)
    X2_inter=X2[a,b,c,:,:,d].reshape(17*17,1)
    X3_inter=X3[a,b,c,:,:,d].reshape(17*17,1)
    
    X1_new = griddata(point, X1_inter, (xi, yi), method='cubic')
    X2_new = griddata(point, X2_inter, (xi, yi), method='cubic')
    X3_new = griddata(point, X3_inter, (xi, yi), method='nearest')
                
    X1_new_inter.append(X1_new)
    X2_new_inter.append(X2_new)
    X3_new_inter.append(X3_new)
    
    del X1_inter,X2_inter,X3_inter,X1_new,X2_new,X3_new  
    return X1_new_inter,X2_new_inter,X3_new_inter


def ATMO_time(HH):
    if int(HH)%3==0:
        return HH
    elif (int(HH)-1)%3==0:
        return str(int(HH)-1).zfill(2)
    elif int(HH)==23:
        return str(21).zfill(2)
    else:
        return str(int(HH)+1).zfill(2)


    

def calc_sunpos(dtime,col,row):
    
    sun = ephem.Sun()
    obs = ephem.Observer()
    obs.date = dtime
    obs.lat = lat_x[row]*math.pi/180.0
    obs.long = lon_y[col]*math.pi/180.0
    sun.compute(obs)
    return np.degrees(sun.az),90.0-np.degrees(sun.alt)    
    
    
def calculate_6s_band3():
    WV_input=WV
    OZ_input=OZ
    AOT550_input=AOT550
    RAA_input=RAA
    SZA_input=Solar_zM
    view_zM_input=view_zM
    xi=np.array([WV_input,OZ_input,AOT550_input,RAA_input,SZA_input,view_zM_input])
    xi=xi.T
    xa=fn1(xi)
    xb=fn2(xi)
    xc=fn3(xi)
    y = xa*data-xb
    SR=y/(1+xc*y)
    return SR


# Viewer_zenith_angle
with open(VZA_path,'rb') as fp:
    view_zM = np.fromstring(fp.read()).reshape(6000,6000)[Validation_site]
with open(VAZ_path,'rb') as fp:
    view_aM = np.fromstring(fp.read()).reshape(6000,6000)[Validation_site]    

# read LUT
outfile1 = LUT_path + '01_band3.csv'
outfile2 = LUT_path + '02_band3.csv'
outfile3 = LUT_path + '03_band3.csv'
X1 = np.loadtxt(outfile1,delimiter=",")
X2 = np.loadtxt(outfile2,delimiter=",")
X3 = np.loadtxt(outfile3,delimiter=",")
    
    
# reshape LUT
X1_reshape=X1.reshape(8,5,12,17,17,19)
X2_reshape=X2.reshape(8,5,12,17,17,19)
X3_reshape=X3.reshape(8,5,12,17,17,19)
del X1,X2,X3

# SZA,SAZ,VZA interpolation
point=get_point()
xi,yi=np.ogrid[0:80:161j, 0:80:161j]
output = Parallel(n_jobs=-1)(delayed(griddata_inter)(X1_reshape,X2_reshape,X3_reshape,a,b,c,d)                             for a in range(len(water))                              for b in range(len(ozone))                              for c in range(len(AOT))                              for d in range(len(raa)))

X1_new_inter_reshape=np.array(output)[:,0].reshape(8,5,12,19,161,161)
X2_new_inter_reshape=np.array(output)[:,1].reshape(8,5,12,19,161,161)
X3_new_inter_reshape=np.array(output)[:,2].reshape(8,5,12,19,161,161)

del X1_reshape,X2_reshape,X3_reshape,output

sza_new = np.linspace(0,80,161)
vza_new = np.linspace(0,80,161)

fn1 = RegularGridInterpolator((water,ozone,AOT,raa,sza_new,vza_new),X1_new_inter_reshape,bounds_error=False,fill_value=np.nan)  
fn2 = RegularGridInterpolator((water,ozone,AOT,raa,sza_new,vza_new),X2_new_inter_reshape,bounds_error=False,fill_value=np.nan)  
fn3 = RegularGridInterpolator((water,ozone,AOT,raa,sza_new,vza_new),X3_new_inter_reshape,bounds_error=False,fill_value=np.nan)

lat_x=np.linspace(50,20,6000)
lon_y=np.linspace(120,150,6000)

BAND=Hi8_band()


# In[8]:


# Atmospheric correction
# Loop in input date
for k in range(len(MM)):
    for m in range(len(DD)):
        for i in range(len(HH)):
            for j in range(len(MIN)):
                start_time=T.time()
                date=YYYY+MM[k]+DD[m]+HH[i]+MIN[j]  
                time=date[-4:]       
                print("start processing {}".format(date))
                
                
                # make dir
                folder_original = target+date+'_original'
                folder_AC = target+date+'_AC'
                mkdir(folder_original)
                mkdir(folder_AC)
            

                
                # Input Atmosphere data
                ds_oz_wv = xr.open_dataset(ATMOS_path + YYYY + MM[k] + ATMO_time(HH[i]) + '.nc')
                oz=ds_oz_wv['gtco3'][int(DD[m])-1,:,:]
                OZ=oz.interp(longitude=lon_y,latitude=lat_x,method="nearest")
                OZ=OZ.values[Validation_site]
                wv=ds_oz_wv['tcwv'][int(DD[m])-1,:,:]
                WV=wv.interp(longitude=lon_y,latitude=lat_x,method="nearest")
                WV=WV.values[Validation_site]

                
                # download AOT and read AOT
                download_AOT(YYYY,MM[k],DD[m],HH[i],folder_original)
                ATMOS_data_s=T.time()
                ds = xr.open_dataset(folder_original+'/H08_'+YYYY+MM[k]+DD[m]+'_'+HH[i]+'00_MSARPbet_ANL.00960_00480.nc')
                aot550=ds['od550aer']
                AOT550=aot550.interp(lon=lon_y,lat=lat_x,method="nearest")
                AOT550=AOT550.values[Validation_site]
                
                
                del oz,wv,aot550,ds_oz_wv,ds
                print("OZ,AOT,WV finish")
                
                 
                # Download Himawari8 data from server
                download_H8data(date)
                print("data download finsih")
                
                
                
                # If file exist do atmosperic correction , else pass
                if  os.path.exists(folder_original+'/'+date+'.ext.01.fld.geoss'):
                    
                    # read himawari8 file
                    with open(folder_original+'/'+date+'.ext.01.fld.geoss','rb') as fp:
                        data = np.fromstring(fp.read(),dtype='>u2').reshape(24000,24000)
                        data=data[2000:8000,7000:13000]
                        data = data[Validation_site]
                        data=DN2tbb(data)
                        data=data/100
                        

                    print("data reading finish")
                                        
                    
                    # Solar angle                    
                    dtime = datetime.datetime(int(YYYY),int(MM[k]),int(DD[m]),int(HH[i]),int(MIN[j]))
                    Solar_aM,Solar_zM = calc_sunpos(dtime,3000,3000)
                        
                    
                    
                    # Calculate RAA
                    RAA = abs(Solar_aM-view_aM)
                    if RAA > 180:
                        RAA = 360 - RAA
                    print("SZA,SAZ finish")
                    
                    
                    # Atmosphere data Unit conversion
                    WV = WV/10
                    OZ = OZ*46.6975764
                    
                    
                    # Processing water vapor and ozone max and min
                    
                    if OZ >= max(ozone):
                        OZ = max(ozone) - (1/10000)
                    if OZ <= min(ozone):
                        OZ = min(ozone) + (1/10000)
                        
                    if WV >= max(water):
                        WV = max(water) - (1/10000)
                    if OZ <= min(water):
                        OZ = min(water) + (1/10000)
                        
                    if AOT550 >= max(AOT):
                        AOT550 = max(AOT) - (1/10000)
                    if AOT550 <= min(AOT):
                        AOT550 = min(AOT) + (1/10000)
                    

                    
                    # Using LUT
                    SR = calculate_6s_band3()

                
                    # Save file and remove download input data
                    #SR=np.array(SR).reshape(6000,6000)
                    SR_file=open(folder_AC+'/'+date+'_b03_'+site_name+'.dat','wb')
                    SR.astype('f4').tofile(SR_file)
                    SR_file.close()
                    remove_original_file(folder_original)
                    end_time=T.time()
                    TIME=end_time-start_time
                    print('time: {:.1f} secs, {:.1f} mins,{:.1f} hours'.format(TIME,TIME/60,TIME/3600))
                    print("delete file finish")
                else:
                    print("file no exists")
                    pass


# In[ ]:




