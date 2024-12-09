import pandas as pd
import xarray as xr
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from dateutil.relativedelta import relativedelta
from numpy import trapz
from scipy import integrate


def ord_buoy_data(buoy_id):
        if buoy_id in ['42057','42058','42059','42060']:
                path=f'/home/fayalacruz/runs/reg_storm_new_ctrl/info/plots/{buoy_id}/{buoy_id}h2020.txt'
                df_boya=pd.read_csv(path,header=0, delim_whitespace=True, dtype=object,
                                    na_values=['99.00',999,9999,99.,999.,9999.])[1:]
                fechas=df_boya[df_boya.columns[:5]].astype(float)
                fechas=fechas.rename(columns={"#YY": "year", "MM":"month","DD":'day','hh':'hour','mm':'minute'})
        else:
                path=f'/home/fayalacruz/runs/ivan2004_def/info/plots/{buoy_id}/{buoy_id}h2004.txt'
                df_boya=pd.read_csv(path,header=0, delim_whitespace=True, dtype=object,
                                    na_values=['99.00',999,9999,99.,999.,9999.])[1:]
                fechas=df_boya[df_boya.columns[:4]].astype(float)
                fechas=fechas.rename(columns={"YYYY": "year", "MM":"month","DD":'day','hh':'hour'})
        
        fechas_with_nans=pd.to_datetime(fechas)
        hs_with_nans=df_boya['WVHT'].astype(np.float)
        dir_with_nans=df_boya['MWD'].astype(np.float)
        tp_with_nans=df_boya['DPD'].astype(np.float)
        if buoy_id in ['42057','42058','42059','42060']:
                wndir_with_nans=df_boya['WDIR'].astype(np.float)
        else:
                wndir_with_nans=df_boya['WD'].astype(np.float)
        wspd_with_nans=df_boya['WSPD'].astype(np.float)
        hs=hs_with_nans[np.logical_not(np.isnan(hs_with_nans))]
        dir=dir_with_nans[np.logical_not(np.isnan(hs_with_nans))]
        tp=tp_with_nans[np.logical_not(np.isnan(hs_with_nans))]
        wndir=wndir_with_nans[np.logical_not(np.isnan(hs_with_nans))]
        wspd=wspd_with_nans[np.logical_not(np.isnan(hs_with_nans))]
        fechas=fechas_with_nans[np.logical_not(np.isnan(hs_with_nans))]
        df_boya=pd.DataFrame(data={'hs':hs.values,'dir':dir.values,'tp':tp.values,'wndir':wndir.values,'wspd':wspd.values},index=fechas)
        return df_boya

def compute_azimuth(angle):
        if angle<=90:
                result = 90 - angle
        elif (angle>90) & (angle<=180):
                result = 270 + (180-angle)
        elif (angle>180) & (angle <=270):
                result = 180 + (270-angle)
        else:
                result = 90 + (360-angle)
        return result

def read_1d_spec_buoy(buoy_id):
        path=f'/home/fayalacruz/runs/reg_storm_new_ctrl/info/plots/{buoy_id}/{buoy_id}w2020.txt'
        df_boya=pd.read_csv(path, delim_whitespace=True, parse_dates={ 'date': ['#YY', 'MM', 'DD','hh','mm']})
        df_boya=df_boya.set_index('date')
        df_boya.index = pd.to_datetime(df_boya.index,format='%Y %m %d %H %M')

        freqs = np.array(list(map(lambda x: float(x),df_boya.columns)))
        times = df_boya.index
        specdens = df_boya.values

        return times,freqs,specdens


def read_2d_spec_buoy(buoy_id):
        url_sufixs = {'sw':'w','r1':'j','r2':'k','a1':'d','a2':'i'}
        dfs={}
        for label,suf in url_sufixs.items():
                path=f'/home/fayalacruz/runs/reg_storm_new_ctrl/info/plots/{buoy_id}/{buoy_id}{suf}2020.txt'
                df_boya=pd.read_csv(path, delim_whitespace=True, parse_dates={ 'date': ['#YY', 'MM', 'DD','hh','mm']})
                df_boya=df_boya.set_index('date')
                df_boya.index = pd.to_datetime(df_boya.index,format='%Y %m %d %H %M')
                dfs[label]=df_boya
                if suf == 'w':
                        freqs = np.array(list(map(lambda x: float(x),df_boya.columns)))

        times = dfs['sw'].index
        spshape = (len(times), len(freqs), 1)
        specdens = dfs['sw'].values.reshape(spshape)
        swdir1=dfs['a1'].values.reshape(spshape)
        swdir2=dfs['a2'].values.reshape(spshape)
        swr1=dfs['r1'].values.reshape(spshape)
        swr2=dfs['r2'].values.reshape(spshape)


        dirs=np.arange(0, 360, 10)

        dirmat = dirs.reshape((1, 1, -1))
        IPI = 1/np.pi
        D2R = np.pi/180
        D_fd = (
                IPI * (0.5 + swr1 * np.cos(D2R * (dirmat - swdir1)) + swr2 * np.cos(2 * D2R * (dirmat - swdir2)))* D2R
                )
        S = specdens * D_fd
        return times,freqs,dirs,S

def read_dir_spread_buoys(buoy_id):
        url_sufixs = {'sw':'w','r1':'j','alpha1':'d'}
        dfs={}
        for label,suf in url_sufixs.items():
                path=f'/home/fayalacruz/runs/reg_storm_new_ctrl/info/plots/{buoy_id}/{buoy_id}{suf}2020.txt'
                df_boya=pd.read_csv(path, delim_whitespace=True, parse_dates={ 'date': ['#YY', 'MM', 'DD','hh','mm']})
                df_boya=df_boya.set_index('date')
                df_boya.index = pd.to_datetime(df_boya.index,format='%Y %m %d %H %M')
                dfs[label]=df_boya
                if suf == 'w':
                        freqs = np.array(list(map(lambda x: float(x),df_boya.columns)))
        times = dfs['sw'].index
        spshape = (len(times), len(freqs), 1)
        specdens = dfs['sw'].values.reshape(spshape)
        theta0=dfs['alpha1'].values.reshape(spshape)
        m1=dfs['r1'].values.reshape(spshape)

        a1=m1*np.cos(np.radians(theta0))
        b1=m1*np.sin(np.radians(theta0))

        # print(a1[0,:,0])

        # print(m1[4,0,0])


        # a1=sw_m1*np.cos(sw_theta0)
        # b1=sw_m1*np.sin(sw_theta0)

        dir_spread=np.ones(len(times))

        for idx,date in enumerate(times):
                Ebar=trapz(specdens[idx,:,0],x=freqs)
                Ebar_1=integrate.simpson(specdens[idx,:,0],x=freqs)
                a1_bar=trapz((a1[idx,:,0]*specdens[idx,:,0]),x=freqs)/Ebar
                b1_bar=trapz((b1[idx,:,0]*specdens[idx,:,0]),x=freqs)/Ebar

                dir_spread[idx]=np.sqrt(2*(1-np.sqrt((a1_bar**2)+(b1_bar**2))))

                if idx==4:
                        print(Ebar)
                        print(Ebar_1)

                #         print(a1_bar,b1_bar)
                #         print(np.sqrt((a1_bar**2)+(b1_bar**2)))



        return times,dir_spread
 

def read_data_int_stations(path):
        ds_points=xr.load_dataset(path)
        data_stations={}
        time_points_ww3=pd.to_datetime(ds_points.time.data)
        for ind,station in enumerate(ds_points.station_name.data):
                pre_station_id=b''.join(station)
                station_id=pre_station_id.decode('utf-8')
                data_stations[station_id]=pd.DataFrame(data={'hs':ds_points.hs.values[:,ind],'dirp':ds_points.th1p.values[:,ind],
                                                             'tp':1/(ds_points.fp.values[:,ind]),'fp':ds_points.fp.values[:,ind],
                                                             'dirm':ds_points.th1m.values[:,ind],'sprp':ds_points.sth1p.values[:,ind],
                                                             'sprm':ds_points.sth1m.values[:,ind]},index=time_points_ww3)
        return data_stations

def read_data_extra_stations(path):
        ds_points=xr.load_dataset(path)
        data_stations={}
        time_points_ww3=pd.to_datetime(ds_points.time.data)
        for ind,station in enumerate(ds_points.station_name.data):
                pre_station_id=b''.join(station)
                station_id=pre_station_id.decode('utf-8')
                data_stations[station_id]=pd.DataFrame(data={'dpt':ds_points.dpt.values[:,ind],'cur':ds_points.cur.values[:,ind],'wnd':ds_points.wnd.values[:,ind],
                                                             'wnddir':ds_points.wnddir.values[:,ind]},index=time_points_ww3)
        return data_stations

def read_data_spec_1d_stations(path):
        ds_points=xr.load_dataset(path)
        time_points_ww3=pd.to_datetime(ds_points.time.data)
        data_stations={}
        for ind,station in enumerate(ds_points.station_name.data):
                pre_station_id=b''.join(station)
                station_id=pre_station_id.decode('utf-8')
                data_stations[station_id]=ds_points.f.values[:,ind,:]
        return time_points_ww3,ds_points.frequency.values[:],data_stations


def read_data_spec_stations(path):
        ds_points=xr.load_dataset(path)
        time_points_ww3=pd.to_datetime(ds_points.time.data)
        data_stations={}
        for ind,station in enumerate(ds_points.station_name.data):
                pre_station_id=b''.join(station)
                station_id=pre_station_id.decode('utf-8')
                data_stations[station_id]=ds_points.efth.values[:,ind,:,:]
        return time_points_ww3,ds_points.frequency.values[:],np.radians(ds_points.direction.values[:]),data_stations

def read_data_src_stations(path,dim,term):
        ds_points=xr.load_dataset(path)
        time_points_ww3=pd.to_datetime(ds_points.time.data)
        data_src_stations={}
        for ind,station in enumerate(ds_points.station_name.data):
                pre_station_id=b''.join(station)
                station_id=pre_station_id.decode('utf-8')
                if dim =='2d':
                        data_src_stations[station_id]=ds_points[term].values[:,ind,:,:]
                else:
                        data_src_stations[station_id]=ds_points[term].values[:,ind,:]
        if dim =='2d':                
                return time_points_ww3,ds_points.frequency.values[:],np.radians(ds_points.direction.values[:]),data_src_stations
        elif dim=='1d':
                return time_points_ww3,ds_points.frequency.values[:],data_src_stations

def read_era5_buoys(path,lon_buoy,lat_buoy):
        era5_dtst=xr.load_dataset(path)
        lon=era5_dtst.longitude.values[:]
        lat=era5_dtst.latitude.values[:]
        time_points_era5=pd.to_datetime(era5_dtst.time.data)
        final_serie=[]
        final_serie_u=[]
        final_serie_v=[]

        for i in range(len(era5_dtst.time)):
                wind_speed=np.sqrt(((era5_dtst.u10.values[i,:,:])**2)+((era5_dtst.v10.values[i,:,:])**2))
                u_component = era5_dtst.u10.values[i,:,:]
                v_component = era5_dtst.v10.values[i,:,:]
                longitude,latitude=np.meshgrid(lon,lat)
                interp = RegularGridInterpolator((lon, lat), wind_speed.T)
                interp_u= RegularGridInterpolator((lon, lat), u_component.T)
                interp_v= RegularGridInterpolator((lon, lat), v_component.T)
                pts = np.array([lon_buoy,lat_buoy])
                final_serie.append(round(interp(pts).tolist()[0],3))
                final_serie_u.append(round(interp_u(pts).tolist()[0],3))
                final_serie_v.append(round(interp_v(pts).tolist()[0],3))
        a=pd.Series(final_serie,index=time_points_era5)
        u=pd.Series(final_serie_u,index=time_points_era5)
        v=pd.Series(final_serie_v,index=time_points_era5)
        return a,u,v

def read_spatial_data(path):
        ds_spatial=xr.load_dataset(path)
        lon_spatial=ds_spatial.longitude.values
        lat_spatial=ds_spatial.latitude.values
        time_points_ww3=pd.to_datetime(ds_spatial.time.data)
        return time_points_ww3,lon_spatial,lat_spatial,ds_spatial.utaw.values,ds_spatial.vtaw.values

def read_hindcast(path):
        ds=xr.open_dataset(path,engine='netcdf4')
        hs=ds.hs.values[:]
        lon=ds.longitude.values[:]
        lat=ds.latitude.values[:]

        time=pd.to_datetime(ds.time.data)
        ds.close()

        locs_buoys = {'42057':(16.908,-81.422),'42058':(14.394,-74.816),'42059':(15.300,-67.483),
                '42060':(16.434,-63.329)}

        points = list((locs_buoys.values()))
        modified_points = np.array([[x[1]+360, x[0]] for x in points])
        lon_points,lat_points=modified_points[:,0],modified_points[:,1]

        pts = np.array([lon_points,lat_points])
        hs_series_buoys=np.empty((4,len(time)))

        for idx,date in enumerate(time):
                hs_map=hs[idx,:,:]

                interpolated_points=RegularGridInterpolator((lon,lat),hs_map.T)
                results=interpolated_points(pts.T)
                hs_series_buoys[:,idx]=results

        hs_hindcast_buoys = {'42057':hs_series_buoys[0,:],'42058':hs_series_buoys[1,:],'42059':hs_series_buoys[2,:],
                '42060':hs_series_buoys[3,:]}
        return time,hs_hindcast_buoys