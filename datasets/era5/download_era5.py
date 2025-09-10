import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-complete',   
    {
        'date'    : '2015-01-01/to/2015-12-31', 
        #'levelist': '1/10/100/137',
        'levtype' : 'sfc',
        'param'   : '10/157/164/167/228',        # windSpeed / humidity / totalCloudCover / 2MetreTemperature  / totalPrecipitation
        'stream'  : 'oper',
        'time'    : '00/to/23/by/3',
        'type'    : 'an',
        'area'    : '66.13/-148.47/64.13/-146.47',   # N, W, S, E, Poker Flat
        'grid'    : '1.0/1.0',
        'format'  : 'netcdf',
    },
    'ERA5-2015.nc'  
)
