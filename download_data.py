"""
This is a script to download the raw dataset used. Please note,
additional setup is required to use this script. See the following
link (https://cds.climate.copernicus.eu/api-how-to) and follow the
specified instructions for the OS being utilized. Run the function
main() to start the download.
"""
import threading
import time as tm
import cdsapi

c = cdsapi.Client()

FEATURES_LIST_1 = ['1', [
    '10m_u_component_of_wind', '2m_dewpoint_temperature', 'evaporation',
    'forecast_albedo', 'total_cloud_cover',
    'volumetric_soil_water_layer_1', 'soil_temperature_level_1',
]]

FEATURES_LIST_2 = ['2', ['10m_v_component_of_wind',
                         '2m_temperature',
                         'surface_pressure',
                         'total_precipitation',
                         'total_column_water_vapour',
                         'total_sky_direct_solar_radiation_at_surface',
                         'toa_incident_solar_radiation']]

TIME = [
    '00:00', '01:00', '02:00',
    '03:00', '04:00', '05:00',
    '06:00', '07:00', '08:00',
    '09:00', '10:00', '11:00',
    '12:00', '13:00', '14:00',
    '15:00', '16:00', '17:00',
    '18:00', '19:00', '20:00',
    '21:00', '22:00', '23:00',
]

YEAR_SET1 = [
    '1979', '1980', '1981',
    '1982', '1983', '1984',
    '1985', '1986', '1987',
    '1988', '1989', '1990',
    '1991', '1992', '1993',
    '1994', '1995', '1996',
    '1997', '1998', '1999',
    '2000', '2001', '2002',
    '2003', '2004', '2005',
    '2006', '2007', '2008',
    '2009', '2010', '2011',
    '2012', '2013', '2014',
    '2015', '2016', '2017',
    '2018', '2019',
]

DAYS = [
    '01', '02', '03',
    '04', '05', '06',
    '07', '08', '09',
    '10', '11', '12',
    '13', '14', '15',
    '16', '17', '18',
    '19', '20', '21',
    '22', '23', '24',
    '25', '26', '27',
    '28', '29', '30',
    '31',
]


def retrieve_set1(variable_set: list) -> None:
    """
    Queues and downloads data from the reanalysis-era5-single-levels
    dataset.
    """
    for y in YEAR_SET1:

        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': variable_set[1],
                'year': y,
                'month': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                ],
                'day': DAYS,
                'time': TIME,
                'format': 'netcdf',
                'area': [
                    43.75, -79.5, 43.5,
                    -79.25,
                ],
            },
            y + '_' + variable_set[0] + '.nc')


def main() -> None:
    """
    Starts threads to download all required data.
    """
    t1_1 = threading.Thread(target=retrieve_set1, args=(FEATURES_LIST_1,))
    t1_2 = threading.Thread(target=retrieve_set1, args=(FEATURES_LIST_2,))

    t1_1.start()
    tm.sleep(3)
    t1_2.start()
    tm.sleep(3)

    t1_1.join()
    t1_2.join()


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        # the names (strs) of imported modules
        'extra-imports': ['threading', 'time', 'cdsapi', 'python_ta.contracts'],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })

    import python_ta.contracts
    python_ta.contracts.DEBUG_CONTRACTS = False
    python_ta.contracts.check_all_contracts()

    import doctest
    doctest.testmod()
