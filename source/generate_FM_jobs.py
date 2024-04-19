from tools.sp3_2_ephemeris import sp3_ephem_to_df

def main():
    sat_names_to_test = ["GRACE-FO-A", "GRACE-FO-B", "TerraSAR-X", "TanDEM-X"]
    dates_to_test = ["2019-01-01", "2023-05-04"]
    num_arcs = 10 #number of OD arcs that will be run
    arc_length = 25 # length of each arc in minutes
    prop_length = 60 * 60 * 12 # length of propagation in seconds
    force_model_configs = [
        {'36x36gravity': True, '3BP': True},
        {'120x120gravity': True, '3BP': True},
        {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True},
        {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True},
        {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True},
        {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True},
        {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True, 'jb08drag': True},
        {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True, 'dtm2000drag': True},
        {'120x120gravity': True, '3BP': True,'solid_tides': True, 'ocean_tides': True, 'knocke_erp': True, 'relativity': True, 'SRP': True, 'nrlmsise00drag': True}
    ]
    for sat_name in sat_names_to_test:
        for date in dates_to_test:
            benchmark(sat_name, date)

    #TODO: remember to slice the ephemeris to 1min intervals
    #TODO: pre-slice the ephemeris into different arcs

if __name__ == "__main__":
    main()

# filter ephemeris to get OD points and pass those as a sub DF
# pass the ephemeris points for the OP benchmarking and pass those as another DF
