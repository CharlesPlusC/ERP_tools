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
            ephemeris_df = sp3_ephem_to_df(sat_name, date)  # Assume this function is correctly implemented
            # now slice the ephemeris into num_arcs arcs, all of length arc_length
            # take every other row of the ephemeris to get to 1-minute intervals
            ephemeris_df = ephemeris_df.iloc[::2]
            print(f"ephemeris df head: {ephemeris_df.head()}")
            for i in range(num_arcs):
                # slice the ephemeris into the observations arc
                start_index = i * arc_length
                end_index = start_index + arc_length
                OD_arc = ephemeris_df.iloc[start_index:end_index]
                
                # Now the OP arc starts at the same time as the OD arc, but ends prop_length seconds later
                prop_end_index = start_index + prop_length // 60  # convert seconds to minutes
                prop_end_index = min(prop_end_index, len(ephemeris_df) - 1)  # ensure we do not go out of bounds
                OP_arc = ephemeris_df.iloc[start_index:prop_end_index]
                print(f"OP arc head: {OP_arc.head()}")
                print(f"time difference between first and last UTC stamp: {OP_arc.iloc[-1]['UTC'] - OP_arc.iloc[0]['UTC']}")
            # benchmark(sat_name, date)

    #TODO: remember to slice the ephemeris to 1min intervals
    #TODO: pre-slice the ephemeris into different arcs

if __name__ == "__main__":
    main()

# filter ephemeris to get OD points and pass those as a sub DF
# pass the ephemeris points for the OP benchmarking and pass those as another DF
