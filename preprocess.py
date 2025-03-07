import numpy as np
import pandas as pd
import ephem
import scipy.odr as sodr
from astropy.table import Table

class Preprocess:
    def __init__(self, scan_number, data_path='./Data/', resampling=50):
        self.scan_number = scan_number
        self.filename_data = f'{data_path}LCORR_AIFA_{scan_number}.csv'
        print(f"Reading data from {self.filename_data}")
        self.resampling = resampling
        self.data = self._load_data()

        # Create a private Timing instance
        self._timing = _Timing(self.data, self.resampling)
        # # attributes
        self._time_resampled = self._timing._get_time_resampled()
        #change: _get_time_resampled() returns a numpy array of time values 
        #instead of a list

        self._time_sun = self._timing._get_sun_info()
        self._twilight = self._timing._get_sun_info(twilight=True)
        self._night_mask = self._timing._get_night_mask(self._time_sun)
    
    def _load_data(self):
        """Loads the data from the specified file and returns it as a DataFrame."""
        return pd.read_csv(self.filename_data, delimiter=' ', header=None)
        
    def calibrate_single_channel(self):
        """Method to do temperature calibration for each antenna.
        Explain the method here ... TODO
        """
        
        print("\n Starting baseline fit and calibration...")
        self.antenna_power = self._get_antenna_power()
        self.antenna_temp = self._get_antenna_temp(self.antenna_power)
        self.antenna_rms = self._get_antenna_rms(self.antenna_temp)
        print("\n Calibration complete.")
        # print(type(self.antenna_power), type(self.antenna_temp), type(self.antenna_rms))
        return self.antenna_power, self.antenna_temp, self.antenna_rms
    
    def calibrate_correlation(self):
        """
        Calibrate the correlation data.
        
        This method gets the correlation data for each baseline (EM, WM, EW) and applies 
        rotation matrices to correct for phase offsets. It then subtracts the baseline
        offset to get the calibrated correlation data.
        
        EM: East-Middle baseline
        WM: West-Middle baseline
        EW: East-West baseline
        
        Returns:
        - dict: Calibrated correlation data and RMS error for each baseline.
                The correlation data and RMS error are stored as a tuple for each baseline.
                The correlation data and RMS error are dictionaries with keys 'Real' and 'Img'.
        """
        print("\n Starting correlation calibration...")
        self.correlation_wm, self.err_correlation_wm = self._get_correlation(baseline="WM")
        self.correlation_em, self.err_correlation_em = self._get_correlation(baseline="EM")
        self.correlation_ew, self.err_correlation_ew = self._get_correlation(baseline="EW")
        print("\n Correlation calibration complete.")
        
        correlation_data = [self.correlation_wm, self.correlation_em, self.correlation_ew]
        correlation_rms = [self.err_correlation_wm, self.err_correlation_em, self.err_correlation_ew]
        return [correlation_data, correlation_rms]
    
    def _get_antenna_power(self):
        """
        Returns the power data for the antennas.
        
        Get raw power data from data columns 4, 5, 6, 7.
        
        Returns:
        - dict: Resampled power data for each antenna.
        """
        # Get power of single channel data:
        pow_M1 = self.data[4]  # Power from Middle (Pluto2) Antenna
        pow_M2 = self.data[5]  # Power from Middle (Pluto3) Antenna
        pow_E = self.data[6]   # Power from East (Pluto2) Antenna
        pow_W = self.data[7]   # Power from West (Pluto3) Antenna
        
        # Resample the power signals by averaging
        r = self.resampling
        pow_M1_res = np.array([np.average(pow_M1[i:i+r]) for i in range(0, len(pow_M1), r)])
        pow_M2_res = np.array([np.average(pow_M2[i:i+r]) for i in range(0, len(pow_M2), r)])
        pow_E_res  = np.array([np.average(pow_E[i:i+r]) for i in range(0, len(pow_E), r)])
        pow_W_res  = np.array([np.average(pow_W[i:i+r]) for i in range(0, len(pow_W), r)])

        # Convert the power data to a structured array        
        resampled_power = np.core.records.fromarrays(
                        [
                            pow_W_res, pow_E_res, pow_M1_res, pow_M2_res
                        ],
                        dtype=[
                            ("West", np.float32), 
                            ("East", np.float32), 
                            ("Middle1", np.float32), 
                            ("Middle2", np.float32)
                        ]
                    )         
        
        # # DEBUG
        # # Print resampled power data for all antennas
        # print("\n Resampled Power Data:")
        # for key in resampled_power.dtype.names: 
        #     print(f"{key}: {resampled_power[key][:5]}")
        
        # print(f"Data Type of res Power: {type(resampled_power)}")
        # Return resampled power data for all antennas
        return resampled_power
    
    def _get_antenna_temp(self, antenna_power):
        """
        Converts power measurements into antenna temperature.
        
        The routine estimates the background baseline found from nighttime data via a polynomial fit.
        We then subtract that fit from the raw power measurements. This yields the “antenna temperature” done
        by baseline fit and correction
        """
        # Get power data for all antennas      
        pow_W_res = antenna_power["West"]
        pow_E_res = antenna_power["East"] 
        pow_M1_res = antenna_power["Middle1"]
        pow_M2_res = antenna_power["Middle2"]
    
        # Select nighttime data for baseline fit
        x_off = self._time_resampled[self._night_mask]
        y_off = {
            "West": pow_W_res[self._night_mask],
            "East": pow_E_res[self._night_mask],
            "Middle1": pow_M1_res[self._night_mask],
            "Middle2": pow_M2_res[self._night_mask]
        }
        
        # Fit baseline for each antenna
        y_fit_off = {
            key: self._do_basefit3(x_off, y_off[key], self._time_resampled)
            for key in y_off
        }
        
        # Compute Antenna Temperature (T_A)
        T_A = {
            "West": pow_W_res - y_fit_off["West"],
            "East": pow_E_res - y_fit_off["East"],
            "Middle1": pow_M1_res - y_fit_off["Middle1"],
            "Middle2": pow_M2_res - y_fit_off["Middle2"]
        }
        
        # Convert the Antenna Temperature data to a structured array
        T_A = np.core.records.fromarrays(
            [
            pow_W_res - y_fit_off["West"],
            pow_E_res - y_fit_off["East"],
            pow_M1_res - y_fit_off["Middle1"],
            pow_M2_res - y_fit_off["Middle2"]
            ],
            dtype=[
                ("West", np.float32),
                ("East", np.float32),
                ("Middle1", np.float32),
                ("Middle2", np.float32),
            ]
        )
        
        # # DEBUG
        # print("\n Antenna Temperature Data:")
        # for key in T_A.dtype.names:
        #     print(f"{key} Antenna:", T_A[key][-10:], len(T_A[key]))
              
        return T_A
    
    # Calculate RMS error for each antenna temperature   
    def _get_antenna_rms(self, antenna_temp):
        """
            Compute RMS error for each antenna temperature
        """
        err_arrays = []
        for key in antenna_temp.dtype.names:
            err_value = self._compute_rms(antenna_temp[key], self._night_mask)
            err_arrays.append(err_value)
            
        # Convert RMS error data to a structured array
        err_T_A = np.core.records.fromarrays(err_arrays, 
                                             dtype=[(key, np.float32) 
                                                    for key in antenna_temp.dtype.names])
        
        # DEBUG
        print("\n RMS errors for each antenna:")
        for key in antenna_temp.dtype.names:
            print(f"{key} err_T_A: {err_T_A[key]}")
            
        return err_T_A
    
    def _get_correlation(self, baseline):
        # Get correlation data for all baselines
        if baseline=="EM":
            # Correlation data for East-Middle baseline
            corr_real = self.data[10]
            corr_imag = self.data[11]
        elif baseline=="WM":
            # Correlation data for West-Middle baseline
            corr_real = self.data[12]
            corr_imag = self.data[13]
        elif baseline=="EW":
            # Correlation data for East-West baseline
            corr_real = self.data[14]
            corr_imag = self.data[15]
        else:
            raise ValueError("Invalid baseline! Choose from 'EM', 'WM', 'EW'")
        
        # Resample the power signals by averaging
        r = self.resampling
        corr_mw_real_res = np.array([np.average(corr_real[i:i+r]) for i in range(0, len(corr_real), r)])
        corr_mw_imag_res = np.array([np.average(corr_imag[i:i+r]) for i in range(0, len(corr_imag), r)])
        
        # Stack the real and imaginary parts of the correlation data
        coor_mw = np.vstack([corr_mw_real_res, corr_mw_imag_res])
        
        # Define the rotation matrix with a phase offset for the baseline
        rot_mw = self._get_rotation_matrix(baseline)
        
        # Apply the rotation matrix to the correlation data
        rot_coor_mw = rot_mw @ coor_mw
        
        # split into real and imaginary part
        rot_coor_real = rot_coor_mw[0]
        rot_coor_img = rot_coor_mw[1]
        
        # Select nighttime data for baseline fit
        x_off = self._time_resampled[self._night_mask]
        y_off = {
            "Real": rot_coor_real[self._night_mask],
            "Img": rot_coor_img[self._night_mask]
        }
        
        # Fit baseline for each antenna
        y_fit_off = {
            key: self._do_basefit3(x_off, y_off[key], self._time_resampled)
            for key in y_off
        }
        
        # Compute new correlation data after baseline offset subtraction
        # Convert correlation data to a structured array
        correlation_calib = np.core.records.fromarrays(
            [
                rot_coor_real - y_fit_off["Real"],
                rot_coor_img - y_fit_off["Img"]
            ],
            dtype=[
                ("Real", np.float32),
                ("Img", np.float32)
            ]
        )
        
        err_correlation_real = self._compute_rms(correlation_calib["Real"], self._night_mask)
        err_correlation_img = self._compute_rms(correlation_calib["Img"], self._night_mask)
        
        # Convert RMS correlation to a structured array
        err_correlation = np.core.records.fromarrays(
            [
                err_correlation_real, err_correlation_img
            ],
            dtype=[
                ("Real", np.float32),
                ("Img", np.float32)
            ]
        )
        
        # # DEBUG
        # print(f"\n Correlation Data for baseline {baseline}:")
        # for key in correlation_calib.dtype.names:
        #     print(f"{key} Correlation:", correlation_calib[key][:5], "Length:", len(correlation_calib[key]))
        #     print(f"{key} Error:", err_correlation[key]) 

        return correlation_calib, err_correlation
        
    
    ## Helper methods
    # Helper method to calculate RMS error
    def _compute_rms(self, y_data, mask_night):
        """
            Computes the RMS error

        Params:
        - y_data: Antenna temperature
        - mask_night: Boolean mask for nighttime indices.
        """
        return np.sqrt(np.sum(y_data[mask_night] ** 2) / len(y_data[mask_night]))
    
    def _get_rotation_matrix(self, baseline):
        """
        Returns a 2D rotation matrix for a given baseline

        Parameters:
        - baseline (str): The name of the baseline ('MW', 'ME', 'EW').

        Returns:
        - np.array: 2x2 rotation matrix.
        """

        # Define phase offsets for all baselines in radians
        phase_offsets = {
            "EM": 2.93,  # East-Middle baseline
            "WM": 0.28,  # West-Middle baseline
            "EW": 3.11   # East-West baseline
        }

        # Get the phase offset
        phase_offset = phase_offsets.get(baseline)

        # Define the rotation matrix
        rot_mat =  np.array([
                            [np.cos(phase_offset), -np.sin(phase_offset)],
                            [np.sin(phase_offset), np.cos(phase_offset)]
                        ])
        return rot_mat

    
    # The baseline fit functions
    def _basefit3(self, p, x):  
        """ Third order polynomial for baseline fit 
        Coefficients [a, b, c, d] for the polynomial.
        """
        a,b,c,d=p
        return a*x**3 + b*x**2 + c*x+d  
    
    def _do_basefit3(self, x, y, x_fit):
        '''
        Baseline fit for 3. order polynomial
        
        Parameter:
        x: x-data set
        y: y-data set
        
        Output:
        y_fit: Baseline fit
        
        '''
        
        lin_model1 = sodr.Model(self._basefit3)
        fit_data = sodr.RealData(x,y ,sx=None, sy=None)
        odr = sodr.ODR(fit_data, lin_model1, beta0=[0,0,0,0])
        out = odr.run()
        
        # Get fit parameters and errors
        a_n, b_n, c_n, d_n = out.beta
        err_a, err_b, err_c, err_d = out.sd_beta
        res_var = out.res_var

        # # DEBUG
        # print("\n residuals:",res_var)
        # print("Parameter 1/a:",(a_n, err_a))
        # print("Parameter 2/b:",(b_n, err_b))
        # print("Parameter 3/c:",(c_n, err_c))
        # print("Parameter 4/d:",(d_n, err_d))

        y_fit = a_n*x_fit**3 + b_n*x_fit**2 + c_n*x_fit + d_n
        return y_fit

    def _get_lambda(self):
        """
        Returns observation wavelength.
        """
        self.obs_freq = 1419e6
        self.obs_bandwidth = 2e6
        self.obs_lambda = 299792458.0/self.obs_freq
        return self.obs_lambda



    def save_data(self):
        '''
        dumps all the preprocessed data into a fits file.

        Begin calibration of single channel data with baseline fits
        calibrated_data is a tuple of 3 Dictionaries
        1. antenna_power : Contains single channel power data for each antenna
        2. antenna_temp : Contains calibrated temperature data for each antenna
        3. antenna_rms : Contains rms error for each antenna temperature

        For each dictionary, the keys are the antenna names:
        'East', 'West', 'Middle1', 'Middle2'
        We shall keep the keys consistent throughout the scripts
        '''
        # preproc_data = Preprocess('322a', data_path='./Data/')
        
        # time_sun = self._time_sun[0]
        obs_lambda = self._get_lambda()
        time_h_res = self.get_resampled_time()
        sunrise = self.get_sunrise()
        sunset = self.get_sunset()
        twil_m = self.get_twilight_morning()
        twil_e = self.get_twilight_evening()

        #Begin calibration of single channel data with baseline fits
        antenna_power, antenna_temp, antenna_rms = self.calibrate_single_channel()

        # Correlation data calibration
        corr_dict = self.calibrate_correlation()

        metadata={
            'Obs_lambda': obs_lambda,
            'Obs_day':self.data[1][0],      
            'Obs_lat':float(self._timing._observer.lat),
            'Obs_lon':float(self._timing._observer.lon),
            'Obs_ele':float(self._timing._observer.elevation),            
            'Sunrise':sunrise,
            'Sunset':sunset,
            'Twil_m':twil_m,
            'Twil_e':twil_e,
            'RMS_west':float(antenna_rms['West']),
            'RMS_east':float(antenna_rms['East']),
            'RMS_middle1':float(antenna_rms['Middle1']),
            'RMS_middle2':float(antenna_rms['Middle2']),
        }


        data_to_save = [
            time_h_res,
            antenna_power['West'],
            antenna_power['East'],
            antenna_power['Middle1'],
            antenna_power['Middle2'],
            antenna_temp['West'],
            antenna_temp['East'],
            antenna_temp['Middle1'],
            antenna_temp['Middle2'],
            corr_dict[0][0]['Real'],
            corr_dict[0][0]['Img'],
            corr_dict[0][1]['Real'],
            corr_dict[0][1]['Img'],
            corr_dict[0][2]['Real'],
            corr_dict[0][2]['Img']
        ]

        keys = (
            'Time_resampled',
            'Power_west',
            'Power_east',
            'Power_middle1',
            'Power_middle2',
            'Temp_west',
            'Temp_east',
            'Temp_middle1',
            'Temp_middle2',
            'Corr_WM_real',
            'Corr_WM_img',
            'Corr_EM_real',
            'Corr_EM_img',
            'Corr_EW_real',
            'Corr_EW_img'
        )

        # print(len(data_to_save),len(keys))
            

        table = Table(data_to_save,names = keys,meta = metadata)

        filename_to_save = f'./Data/Processed_{self.scan_number}.fits'
        table.write(filename_to_save,overwrite=True)

        print(f'Preprocessing Finished and written to {filename_to_save}')
        return
    
    # Getters for resampled time, sunrise, sunset, transit and twilight times 
    # (Public methods)
    def get_resampled_time(self):
        """
        Returns:
            np.array: Resampled time data.
        """
        return self._time_resampled
    
    def get_sunrise(self):
        """
        Returns:
            datetime: The time of sunrise.
        """
        return self._time_sun[1]
    
    def get_sunset(self):
        """
        Returns:
            datetime: The time of sunset.
        """
        return self._time_sun[3]
    
    def get_transit(self):
        """
        Returns:
            datetime: The time of sun transit.
        """
        return self._time_sun[2]
    
    def get_twilight_morning(self):
        """
        Returns:
            datetime: The time of morning twilight.
        """
        return self._twilight[1]
    
    def get_twilight_evening(self):
        """
        Returns:
            datetime: The time of evening twilight.
        """
        return self._twilight[2]

    
    
# Private class for handling time-based calculations
class _Timing:
    """Handles time-based calculations (Private class)."""
    def __init__(self, data, resampling):
        self.data = data
        self.resampling = resampling
        self._observer = self._setup_observer()
        self._time_resampled = self._get_time_resampled()
   
    def _setup_observer(self):
        """Initializes an ephem.Observer instance with location parameters."""
        obs = ephem.Observer()
        obs.lat = '50.569417'
        obs.lon = '6.722'
        obs.elevation = 435
        return obs
        
    def _get_sec(self, time_str):
        '''
        Converts a time string (HH:MM:SS) into fractional hours.

        input: time string in format 00:00:00
        split time string into hours(h), minutes(m) and seconds(s)
        output: converted fractional time
        '''
        h,m,s = time_str.split(':')
        return (float(h)*3600+float(m)*60+float(s))/3600

    def _get_time_resampled(self): # == _time_h_res
        '''
        Reads the data file and converts time data to a fractional time array and resamples.
        
        input: scan number for datafile
        use get_sec function to convert time data to fractional time array
        output: fractional time array
        '''
        time = self.data[2]
        time_frac = [self._get_sec(time_str) for time_str in time]
        time_resampled = [np.average(time_frac[i:i+self.resampling]) 
                          for i in range(0,len(time_frac),self.resampling)]
        return np.array(time_resampled)

    def _get_sun_info(self, twilight=False):
        """
        If twilight=False:
            Gets sunrise, transit, and sunset times  
        If twilight=True:
            Sets the horizon to -18 deg and computes the times of astronomical twilight
    
        Returns:
            If twilight=False -> (time_sun, Rise, Transit, Set)
            If twilight=True  -> (twil_sun, T_m, T_e)
        """
        # Set the observer date based on the data file
        obs_day = self.data[1][0]
        self._observer.date = obs_day
        print("\n We still need to update observer to AIfA ... \n")
        
        if twilight:
            # For astronomical twilight
            self._observer.horizon = '-18'
            sun = ephem.Sun()
            sun.compute(self._observer)
            
            twil_morn = str(self._observer.next_rising(sun, use_center=True))
            twil_ev   = str(self._observer.next_setting(sun, use_center=True))
            print("Twilight morning at AIfA [UTC]:", twil_morn)
            print("Twilight evening at AIfA [UTC]:", twil_ev)
            
            # Convert HH:MM:SS to fractional hours
            T_m = int(twil_morn[-8:-6]) + int(twil_morn[-5:-3])/60 + int(twil_morn[-2:])/3600
            T_e = int(twil_ev[-8:-6])   + int(twil_ev[-5:-3])/60   + int(twil_ev[-2:])/3600
            
            twil_sun = np.array([])
            for t in self._time_resampled:
                if T_m <= t <= T_e:
                    twil_sun = np.append(twil_sun, t)
            
            return twil_sun, T_m, T_e
        
        else:
            # Normal sunrise/transit/sunset
            self._observer.horizon = '0'
            sun = ephem.Sun()
            sun.compute(self._observer)
            
            sunrise = str(self._observer.next_rising(sun))
            transit = str(self._observer.next_transit(sun))
            sunset  = str(self._observer.next_setting(sun))
            print("Observation date:", self._observer.date)
            print("Sunrise at AIfA [UTC]:", sunrise)
            print("Transit at AIfA [UTC]:", transit)
            print("Sunset at AIfA [UTC]:", sunset)
            
            Rise    = int(sunrise[-8:-6]) + int(sunrise[-5:-3])/60 + int(sunrise[-2:])/3600
            Transit = int(transit[-8:-6]) + int(transit[-5:-3])/60 + int(transit[-2:])/3600
            Set     = int(sunset[-8:-6])  + int(sunset[-5:-3])/60  + int(sunset[-2:])/3600
            
            time_sun = np.array([])
            for t in self._time_resampled:
                if Rise <= t <= Set:
                    time_sun = np.append(time_sun, t)
            
            return time_sun, Rise, Transit, Set
        
    def _get_night_mask(self, time_sun):
        """
        Computes a boolean mask for selecting nighttime data.

        Returns:
        - mask_night (np.array): Boolean array where True corresponds to nighttime data.
        """
        sun_rise = time_sun[1]
        sun_set = time_sun[3]

        # Create a mask for nighttime (before sunrise & after sunset)
        mask_night = (
            ((self._time_resampled >= 0) & (self._time_resampled <= sun_rise)) |
            ((self._time_resampled >= sun_set) & (self._time_resampled <= 24))
        )
        return mask_night
