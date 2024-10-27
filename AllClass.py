##
##        @file    AllClass.py
##        @brief
##        @author  Wei-Lun Chen (wlchen)
##                 $LastChangedBy: wlchen $
##        @date    $LastChangedDate: 2024-10-17 16:42:21 #$
##        @version $LastChangedRevision: 2499 $
##

from my_header import *
from args import *

# --- Particle Class Definition ---
class Particle:
    def __init__(self, *args):
        # Define the attributes for the Particle class
        attributes = ["MassPDG", "BRPDG", 'DC', "Int"]  
        # PDG mass, branching ratio, decay constant, additional parameter
        for attr, value in zip(attributes, args):
            setattr(self, attr, value)  
            # Dynamically assign the values to the attributes

    @classmethod
    def ang0(cls, x, j, Q0, N, t0):
        mtau   = 1.776
        offset = int(2*t0)
        Q      = np.log((1 + np.cos(x)) / 2)
        return -2 * 2**(2*t0) / np.pi * np.cos(j * x) * ((mtau**2 - Q**2 > 0).astype(float)) * (-2 / Q +
        4 * Q / mtau**2 - 2 * Q**3 / mtau**4) / ((1 + np.cos(x))**offset) * np.tanh((Q / Q0)**N)

    @classmethod
    def ang1(cls, x, j, Q0, N, t0):
        mtau   = 1.776
        offset = int(2*t0)
        Q      = np.log((1 + np.cos(x)) / 2)
        return -2 * 2**(2*t0) / np.pi * np.cos(j * x) * ((mtau**2 - Q**2 > 0).astype(float)) * (-2 / Q +
        6 * Q**3 / mtau**4 - 4 * Q**5 / mtau**6) / ((1 + np.cos(x))**offset) * np.tanh((Q / Q0)**N)

    @classmethod
    def error_kaon(cls, Q0):
        return np.tanh(0.4938/Q0)

    @classmethod
    def error_kstar(cls, Q0):
        return np.tanh(0.892/Q0)
      
    @classmethod
    def find_J_CRI(cls, Q0_Q_tuple, J, Val, rounding=6, **kwargs):
      
        Q0, N, t0 = Q0_Q_tuple
    
        J_CRI = None
    
        if J == 0:
            coefficients = np.array([quad(cls.ang0, 0, np.pi, args=(j, Q0, N, t0))[0] for j in range(50)])
        elif J == 1:
            coefficients = np.array([quad(cls.ang1, 0, np.pi, args=(j, Q0, N, t0))[0] for j in range(50)])
        else:
            raise ValueError(f"No supported angular momentum: {J}")
          
        J_CRI = None
        for j, value in enumerate(coefficients):
            if np.abs(value) < Val and all(np.abs(coefficients[k]) < Val for k in range(j + 1, len(coefficients))):
                J_CRI = j
                break
    
        A = round(cls.error_kaon(Q0) , rounding)
        B = round(cls.error_kstar(Q0), rounding)
        
        Debug = kwargs.get('Debug', None)
        if Debug == True:
            print(f'Q0_J={J}: {round(Q0, rounding)}, J_CRI_J{J}: {J_CRI}, tanh is {A if J == 0 else B}')
            return J_CRI, Q0
        else:
            return J_CRI, Q0
    
    @classmethod
    def kernel_approx(cls, J, data, t0=0.5, N=1, rounding=6, **kwargs):
        offset    = int(2 * t0)
        Debug     = kwargs.get('Use_Debug', None)
        Use_Q0omt = kwargs.get('Use_Q0omt', None)

        if Use_Q0omt is True:
            Q0_values = np.linspace(0.0025, 0.1, 50)
            #Val = 1000 * (data[offset]**-1)
            Val = 1
            Q0_Q_pairs = [(Q0, N, t0) for Q0 in Q0_values]

            with mp.Pool() as pool:
                J_CRI_Q0_pairs = pool.starmap(cls.find_J_CRI, [(Q0_Q_tuple, J, Val) for Q0_Q_tuple in Q0_Q_pairs])

            J_CRI_values = [x[0] for x in J_CRI_Q0_pairs if x[0] is not None]

            if not J_CRI_values:
                raise ValueError("No valid J_CRI values found. Please check the input data and parameters.")

            min_J_CRI = min(J_CRI_values)
            best_Q0 = Q0_values[J_CRI_values.index(min_J_CRI)]

            # Calculate the ratio arrays
            ang_function = cls.ang0 if J == 0 else cls.ang1
            coefficients = np.array([quad(ang_function, 0, np.pi, args=(j, best_Q0, N, t0))[0] for j in range(24)])
        
            A = round(cls.error_kaon(best_Q0), rounding)
            B = round(cls.error_kstar(best_Q0), rounding)

            print(f'For J={J}, Best Q0: {round(best_Q0, rounding)} with tanh value is {A if J == 0 else B}, Min J_CRI: {min_J_CRI}')
        
            if Debug is True:
                print('Kernel coeffient K(Q) is' , coefficients)
                return coefficients * data[offset]
            else:
                return coefficients * data[offset]

        elif isinstance(Use_Q0omt, (int, float)):
            Q0 = Use_Q0omt
            print("Using Q0: ", Q0)

            # Calculate the ratio arrays
            ang_function = cls.ang0 if J == 0 else cls.ang1
            coefficients = np.array([quad(ang_function, 0, np.pi, args=(j, Q0, N, t0))[0] for j in range(24)])

            A = round(cls.error_kaon(Q0),  rounding)
            B = round(cls.error_kstar(Q0), rounding)
            print(f'For J={J}, Q0: {round(Q0, rounding)}, with tanh value is {A if J == 0 else B}')

            if Debug is True:
                # Print the results
                print('Kernel coeffient is' , coefficients)
                return coefficients * data[offset]
            else:
                return coefficients * data[offset]

        else:
            raise ValueError("Invalid input for Q0. Please provide a valid value for Q0.")

class LatticeFermion:
    def __init__(self, **kwargs):
        attributes = ["Group","Beta","Inva","a", "TimeSlicing", "L", "ZA"]
        for attr in attributes:
            setattr(self, attr, kwargs.get(attr, None))
        
        self.perfac = Prefactor # GF ** 2 * mtau ** 3 / (16 * np.pi)
        self.vusex  = VUSPDGEX
        self.vusin  = VUSPDGIN
        self.vpi    = 1
        self.cof_ex = Prefactor * SEW * VUSPDGEX ** 2 / totaldecaywidth 
        self.cof_in = Prefactor * SEW * VUSPDGIN ** 2 / totaldecaywidth 
        
    @classmethod
    def twopt_cosh_function(t , p, TimeSlicing):
        a = p['A']  # array of a[i]s
        b = p['M']  # array of E[i]s
        
        NT = TimeSlicing // 2
        return sum(2* ai * gv.exp(- bi * NT ) * gv.cosh((-1) * bi * ( t - NT )) for ai, bi in zip(a, b))

    @classmethod
    def twopt_exp_function(t,p):
        a = p['A'] # array of a[i]s
        b = p['M'] # array of E[i]s
        
        return sum(ai * gv.exp(-bi * t) for ai, bi in zip(a, b))

    def lattice_spacing(self):
        '''Converting from fm to GeV'''
        temp = self.a
        result = 197.3 / (1000 * temp)
        return result
    
    def spatial_volume(self):
        '''Return the spatial volume'''
        Volume = (self.L)**3
        return Volume
    
    def cof_chebshev(self, degree):
        '''This function generates a array which indicates the cof of shifted chebshev'''
        coeffs = []
        # for T_0
        coeffLine = [1]
        coeffs.append(coeffLine)
        # for T_1
        coeffLine = [0, 1]
        coeffs.append(coeffLine)
        #print(coeffs)

        for i in range(2, degree + 1):
            coeffLine = [0] * (1 + i)
            coeffLine[0] = -coeffs[i - 2][0]
            for j in range(1, i - 1):
                coeffLine[j] = 2 * coeffs[i - 1][j - 1] - coeffs[i - 2][j]
            coeffLine[-2] = 2 * coeffs[i - 1][-2]
            coeffLine[-1] = 2 * coeffs[i - 1][-1]
            coeffs.append(coeffLine)        

        shiftedCoeffs = [0] * (degree + 1)
        for i in range(degree + 1):
            binom = 1
            for j in range(i + 1):
                shiftedCoeffs[i - j] += coeffs[-1][i] * 2 ** (i - j) * (-1) ** j * binom
                binom *= (i - j) / (j + 1)

        return np.array(shiftedCoeffs)             
            
    def file_path(self):
        """Return the absolute path of a folder under the Group directory."""
        # Get the absolute path of the current working directory
        base_path  = os.getcwd()

        # Construct the path by expanding the user directory and joining with self.Group
        group_path = os.path.join(base_path, self.Group)

        # Check if the path exists, raise an error if it doesn't
        if not os.path.exists(group_path):
            raise FileNotFoundError(f"The directory {group_path} does not exist.")

        # Return the absolute path
        return group_path

    def read_file(self, **kwargs):
        path = self.file_path()

        # Check if 'GammaMatrix' is provided in kwargs, if not, skip the file processing part
        GammaMatrix = kwargs.get('GammaMatrix', None)
        ConfigNum   = kwargs.get('ConfigNum'  , None)

        if GammaMatrix:
            # Only process the file if GammaMatrix is provided
            IMP = ['1', '2', '3', '4', '51', '52', '53', '54']
            LL  = ['1_1', '2_2', '4_4', '8_8', '11_11', '13_13', '14_14', '7_7']

            # Refactor GammaDict into a single dictionary
            GammaDict = {IMP[i]: LL[i] for i in range(8)}

            # Construct the filename based on GammaMatrix and other attributes
            filename = f'{path}/corr_Z2_l{self.L}t{self.TimeSlicing}_Ls{self.Ls}_m{self.mud}_LL_m{self.s}_LL_{GammaDict.get(GammaMatrix)}'
      
            # Read the file
            data = pd.read_csv(filename, sep='\s+', header=None, parse_dates=True)

            return cp.array(data[6])
        
        # If 'GammaMatrix' is not provided, you can return None or handle it differently
        
        elif ConfigNum:
          
            # Define the filename
            filename = f'{path}/A4A4/conf_{ConfigNum}'
          
            # Read the file
            data = pd.read_csv(A4A4, sep='\s+', header=None, parse_dates=True)
            return cp.array(data)

        return None

    def jacknifed_correlation(self, ND = 1, **kwargs):
        
        Debug = kwargs.get('Use_Debug', None)
        EO    = kwargs.get('EO', None)

        data = self.read_file(**kwargs)
        
        TotalLength = len(data) // ND
        Ncfg        = self.Ncfg // ND
        TimeSlice   = self.TimeSlicing
        Exp         = TotalLength // (Ncfg * TimeSlice)
        
        if ND == 1 :
            data_array = data.reshape(Ncfg, Exp, TimeSlice)
        else:
            data_array = data[TotalLength:2*TotalLength].reshape(Ncfg, Exp, TimeSlice)

        # Average over Exp (axis=1)
        average_over_exp   = cp.mean(data_array, axis=1)

        # Perform jackknife averaging over Ncfg (axis=0)
        sum_over_ncfg      = cp.sum(average_over_exp, axis=0)
        jackknife_averages = cp.zeros((Ncfg, TimeSlice))

        #for i in range(Ncfg):
        #    jackknife_sum = sum_over_ncfg - average_over_exp[i::Ncfg, :]
        #    jackknife_averages[i, :] = jackknife_sum / (Ncfg - 1)
        sum_over_ncfg_expanded = cp.tile(sum_over_ncfg, (Ncfg, 1))
        jackknife_averages = (sum_over_ncfg_expanded - average_over_exp) / (Ncfg - 1)

        # Compute the mean of the jackknife averages
        mean_jackknife_averages = cp.mean(jackknife_averages, axis=0, keepdims=True)

        # Calculate the average value over Ncfg (axis=0)
        average_over_ncfg = cp.mean(average_over_exp, axis=0)

        # Calculate the covariance matrix
        
        # Compute the mean of the jackknife averages
        mean_jackknife_averages = cp.mean(jackknife_averages, axis=0, keepdims=True)

        # Calculate the deviations from the mean for each jackknife average
        deviations = jackknife_averages - mean_jackknife_averages

        # Compute the jackknife error
        covariance_matrix_jack = cp.mean(cp.expand_dims(deviations, axis=2) * cp.expand_dims(deviations, axis=1), axis=0) * (Ncfg - 1)

        result = gv.gvar(average_over_ncfg.get() , covariance_matrix_jack.get())
        
        if EO == 'Even':
            result = result[::2]
        elif EO == 'Odd':
            result = result[1::2]
        else:
            result
            
        if Debug == True:
            print(f"Jacknifed correlation function is {result}.")
            return result
        else:
            return result
    
    def correlation(self, ND, **kwargs):
        
        return self.jacknifed_correlation(ND, **kwargs).sdev

    def cbar(self, t, ND = 1, t0 = 0.5, **kwargs):

        Debug     =  kwargs.get('Use_Debug', None)
        EO        = kwargs.get('EO', None)
        ext_data  = kwargs.get("Ext_data", None)
        
        if ext_data is not None:
            data = ext_data
        else:
            data = self.jacknifed_correlation(ND, **kwargs)
            
        offset = int(2 * t0)
        Cbar   = np.roll(data, -offset) / data[offset]
        
        if Debug == True:
            print(f'Cbar of {self.Name} is {Cbar[1:t+1]}.')
            return Cbar[1:t+1]    
        else:
            return Cbar[1:t+1]   
        
    def directsum_chebshev(self, T, ND = 1, t0 = 0.5, **kwargs):
        
        Debug     = kwargs.get('Use_Debug', None)
        EO        = kwargs.get('EO', None)
        ext_data  = kwargs.get("Ext_data", None)

        results = []
        
        for t in range(1,T):
            data_CCbar = self.cbar(t, ND, t0, **kwargs)
            Cof        = self.cof_chebshev(t)[1::]
            result     = cp.sum(data_CCbar * Cof) + self.cof_chebshev(t)[0]
            results.append(result)
            
        return np.array(results)

    def make_prior(self, A_list, AE_list, M_list, ME_list, N=1):
        '''
        Constructs a dictionary of priors for a fit based on the given lists of parameters.

        Parameters:
        A_list (list): List of Amplitude 
        AE_list (list): List of error of amplitude 
        M_list (list): List of Mass
        ME_list (list): List of error of mass
        N (int, optional): Specifies how many state uis fitted. Default is one.

        Raises:
        ValueError: If the lists do not have the same length or if N is greater than the number of elements in the lists.

        Returns:
        BufferDict: A BufferDict with keys 'a' and 'b', each containing a list of GVar objects.
        '''        

        if len(A_list) != len(AE_list) or len(M_list) != len(ME_list) or len(A_list) != len(M_list):
            raise ValueError("All input lists must have the same length")\
                
        if N > len(A_list):
            raise ValueError("N cannot be greater than the number of elements in the input lists")

        prior = gv.BufferDict()
        prior['A'] = [gv.gvar(f"{A}({AE})") for A, AE in zip(A_list[:N], AE_list[:N])]
        prior['M'] = [gv.gvar(f"{M}({ME})") for M, ME in zip(M_list[:N], ME_list[:N])]

        return prior

    def fit_build_list(self, b, d, A_list, AE_list, M_list, ME_list, data, ND=1, N=1, function_type='exp', set=1):
        """
        Generalized function to perform either an exponential or hyperbolic cosine fit.

        Parameters:
        b (int): Starting index for data fitting.
        d (int): Ending index for data fitting.
        A_list (list): List of initial amplitude guesses.
        AE_list (list): List of amplitude error guesses.
        M_list (list): List of initial mass guesses.
        ME_list (list): List of mass error guesses.
        data (array-like): Data to fit.
        ND (int): Parameter for jackknife correlation (default is 1).
        N (int): A parameter for the prior guess function (default is 1).
        function_type (str): Type of function to use for fitting ('exp' or 'cosh').
        set (int): Optional flag for printing results when using 'cosh' function (default is 1).

        Returns:
        np.array: Fitted values for the entire range of time slices.
        """
        
        # Step 1: Perform jackknife correlation on data
        data = self.jacknifed_correlation(ND, data=data)
        
        # Step 2: Create x array and prior information
        x     = np.arange(b, d)
        prior = self.make_prior(A_list, AE_list, M_list, ME_list, N)
        
        # Step 3: Choose the fitting function based on the function_type
        if function_type == 'exp':
            fit_function = cls.twopt_exp_function()
        elif function_type == 'cosh':
            fit_function = cls.twopt_cosh_function()
        else:
            raise ValueError("Invalid function_type. Choose either 'exp' or 'cosh'.")
        
        # Step 4: Perform the nonlinear fit
        fit = lsqfit.nonlinear_fit(data=(x, data[b:d]), fcn=fit_function, prior=prior, p0=None, svdcut=1e-15)
        
        # Step 5: Return fitted values based on the function type
        if function_type == 'exp':
            return np.ravel([sum(fit.p['A'] * np.exp(-fit.p['M'] * t)) for t in range(self.TimeSlicing)])
        
        elif function_type == 'cosh':
            NT = self.TimeSlicing // 2
            if set == 1:
                print(fit)
            return np.ravel([sum(2 * fit.p['A'] * np.exp(-fit.p['M'] * NT) * np.cosh(-fit.p['M'] * (t - NT))) for t in range(self.TimeSlicing)])

    def plot_directsum_chebshev(self, T, ND = 1, t0=0.5, title=None, size = 12, **kwargs):
      
        DirectSum_results = self.directsum_chebshev(T, ND, t0, **kwargs)
        x_ranges = [x for x in range(1, T)]
        y_values = gv.mean(DirectSum_results)
        y_err    = gv.sdev(DirectSum_results)
        
        plt.figure(figsize=[10,8])
        plt.errorbar(x_ranges, y_values, yerr=y_err, fmt='o', capsize=5)
        plt.fill_between(x_ranges, -1, 1, color='gray', alpha=0.3)  # Add the shaded region
        plt.xticks(np.arange(1, T, 1.0), fontsize = size )
        plt.xlabel('T', fontsize = size)
        plt.ylabel('DirectSum', fontsize = size)
        if title:
            plt.title(title)
            
        plt.show()

    def plot_correlator(self, b, d, ND = 1, size = 12, **kwargs):
      
        data = self.jacknifed_correlation(ND, **kwargs)  
        plt.figure(figsize=[12,10])

        t_ranges = [x for x in range(b, d)]
        plt.errorbar(t_ranges, gv.mean(data[b:d]), yerr=gv.sdev(data[b:d]), fmt='o', capsize=5)
        plt.xticks(np.arange(b, d, 2.0), fontsize=size)

        plt.xlabel('T', fontsize=2 * size)
        plt.ylabel('Correlator', fontsize=2 * size)
        plt.suptitle(f'Correlator of {self.Name}', fontsize= 2 * size)
        #plt.savefig(f'CorFit/Correlator{self.Name}.png')
        plt.axhline(y= 0 , color='r', linestyle='-')
        
        if kwargs.get('log', False):
            plt.yscale('log')
            
        plt.show()
        return None
    
    def plot_effectivemass(self, b, d, size = 24, ND = 1, **kwargs):
        
        data    = self.jacknifed_correlation(ND, **kwargs)  
        t_range = cp.array([t for t in range(b, d)])
        diff    = cp.array([cp.log(data[t+1])-cp.log(data[t]) for t in range(b,d)])
        size    = 24
        
        plt.figure(figsize=[12,10])
        plt.errorbar(t_range , gv.mean(diff.get()),yerr=gv.sdev(diff.get()), fmt='o', capsize=5)
        plt.xticks(np.arange(b, d, 2.0), fontsize = size //2 )

        plt.xlabel('T', fontsize=size)
        plt.ylabel('EffectiveMass', fontsize=size)
        plt.suptitle(f'EffectiveMass of {self.Name}', fontsize = int (4 * size / 3))
        plt.axhline(y=0, color='r', linestyle='-')
            
        plt.show()
        return None

    def check_eigenvalues(self, ND = 1, n=1e-15, **kwargs):
        '''
        Checks the eigenvalues of a covariance matrix. 

        Parameters:
        data: The Average Data.

        Returns:
        int: The number of eigenvalues smaller than 1e-12.
        '''
        
        # Check the dimension of the covariance matrix
        covariance_matrix=gv.evalcov(self.correlation(ND, **kwargs))
        dim = covariance_matrix.shape
        print(f"The dimension of the matrix is {dim}.")

        # Verify symmetry
        assert cp.allclose(covariance_matrix, covariance_matrix.T)

        # Verify diagonal elements
        error_array = cp.sqrt(cp.diag(covariance_matrix))
        assert cp.all(cp.diag(covariance_matrix) >= 0)
        assert cp.allclose(cp.diag(covariance_matrix), error_array**2)

        # Verify positive semi-definiteness by checking the non-negative eigenvalues
        eigvals = cp.linalg.eigvals(covariance_matrix)
        
        neg_eigvals     = eigvals[eigvals < n]
        num_neg_eigvals = len(neg_eigvals)
        num_eigvals     = len(eigvals)
        
        print(f"There are {num_eigvals} eigenvalues in the covariance matrix.")
        print(f"There are {num_neg_eigvals} eigenvalues in the covariance matrix smaller than {n}.")

        if num_neg_eigvals > 0:
            print(f"The eigenvalues smaller than 1e-12 are: \n {neg_eigvals}")

        return num_neg_eigvals

    def timeslicing(self, t):
        return np.arange(1, t + 1)

    def make_prior_Chebyshen(self, t):
        prior = gv.BufferDict()
        prior['erfinv(Tj)'] = gv.gvar(t*['0(1)'])*(gv.sqrt(2)**(-1))
        return prior

    def fcnVB(self, t, prior):
        Tj = prior['Tj']
        IDK = [gv.gvar(0, 0) for i in range(len(t))]
        for i in range(len(t)):
            n = i+1
            IDK[n-1] = spsp.binom(2*n, n)/2
            for j in range(n):
                r = j+1
                IDK[n-1] +=  Tj[r-1] * spsp.binom(2*n, n-r)
            IDK[n-1] *= 2.0**(1-2*n)
        return IDK

    def chebshev_fit_base(self, t, ND = 1, t0=0.5, svdcut=None, debug=False, print_fit=False, **kwargs):

        Debug     = kwargs.get('Use_Debug', None)
        EO        = kwargs.get('EO', None)

        # Calculate Cbar and time slicing
        Cbar    = self.cbar(t , ND = ND, t0 = t0, **kwargs)
        data_t  = self.timeslicing(t)
        if EO == "None":
            data_t = data_t
        elif EO == "Even":
            data_t = 2* data_t
        elif EO == "Odd":
            data_t = 2* data_t - 1
        elif EO == "True":
            data_t = data_t
        else:
            data_t = data_t
            
        # Create prior for Chebyshev fit
        prior = self.make_prior_Chebyshen(t)

        # Fit the data using lsqfit with optional SVD cut
        fit = lsqfit.nonlinear_fit(
            data=(data_t, Cbar), 
            fcn=self.fcnVB, 
            prior=prior, 
            svdcut=svdcut, 
            fitter='scipy_least_squares', 
            debug=debug
        )

        # Print fit results if required
        if print_fit:
            print(fit)
            print(fit.format(maxline=True))
            return None

        # Return the fitted parameter Tj
        return fit.p['Tj']

    def chebshev_fit(self, t, data, t0=0.5, **kwargs):
        Debug = kwargs.get('Use_Debug', None)

        if Debug == True:
            return self.chebshev_fit_base(t, data, t0, svdcut=1e-12, debug=True, print_fit=True, **kwargs)
        else:
            return self.chebshev_fit_base(t, data, t0, debug=True, **kwargs)
        ##return self.chebshev_fit_base(t, data, t0, debug=True, **kwargs)

    def chebshev_fit_chi(self, t, data, t0=0.5, **kwargs):
        Debug = kwargs.get('Use_Debug', None)

        return self.chebshev_fit_base(t, data, t0, svdcut=1e-12, debug=True, print_fit=True, **kwargs)


class MockData(LatticeFermion):
    def __init__(self, **kwargs):
        attributes = ["Name","Beta","L","Alpha",'Amplitude','S','Mass']
        for attr in attributes:
            setattr(self, attr, kwargs.get(attr, None))
            super().__init__(**kwargs)

    def jacknifed_correlation(self, amp = None, mas = None):
        '''a for decay constant and b for mass.'''
        '''Use same name to do function overiding.'''
        amp = self.Amplitude
        mas = self.Mass
        
        i = np.arange(0, 48)
        SingleExP = amp ** 2 * mas * np.exp(-mas * i) / 2
        
        return SingleExP

    def mock_decaywidth(self, t, t0=0.5, S = None, amp = None, mas = None, **kwargs):
        '''
        Use_Fit = True will use Fitting, else use direct sum        
        '''
        S         = self.S
        amp       = self.Amplitude
        mas       = self.Mass
        Debug     = kwargs.get('Use_Debug', None)
        Use_Fit   = kwargs.get('Use_Fit', None)
        SingleExP = self.jacknifed_correlation(amp, mas)
                
        if Use_Fit == True:
            Singlet = self.directsum_chebshev(t + 1, SingleExP, t0)
        else:
            Singlet = self.chebshev_fit_base(t , ND=1, t0=t0)
        
        ker    = Particle.kernel_approx(S, SingleExP, t0, Use_Debug = Debug, Use_Q0omt = Q0)
        result = self.cof_in * (0.5 * ker[0] + cp.dot(Singlet, ker[1:t+1]) )
        
        print(f"Decay rate of {self.Name} is", result)
        
        if Debug == True:
            #print("Kernel coeffient ", ker)
            print("prefactor is ", self.cof_in)
            print("Cof is", Singlet)
            print("The decay width is ", result)
            return result
        else:
            return result
      
    def mock_decaywidth_by_integral(self, S = None, amp  = None, mas = None, **kwargs):
        
        S     = self.S
        amp   = self.Amplitude
        mas   = self.Mass
        Rat   = mas / 1.776
        Debug = kwargs.get('Use_Debug', None)

        if S == 1:
            integrand = amp**2 * (1 - Rat**2)**2 * (1 + 2 * Rat**2) 
        elif S == 0:
            integrand = amp**2 * (1 - Rat**2)**2
            
        result = self.cof_in * integrand
        print("Integral gives ", result)

        if Debug == True:
            return result
        else:
            return result

    def compare_decaywidths(self, T, t0=0.5, **kwargs):
        
        Save      = kwargs.get('Save_Pic', None)
        Debug     = kwargs.get('Use_Debug', None)
        Use_Fit   = kwargs.get('Use_Fit', None)
        t_values  = np.arange(0, T)
        size      = 24
        tick_size = 16
        
        integral_value = self.mock_decaywidth_by_integral(**kwargs)
        
        decay_values = []
        for t in t_values:
            try:
                decay_value = self.mock_decaywidth(t, t0=t0, Use_Fit = Use_Fit , Use_Debug = Debug)
                decay_values.append(decay_value)
            except ValueError as e:
                if Debug == True:
                    print(f"Error calculating decay width at t={t}: {e}")
                    print(f"Length of t_values: {len(t_values)}")
                    print(f"Length of decay_values: {len(decay_values)}")
                else:
                    decay_values.append(np.nan)

        plt.figure(figsize=(10, 6))
        plt.errorbar(t_values, 
                     gv.mean(decay_values),
                     gv.sdev(decay_values), 
                     label='mock_decaywidth(t)', 
                     color='b')
        plt.axhline(y=gv.mean(integral_value), 
                    color='r', 
                    linestyle='--', 
                    label='Exclusive decaywidth')

        plt.title(f'Comparison of Decay Widths for {self.Name}', fontsize = size)
        plt.xlabel('N', fontsize = size)
        plt.ylabel('Decay Width', fontsize = size)
        plt.legend(fontsize = size // 1.5)
        plt.grid(True)
        plt.xticks(t_values, fontsize=tick_size) 
        plt.yticks(fontsize=tick_size)
        plt.tight_layout()
        if Save == True:
            plt.savefig(f'Figs/DirectSumof{self.Name}{EO}Grid.png')
            plt.show()
        else:
            plt.show()

class WilsonFermion(LatticeFermion):
    def __init__(self, **kwargs):
        attributes = ["Group","Beta","Inva","a","TimeSlicing", 
                      "L","Alpha","r"]
        for attr in attributes:
            setattr(self, attr, kwargs.get(attr, None))
        super().__init__(**kwargs)

class DomainWall(LatticeFermion):
    def __init__(self, **kwargs):
        attributes = ["Group","Beta","Inva","a","TimeSlicing", 
                      "L","Ls",'M5','PlaquetteAverage']
        for attr in attributes:
            setattr(self, attr, kwargs.get(attr, None))
        super().__init__(**kwargs)

class RBCQCD(DomainWall):
    def __init__(self, **kwargs):
        attributes = ["Group","Name","Ncfg","Beta","Inva","a","TimeSlicing", 
                      "L","Ls","ZV","s","mud",'M5','PlaquetteAverage']
        for attr in attributes:
            setattr(self, attr, kwargs.get(attr, None))
        super().__init__(**kwargs)
        self.IMP = [1, 2, 3, 4, 51, 52, 53, 54]
        
    def spectral(self, S, T, t0=0.5, **kwargs):
        # This function computes the spectral result based on the Chebyshev fit and the kernel.
        # S (0 or 1) selects the kernel with related angular momentum.
        Debug       = kwargs.get('Use_Debug', None)
        Use_Fit     = kwargs.get('Use_Fit', None)
        GammaMatrix = kwargs.get("GammaMatrix", None)
        ext_data    = kwargs.get("Ext_data", None)
        EO          = kwargs.get("EO", None)
        
        if Use_Fit == True:
            Cheb = self.chebshev_fit_base(T, t0, **kwargs)
        else:
            Cheb = self.directsum_chebshev(T, t0, **kwargs)

        Ker  = Particle.kernel_approx(S, ext_data, t0, Use_Debug = Debug, Use_Q0omt = Q0)
        result = 0.5 * Ker[0] + cp.dot(Cheb, Ker[1:T+1])
        return result
        
    def vus(self, T, t0=0.5, **kwargs):
        
        Ina         = self.Inva
        ZA          = self.ZA
        Debug       = kwargs.get('Use_Debug', None)
        Use_Fit     = kwargs.get('Use_Fit', None)
        EO          = kwargs.get("EO", None)
        ND          = 1
        Volume      = self.L**3
                
        if EO == "None":
            data = {}
            for i in self.IMP:
                data[f'data_{i}'] = self.jacknifed_correlation(ND=ND, GammaMatrix=str(i), EO=EO)
            data_1, data_2, data_3, data_4 = data['data_1'], data['data_2'], data['data_3'], data['data_4']
            data_51, data_52, data_53, data_54 = data['data_51'], data['data_52'], data['data_53'], data['data_54']
        
        elif EO == "Even":
            data = {}
            for i in self.IMP:
                data[f'data_{i}'] = self.jacknifed_correlation(ND=ND, GammaMatrix=str(i), EO="Even")
            data_1, data_2, data_3, data_4 = data['data_1'], data['data_2'], data['data_3'], data['data_4']
            data_51, data_52, data_53, data_54 = data['data_51'], data['data_52'], data['data_53'], data['data_54']
        
        elif EO == "Odd":
            data = {}
            for i in self.IMP:
                data[f'data_{i}'] = self.jacknifed_correlation(ND=ND, GammaMatrix=str(i), EO="Odd")
            data_1, data_2, data_3, data_4 = data['data_1'], data['data_2'], data['data_3'], data['data_4']
            data_51, data_52, data_53, data_54 = data['data_51'], data['data_52'], data['data_53'], data['data_54']

            data_5S = 1/3*(data_51+ data_52 + data_53) / Volume
            data_4  = data_4 / Volume
            data_S  = 1/3*(data_1 + data_2  + data_3)  / Volume
            data_54 = data_54 / Volume
            
            Axial_T  = self.spectral(1, T, t0, Ext_data = data_54, Use_Debug = Debug, Use_Fit = Fit)
            Vector_T = self.spectral(0, T, t0, Ext_data = data_4 , Use_Debug = Debug, Use_Fit = Fit)
            Axial_S  = self.spectral(1, T, t0, Ext_data = data_5S, Use_Debug = Debug, Use_Fit = Fit)
            Vector_S = self.spectral(0, T, t0, Ext_data = data_S , Use_Debug = Debug, Use_Fit = Fit)

            Spectrum_Sum = (Axial_T + Vector_T + Axial_S + Vector_S)
            result = np.sqrt((Spectrum_Sum*Prefactor*SEW*Ina**3/(totaldecaywidth*ZA**2))**(-1))
            
            print(f"Vus = {result} for data = {self.Name} with N={T} terms")
            
            if Debug:
                print(f"Vus inclusive is {self.vusin}, result of cheb is {result}")
                return result
            else:
                return result

        elif EO == "True":
            print("Hello world")
            sys.exit()
            return 1

        data_5S = 1/3*(data_51+ data_52 + data_53) / Volume
        data_4  = data_4 / Volume
        data_S  = 1/3*(data_1 + data_2  + data_3)  / Volume
        data_54 = data_54 / Volume
        
        Axial_T  = self.spectral(1, T, t0, Ext_data = data_54, Use_Debug = Debug, Use_Fit = Fit)
        Vector_T = self.spectral(0, T, t0, Ext_data = data_4 , Use_Debug = Debug, Use_Fit = Fit)
        Axial_S  = self.spectral(1, T, t0, Ext_data = data_5S, Use_Debug = Debug, Use_Fit = Fit)
        Vector_S = self.spectral(0, T, t0, Ext_data = data_S , Use_Debug = Debug, Use_Fit = Fit)

        Spectrum_Sum = (Axial_T + Vector_T + Axial_S + Vector_S)
        V_us = np.sqrt((Spectrum_Sum*Prefactor*SEW*Ina**3/(totaldecaywidth*ZA**2))**(-1))
        
        print(f"Vus = {V_us} for data = {self.Name} with N={T} terms")
        
        if Debug:
            print(f"Vus inclusive is {self.vusin}, result of cheb is {V_us}")
            return V_us
        else:
            return V_us

    def plot_vus(self, N, t0=0.5, **kwargs):
        
        Use_Fit    = kwargs.get('Use_Fit', None)
        EO         = kwargs.get("EO", None)
        Save       = kwargs.get("Save_Pic", None)
        Debug      = kwargs.get('Use_Debug', None)
        N_values   = np.arange(0, N)
        size       = 24
        tick_size  = 16

        vus_values = [self.vus(T, t0 = t0, 
                               Use_Fit = Use_Fit, 
                               EO = EO, 
                               Use_Debug = Debug) for T in range(1,N+1)]
        Title_name = f"Vus as N , data= RBCQCD{self.Name}"
        
        plt.figure()
        plt.errorbar(N_values, 
                     gv.mean(vus_values), 
                     gv.sdev(vus_values), 
                     marker='o',
                     label='Vus_cheb', 
                     color='b')
        plt.axhline(y=gv.mean(self.vusex), 
                    color='r', 
                    linestyle='--', 
                    label='V_us_exp')
        plt.xlabel('N', fontsize = size)
        plt.ylabel('Vus', fontsize = size)
        plt.title(Title_name, fontsize = tick_size)
        plt.grid(True)
        plt.xticks(N_values, fontsize= tick_size) 
        plt.yticks(fontsize= tick_size)
        plt.tight_layout()

        if Save == True:
            plt.savefig(f'Figs/Vus_of_{self.Name}_{EO}Grid.png')
            plt.show()
        else:
            plt.show()


    def plot_all_GammaMatrix(self, T, t0 = 0.5 ,Num=3, title='None', size = 12, **kwargs):
        EO    = kwargs.get('EO', None)
        Debug = kwargs.get('Use_Debug', None)
        Save  = kwargs.get('Save_Pic', None)

        IMP = ['1', '2', '3', '4', '51', '52', '53', '54']

        t0_dict = {
            0.5: {'color': 'blue', 'fmt': 'o'},
            1.0: {'color': 'gold', 'fmt': '^'},
            1.5: {'color': 'purple', 'fmt': 's'},
            2.0: {'color': 'black', 'fmt': 'd'},
        }

        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 24))

        axes = axes.flatten()

        for i, GammaMatrix in enumerate(IMP):
            ax = axes[i]

            x_values = [x for x in range(1, T)]
            offset = 0.1
            
            if Debug == True:
                print(GammaMatrix)

            for j in range(Num):
                t0 = 0.5 + 0.5 * j  

                DirectSum_results = self.directsum_chebshev(T, ND = 1, t0=t0, GammaMatrix=GammaMatrix, **kwargs)
                y_values = gv.mean(DirectSum_results)
                y_err    = gv.sdev(DirectSum_results)

                x_values_offset = [x + offset * j for x in x_values]  
                ax.errorbar(x_values_offset, y_values, yerr=y_err, fmt=t0_dict[t0]['fmt'], capsize=5, color=t0_dict[t0]['color'], label=f'GammaMatrix={GammaMatrix}, t0={t0}')

            ax.fill_between(x_values_offset, -1, 1, color='gray', alpha=0.3)  
            ax.legend(fontsize= size + 8)
            ax.set_xticks(np.arange(1, T, 1.0))
            ax.tick_params(axis='both', which='major', labelsize= size + 4)
            
            ax.set_xlabel(r'$N_t$', fontsize=24)
            ax.set_ylabel(r'$\left\langle\bar{T}_{N_t}\left(e^{-\hat{H}} \right)\right\rangle$', fontsize=2 * size)

            if title:
                plt.title(title)

        fig.suptitle(f'DirectSum of Chebeshev from RBC/UKQCD(2018) Data set {EO}, L={self.L}', fontsize=3 * size)
                
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if Save == True:
            plt.savefig(f'Figs/DirectSumof{self.Name}{EO}Grid.png')
            plt.show()
        else:
            plt.show()

        
class BRIDGE(DomainWall):
  
    def __init__(self, **kwargs):
        attributes = ["Group", "Name", "Ncfg", "Beta", "Inva", "a", "TimeSlicing", 
                      "L","Ls","ZV","ms","mud",'M5']
        for attr in attributes:
            setattr(self, attr, kwargs.get(attr, None))
            
    def ReadFile(self):
        data = pd.read_csv(f'{self.Group}/A4A4',header=None,sep='\s+')
        return data 


# --- Define Instances of the Particle Class ---
# Tau lepton particle with PDG mass, branching ratio, decay constant, and additional parameter
Tau  = Particle(gv.gvar(1.77686, 0.00018), 1, 1, 1)
Kaon = Particle(gv.gvar(0.493677, 0.000013), gv.gvar(6.96 * 10 ** (-3), 0.1 * 10 ** (-3)),
                gv.gvar(0.156, 0.0012), 0.00708743)
Kstar = Particle(gv.gvar(0.89166, 0.00026), gv.gvar(1.42 * 10 ** (-2), 0.07 * 10 ** (-2)),
                 gv.gvar(0.2228, 0.02), 0.0140239)
Pion = Particle(gv.gvar(0.139, 0.00026), gv.gvar(0.1082, 0.0005), 0.113876)

## Instances for different lattice data.
MockKaon  = MockData(Name = 'Kaon' , L = 1, S = 0, Mass = Kaon.MassPDG , Amplitude = Kaon.DC)
MockKstar = MockData(Name = 'Kstar', L = 1, S = 1, Mass = Kstar.MassPDG, Amplitude = Kstar.DC)
MockPion  = MockData(Name = 'Pion' , L = 1, S = 0, Mass = Pion.MassPDG , Amplitude = Pion.DC)

FourEight = RBCQCD(Group='RBCQCD', Name='FourEight', Ncfg = 88, TimeSlicing = 96,L = 48,
                   Ls=12,s=0.0362,    Inva = Inverse_a48, ZA = ZA_48,
                  mud=0.00078 ,M5=1.8,PlaquetteAverage=gv.gvar(0.5871119,0.00000025))
SixFour   = RBCQCD(Group='RBCQCD', Name='SixFour', Ncfg = 80, TimeSlicing = 128, L = 64, 
                  Ls = 12, s=0.02661, Inva = Inverse_a64, ZA = ZA_64,
                  mud=0.000678,M5=1.8,PlaquetteAverage=gv.gvar(0.6153342,0.00000021))
Pegasus   = BRIDGE(Group="Pegasus", Name= "Pegasus", Ncfg = 10, TimeSlicing = 64, L = 32, Ls = 16, M5= 1.8)


start = time.time()  # 現在時刻（処理開始前）を取得


#SixFour.jacknifed_correlation(GammaMatrix="54")
#SixFour.jacknifed_correlation(1, GammaMatrix="54", EO = "Even")
#FourEight.plot_correlator(3, 12 , GammaMatrix="54", log = True)
#FourEight.plot_directsum_chebshev(10 , GammaMatrix="1", EO = "Even")
#FourEight.plot_all_GammaMatrix(10)
#FourEight.chebshev_fit_base(8, GammaMatrix="54")
#FourEight.jacknifed_correlation(ND = ND, GammaMatrix = "54", EO = EO, Use_Debug = Debug)
#FourEight.vus(T, EO = EO, Use_Fit = Fit , Use_Debug = Debug)
#SixFour.vus(T, EO = EO, Use_Fit = Fit, Use_Debug = Debug)
#SixFour.plot_vus(T, EO = EO, Use_Fit = Fit, Use_Debug = Debug, Save_Pic = Save)
FourEight.plot_vus(T, EO = EO, Use_Fit = Fit, Use_Debug = Debug, Save_Pic = Save)

#MockKstar.mock_decaywidth(T, Use_Debug = Debug, Use_Fit = Fit)
#MockKstar.mock_decaywidth_by_integral(Use_Debug = Debug)
#MockPion.compare_decaywidths(T, Use_Debug = Debug) / MockPion.vusin**2
#MockKaon.compare_decaywidths(T, Use_Debug = Debug, Use_Fit = Fit, Save_Pic = Save)
#MockKstar.compare_decaywidths(T, Use_Debug = Debug, Use_Fit = Fit, Save_Pic = Save)
#print(MockKaon.jacknifed_correlation())

end = time.time()  # 現在時刻（処理完了後）を取得

time_diff = end - start  # 処理完了後の時刻から処理開始前の時刻を減算する
print(time_diff, "seconds is used")  # 処理にかかった時間データを使用