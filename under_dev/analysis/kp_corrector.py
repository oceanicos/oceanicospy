import numpy as np
from ..utils import wave_props,constants,extras

class Kp_corrector(self):
      

def _compute_nonadaptive_Kp(self,freqs):
        
        total_depth = self.anchoring_depth + self.sensor_height
        L = np.array([wave_props.wavelength(1/f, total_depth) for f in freqs])
        k = 2*np.pi/L
        Kp = np.cosh(k * self.sensor_height) / np.cosh(k * total_depth)
        if freqs[0]==0:
            Kp[0] = 1

        Kp_min = (np.cosh(np.pi/(total_depth - self.sensor_height)*self.sensor_height)) / \
                (np.cosh(np.pi/(total_depth - self.sensor_height)*total_depth))
        Kp = np.clip(Kp, Kp_min, 1)

        # fmax_Kp = 1/(2*np.pi)*np.sqrt(9.8*(np.pi/(total_depth - self.sensor_height))*np.tanh(np.pi/(total_depth-self.sensor_height)*total_depth))
        # freqs_to_modify = freqs <= fmax_Kp

        return Kp
    
    def _compute_adaptive_Kp(self,freqs,PSD):
        total_depth = self.anchoring_depth + self.sensor_height
        L = np.array([wave_props.wavelength(1/f, total_depth) for f in freqs])
        k = 2*np.pi/L
        Kp = np.cosh(k * self.sensor_height) / np.cosh(k * total_depth)
        Kp_min = (np.cosh(np.pi/(total_depth - self.sensor_height)*self.sensor_height)) / \
                (np.cosh(np.pi/(total_depth - self.sensor_height)*total_depth))

        if freqs[0] == 0:
            Kp[0] = 1

        PSD_Kp = PSD / (Kp**2)
        PSD_Kp_smoothed = self._smooth_psd_spectrum(PSD_Kp,24)
        minima_idx, _ = find_peaks(-np.log(PSD_Kp_smoothed),prominence=0.5)
        last_min_idx = minima_idx[-1]
        f_maxpcorr = freqs[last_min_idx]

        # Assuming f_maxpcorr is always lower than fcmax and fmax_Kp
        total_depth = self.anchoring_depth + self.sensor_height
        L_max = wave_props.wavelength(1/f_maxpcorr, total_depth)
        k_max = 2*np.pi/L_max
        Kp_min = np.cosh(k_max * self.sensor_height) / np.cosh(k_max * total_depth)

        Kp_final = Kp.copy()
        Kp_final[0:last_min_idx] =  Kp[0:last_min_idx]
        Kp_final[last_min_idx:] = Kp_min

        # plt.figure()
        # plt.loglog(freqs, PSD_Kp_smoothed, label='corrected')
        # plt.axvline(freqs[last_min_idx])
        # plt.savefig('/scratchsan/medellin/ffayalac/IG_analysis/src/field_analysis/plot_one_spectra.png')
        return Kp_final
    
    def _correction_by_Kp(self,freqs,PSD,kp_method='adaptive'):
        "Based on https://doi.org/10.1016/j.cageo.2017.06.010"
        if kp_method == 'nonadaptive':
            Kp = self._compute_nonadaptive_Kp(freqs)
        else:
            Kp = self._compute_adaptive_Kp(freqs,PSD)
        PSD_Kp = PSD / (Kp**2)

        return freqs, PSD_Kp, PSD
