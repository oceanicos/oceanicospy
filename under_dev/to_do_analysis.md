Guidelines to proofread the analysis subpackage in ***oceanicospy***

# Standard logic per class

The two main classes (for now) in this method are: *WaveSpectralAnalyzer* ad *WaveTemporalAnalyzer.* They are treated as objects that can do their corresponding intended analysis

Because there are different ways to perform a temporal or spectral analysis, those manners can just be implemented as methods in the main class.

# Things to check/implement

## Spectral analysis

- Verifying the computation of Hs for ig and sw band, this is particulary required in terms of expected behaviour in areas where ig-driven hs are higher.
- Finishing the plotting for scalograms for the waveletes methods in the examples/spectral_analysis.ipynb. This needs to be properly refactored because is too much resource-consuming

# Temporal analysis

* Do we really need a detrended time series for the up-crossing method? this can be verified quickly.
* The use of anchoring_depth and sensor height has to be properly defined and distinguished for Kp correction

# Final disclaimer

**Everything that has been code is not a straitjacket, feel free to purpose something that can lead to a stronger version. Any improvement is encouraged on the source code and the guide example on examples/reading_field_data.ipynb.**

Add claim your credits in the notes section for the methods you refactor or create. Making yourself visible is key to tracking efforts.
