# radio_flux_calc
Flux calculator for radio images. Calculates flux in Jy from one or two radio images, calculating also the RMS and integrated noise. Also calculates the mean flux inside of the region in Jy/beam.

usage:

radio_flux_calc.py [-h] -i1 IMAGE1 -sr SOURCE_REG -nr NOISE_REG [-i2 IMAGE2] [-m1 MASK1] [-m2 MASK2]

Calculate radio flux and noise from FITS images using DS9 regions.

options:
  -h, --help            show this help message and exit
  
  -i1 IMAGE1, --image1 IMAGE1
                        Path to primary FITS image
                        
  -sr SOURCE_REG, --source_reg SOURCE_REG
                        Path to DS9 source region file
                        
  -nr NOISE_REG, --noise_reg NOISE_REG
                        Path to DS9 noise region file
                        
  -i2 IMAGE2, --image2 IMAGE2
                        Path to secondary FITS image for comparison
                        
  -m1 MASK1, --mask1 MASK1
                        Path to FITS mask for image1
                        
  -m2 MASK2, --mask2 MASK2
                        Path to FITS mask for image2
