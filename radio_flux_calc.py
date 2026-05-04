#!/usr/bin/env python3
import argparse
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from regions import Regions
from astropy.stats import mad_std

def get_beam_area_pixels(header):
    """Calculates the beam area in pixels."""
    bmaj = header['BMAJ'] * 3600
    bmin = header['BMIN'] * 3600
    pixscale = abs(header['CDELT1'] * 3600.)
    return (np.pi * bmaj * bmin) / (4 * np.log(2) * pixscale**2)

def create_region_mask(region_path, wcs, shape):
    """Reads a DS9 region file and creates a boolean mask matching the image shape."""
    regions = Regions.read(region_path, format='ds9')
    full_mask = np.zeros(shape, dtype=bool)
    
    # Loop handles files with multiple region shapes by combining them
    for reg in regions:
        pix_reg = reg.to_pixel(wcs)
        reg_mask = pix_reg.to_mask()
        if reg_mask is not None:
            img_mask = reg_mask.to_image(shape)
            if img_mask is not None:
                full_mask |= img_mask.astype(bool)
                
    return full_mask

def process_image(img_path, source_reg_path, noise_reg_path, fits_mask_path=None):
    """Extracts data, applies masks, and calculates all requested statistics."""
    # 1. Load Image Data
    with fits.open(img_path) as hdul:
        data = hdul[0].data.squeeze()
        header = hdul[0].header
        wcs = WCS(header).celestial
        
    beam_area_pix = get_beam_area_pixels(header)

    # 2. Apply optional FITS mask (e.g., from source finding)
    if fits_mask_path:
        with fits.open(fits_mask_path) as hdul:
            fits_mask = hdul[0].data.squeeze().astype(bool)
            data[~fits_mask] = np.nan

    # 3. Create region masks
    source_mask = create_region_mask(source_reg_path, wcs, data.shape)
    noise_mask = create_region_mask(noise_reg_path, wcs, data.shape)

    # 4. Extract valid pixels
    source_pixels = data[source_mask]
    source_pixels = source_pixels[~np.isnan(source_pixels)]
    
    noise_pixels = data[noise_mask]
    noise_pixels = noise_pixels[~np.isnan(noise_pixels)]

    # 5. Calculate Statistics
    stats = {}
    
    # Noise Stats
    stats['noise_rms'] = np.sqrt(np.mean(noise_pixels**2))
    stats['noise_std'] = np.std(noise_pixels)
    stats['noise_robust'] = mad_std(noise_pixels)
    
    # Source Stats
    stats['n_pixels'] = len(source_pixels)
    stats['n_beams'] = stats['n_pixels'] / beam_area_pix
    
    stats['mean_flux_jy_beam'] = np.mean(source_pixels)
    stats['flux_jy'] = np.sum(source_pixels) / beam_area_pix
    
    # Integrated noise: RMS_beam * sqrt(N_beams)
    # Using robust noise (MAD) as the baseline for radio astronomy
    stats['integrated_noise_jy'] = stats['noise_robust'] * np.sqrt(stats['n_beams'])

    return stats

def print_stats(name, stats):
    """Nicely formats and prints the results."""
    print(f"\n{'='*40}")
    print(f"Results for: {name}")
    print(f"{'='*40}")
    
    print("\n--- Source Region ---")
    print(f"Total Flux:        {stats['flux_jy']:.6f} Jy")
    print(f"Mean Flux:         {stats['mean_flux_jy_beam']:.6e} Jy/beam")
    print(f"Integrated Noise:  {stats['integrated_noise_jy']:.6f} Jy")
    print(f"Valid Pixels:      {stats['n_pixels']}")
    print(f"Independent Beams: {stats['n_beams']:.2f}")

    print("\n--- Background Noise Region ---")
    print(f"DS9 RMS:           {stats['noise_rms']:.6e} Jy/beam")
    print(f"DS9 StdDev:        {stats['noise_std']:.6e} Jy/beam")
    print(f"Robust Noise (MAD):{stats['noise_robust']:.6e} Jy/beam")

def main():
    parser = argparse.ArgumentParser(description="Calculate radio flux and noise from FITS images using DS9 regions.")
    parser.add_argument("-i1", "--image1", required=True, help="Path to primary FITS image")
    parser.add_argument("-sr", "--source_reg", required=True, help="Path to DS9 source region file")
    parser.add_argument("-nr", "--noise_reg", required=True, help="Path to DS9 noise region file")
    
    # Optional arguments
    parser.add_argument("-i2", "--image2", help="Path to secondary FITS image for comparison")
    parser.add_argument("-m1", "--mask1", help="Path to FITS mask for image1")
    parser.add_argument("-m2", "--mask2", help="Path to FITS mask for image2")
    
    args = parser.parse_args()

    # Process Image 1
    stats1 = process_image(args.image1, args.source_reg, args.noise_reg, args.mask1)
    print_stats("Image 1", stats1)

    # Process Image 2 (If provided)
    if args.image2:
        stats2 = process_image(args.image2, args.source_reg, args.noise_reg, args.mask2)
        print_stats("Image 2", stats2)
        
        # Print Comparison
        print(f"\n{'='*40}")
        print("COMPARISON (Image 1 vs Image 2)")
        print(f"{'='*40}")
        diff_flux = stats1['flux_jy'] - stats2['flux_jy']
        ratio_flux = stats1['flux_jy'] / stats2['flux_jy'] if stats2['flux_jy'] != 0 else np.nan
        
        print(f"Flux Difference (1 - 2): {diff_flux:.6f} Jy")
        print(f"Flux Ratio (1 / 2):      {ratio_flux:.4f}")
        print(f"\nImage 1 Flux: {stats1['flux_jy']:.6f} +/- {stats1['integrated_noise_jy']:.6f} Jy")
        print(f"Image 2 Flux: {stats2['flux_jy']:.6f} +/- {stats2['integrated_noise_jy']:.6f} Jy")
        print(f"{'='*40}\n")

if __name__ == "__main__":
    main()