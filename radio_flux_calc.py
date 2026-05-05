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

def get_combined_mask(region_path, wcs, shape):
    """Reads a DS9 region file and combines all shapes into a single boolean mask."""
    regions = Regions.read(region_path, format='ds9')
    full_mask = np.zeros(shape, dtype=bool)
    for reg in regions:
        pix_reg = reg.to_pixel(wcs)
        reg_mask = pix_reg.to_mask()
        if reg_mask is not None:
            img_mask = reg_mask.to_image(shape)
            if img_mask is not None:
                full_mask |= img_mask.astype(bool)
    return full_mask

def process_image(img_path, source_reg_path, noise_reg_path=None, fits_mask_path=None, flux_scale_err=0.10):
    """Extracts data, applies masks, and calculates statistics for multiple regions."""
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

    stats = {'sources': [], 'noise': None}

    # 3. Handle optional background noise region (Combined into one global mask)
    if noise_reg_path:
        noise_mask = get_combined_mask(noise_reg_path, wcs, data.shape)
        noise_pixels = data[noise_mask]
        noise_pixels = noise_pixels[~np.isnan(noise_pixels)]

        noise_stats = {
            'noise_rms': np.sqrt(np.mean(noise_pixels**2)),
            'noise_std': np.std(noise_pixels),
            'noise_robust': mad_std(noise_pixels)
        }
        stats['noise'] = noise_stats

    # 4. Handle source regions (Iterated individually)
    source_regions = Regions.read(source_reg_path, format='ds9')
    
    for i, reg in enumerate(source_regions):
        pix_reg = reg.to_pixel(wcs)
        reg_mask = pix_reg.to_mask()
        
        # Skip if region is completely outside the image
        if reg_mask is None: continue
        img_mask = reg_mask.to_image(data.shape)
        if img_mask is None: continue
        
        # Extract valid pixels for this specific shape
        source_pixels = data[img_mask.astype(bool)]
        source_pixels = source_pixels[~np.isnan(source_pixels)]
        
        # If the region hit a completely masked/NaN area, skip math to prevent crashes
        if len(source_pixels) == 0:
            continue

        # Calculate Source Statistics
        src_stats = {
            'id': i + 1,
            'n_pixels': len(source_pixels),
            'n_beams': len(source_pixels) / beam_area_pix,
            'mean_flux_jy_beam': np.mean(source_pixels),
            'flux_jy': np.sum(source_pixels) / beam_area_pix
        }
        
        src_stats['calibration_error_jy'] = flux_scale_err * src_stats['flux_jy']
        
        # Calculate errors dynamically based on whether noise was provided
        if stats['noise']:
            src_stats['integrated_noise_jy'] = stats['noise']['noise_robust'] * np.sqrt(src_stats['n_beams'])
            src_stats['total_flux_error_jy'] = np.sqrt(src_stats['integrated_noise_jy']**2 + src_stats['calibration_error_jy']**2)
        else:
            src_stats['integrated_noise_jy'] = 0.0
            src_stats['total_flux_error_jy'] = src_stats['calibration_error_jy']
            
        stats['sources'].append(src_stats)

    return stats

def print_stats(name, stats, flux_scale_err):
    """Nicely formats and prints the results as a table."""
    print(f"\n{'='*85}")
    print(f"Results for: {name}")
    print(f"{'='*85}")

    if stats['noise']:
        print("\n--- Background Noise Region ---")
        print(f"DS9 RMS:           {stats['noise']['noise_rms']:.6e} Jy/beam")
        print(f"Robust Noise (MAD):{stats['noise']['noise_robust']:.6e} Jy/beam")
    else:
        print("\n--- Background Noise Region: [Skipped] ---")

    print(f"\n--- Source Regions (Calibration Error Assumed: {flux_scale_err*100:.1f}%) ---")
    print(f"{'ID':<4} | {'Total Flux (Jy)':<18} | {'Total Error (Jy)':<18} | {'Mean (Jy/beam)':<15} | {'N_beams':<8}")
    print("-" * 85)
    
    total_image_flux = 0
    for src in stats['sources']:
        total_image_flux += src['flux_jy']
        print(f"{src['id']:<4} | {src['flux_jy']:<18.6f} | {src['total_flux_error_jy']:<18.6f} | {src['mean_flux_jy_beam']:<15.6e} | {src['n_beams']:<8.2f}")
    
    print("-" * 85)
    print(f"Sum of all regions: {total_image_flux:.6f} Jy")


def print_comparison(stats1, stats2):
    """Prints a region-by-region comparison table for two images."""
    print(f"\n{'='*85}")
    print("COMPARISON (Image 1 vs Image 2)")
    print(f"{'='*85}")
    
    # Ensure both lists are the same length before zipping
    min_len = min(len(stats1['sources']), len(stats2['sources']))
    
    print(f"{'ID':<4} | {'Img 1 Flux (Jy)':<16} | {'Img 2 Flux (Jy)':<16} | {'Diff (1-2)':<12} | {'Ratio (1/2)':<10}")
    print("-" * 85)
    
    for i in range(min_len):
        src1 = stats1['sources'][i]
        src2 = stats2['sources'][i]
        
        diff_flux = src1['flux_jy'] - src2['flux_jy']
        ratio_flux = src1['flux_jy'] / src2['flux_jy'] if src2['flux_jy'] != 0 else np.nan
        
        print(f"{src1['id']:<4} | {src1['flux_jy']:<16.6f} | {src2['flux_jy']:<16.6f} | {diff_flux:<12.6f} | {ratio_flux:<10.4f}")

    print("-" * 85 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Calculate radio flux and noise from FITS images using DS9 regions.")
    parser.add_argument("-i1", "--image1", required=True, help="Path to primary FITS image")
    parser.add_argument("-sr", "--source_reg", required=True, help="Path to DS9 source region file")
    
    parser.add_argument("-nr", "--noise_reg", help="Path to DS9 noise region file (Optional)")
    parser.add_argument("-i2", "--image2", help="Path to secondary FITS image for comparison")
    parser.add_argument("-m1", "--mask1", help="Path to FITS mask for image1")
    parser.add_argument("-m2", "--mask2", help="Path to FITS mask for image2")
    parser.add_argument("-fe", "--flux_err", type=float, default=0.10, 
                        help="Fractional flux scale error (default: 0.10). Pass 0 to disable.")
    
    args = parser.parse_args()

    # Process Image 1
    stats1 = process_image(args.image1, args.source_reg, args.noise_reg, args.mask1, args.flux_err)
    print_stats("Image 1", stats1, args.flux_err)

    # Process Image 2 (If provided)
    if args.image2:
        stats2 = process_image(args.image2, args.source_reg, args.noise_reg, args.mask2, args.flux_err)
        print_stats("Image 2", stats2, args.flux_err)
        print_comparison(stats1, stats2)

if __name__ == "__main__":
    main()
