#!/usr/bin/env python3
import argparse
import numpy as np
import csv
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
    with fits.open(img_path) as hdul:
        data = hdul[0].data.squeeze()
        header = hdul[0].header
        wcs = WCS(header).celestial
        
    beam_area_pix = get_beam_area_pixels(header)

    if fits_mask_path:
        with fits.open(fits_mask_path) as hdul:
            fits_mask = hdul[0].data.squeeze().astype(bool)
            data[~fits_mask] = np.nan

    stats = {'sources': [], 'noise': None}

    if noise_reg_path:
        noise_mask = get_combined_mask(noise_reg_path, wcs, data.shape)
        noise_pixels = data[noise_mask]
        noise_pixels = noise_pixels[~np.isnan(noise_pixels)]

        stats['noise'] = {
            'noise_rms': np.sqrt(np.mean(noise_pixels**2)),
            'noise_std': np.std(noise_pixels),
            'noise_robust': mad_std(noise_pixels)
        }

    source_regions = Regions.read(source_reg_path, format='ds9')
    
    for i, reg in enumerate(source_regions):
        pix_reg = reg.to_pixel(wcs)
        reg_mask = pix_reg.to_mask()
        
        if reg_mask is None: continue
        img_mask = reg_mask.to_image(data.shape)
        if img_mask is None: continue
        
        source_pixels = data[img_mask.astype(bool)]
        source_pixels = source_pixels[~np.isnan(source_pixels)]
        
        if len(source_pixels) == 0:
            continue

        src_stats = {
            'id': i + 1,
            'n_pixels': len(source_pixels),
            'n_beams': len(source_pixels) / beam_area_pix,
            'mean_flux_jy_beam': np.mean(source_pixels),
            'flux_jy': np.sum(source_pixels) / beam_area_pix
        }
        
        src_stats['calibration_error_jy'] = flux_scale_err * src_stats['flux_jy']
        
        if stats['noise']:
            src_stats['integrated_noise_jy'] = stats['noise']['noise_robust'] * np.sqrt(src_stats['n_beams'])
            src_stats['total_flux_error_jy'] = np.sqrt(src_stats['integrated_noise_jy']**2 + src_stats['calibration_error_jy']**2)
        else:
            src_stats['integrated_noise_jy'] = 0.0
            src_stats['total_flux_error_jy'] = src_stats['calibration_error_jy']
            
        stats['sources'].append(src_stats)

    return stats

def save_to_csv(stats1, stats2, output_file):
    """Saves the results directly to a CSV file."""
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        
        # Single Image Mode
        if stats2 is None:
            writer.writerow(['ID', 'Flux (Jy)', 'Flux error (Jy)'])
            for src in stats1['sources']:
                writer.writerow([src['id'], f"{src['flux_jy']:.6f}", f"{src['total_flux_error_jy']:.6f}"])
        
        # Two Image Comparison Mode
        else:
            writer.writerow(['ID', 'Img1 Flux (Jy)', 'Img1 Flux error (Jy)', 'Img2 Flux (Jy)', 'Img2 Flux error (Jy)', 'Diff 1-2 (Jy)', 'Ratio 1/2'])
            min_len = min(len(stats1['sources']), len(stats2['sources']))
            for i in range(min_len):
                src1 = stats1['sources'][i]
                src2 = stats2['sources'][i]
                
                diff_flux = src1['flux_jy'] - src2['flux_jy']
                ratio_flux = src1['flux_jy'] / src2['flux_jy'] if src2['flux_jy'] != 0 else np.nan
                
                writer.writerow([
                    src1['id'],
                    f"{src1['flux_jy']:.6f}", f"{src1['total_flux_error_jy']:.6f}",
                    f"{src2['flux_jy']:.6f}", f"{src2['total_flux_error_jy']:.6f}",
                    f"{diff_flux:.6f}", f"{ratio_flux:.4f}"
                ])
                
    print(f"Success! Data saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Calculate radio flux and export to CSV.")
    parser.add_argument("-i1", "--image1", required=True, help="Path to primary FITS image")
    parser.add_argument("-sr", "--source_reg", required=True, help="Path to DS9 source region file")
    
    parser.add_argument("-nr", "--noise_reg", help="Path to DS9 noise region file (Optional)")
    parser.add_argument("-i2", "--image2", help="Path to secondary FITS image for comparison")
    parser.add_argument("-m1", "--mask1", help="Path to FITS mask for image1")
    parser.add_argument("-m2", "--mask2", help="Path to FITS mask for image2")
    parser.add_argument("-fe", "--flux_err", type=float, default=0.10, help="Fractional flux scale error (default: 0.10)")
    parser.add_argument("-o", "--output", default="region_fluxes.csv", help="Output CSV filename (default: region_fluxes.csv)")
    
    args = parser.parse_args()

    # Process Image 1
    stats1 = process_image(args.image1, args.source_reg, args.noise_reg, args.mask1, args.flux_err)
    
    # Process Image 2 (If provided)
    stats2 = None
    if args.image2:
        stats2 = process_image(args.image2, args.source_reg, args.noise_reg, args.mask2, args.flux_err)

    # Save outputs
    save_to_csv(stats1, stats2, args.output)

if __name__ == "__main__":
    main()
