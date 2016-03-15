# Build a quasar composite spectrum
## About

This notebook creates a composite quasar spectrum from a collection of quasar spectra.
The quasar spectra were retrieved from the *Sloan Digital Sky Survey* database DR 12 (http://skyserver.sdss.org/dr12/en/home.aspx) and stored in FITS files.

The notebook reads all FITS files in the given folder, corrects each individual spectrum for redshift (from observed frame to emitted frame), then re-bin-s the data in bins of a user-defined size. It then continues by calculating normalisation factors for each spectrum by the reciprocal of the division of the average spectral flux density in the overlapping region by the average spectral flux density of the adjacent (in the redshift range) spectrum in the same overlapping region.
Finally, all spectra are combined to one composite spectrum by calculating the *geometric* mean value of all values in a bin, for each bin.

## Credits

This notebook and associated Python modules were developed as part of a group project within the context of module S382 Astrophysics, lectured at the Open University in the UK (http://www.open.ac.uk/courses/modules/s382).

