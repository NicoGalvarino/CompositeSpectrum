import numpy as np
import scipy.stats as st
import pandas as pd
import math
import os
import pyfits as fits

def ReadSdssDr12FitsFile(inFilename):
    """
    Reads a Sloan Digital Sky Survey FITS file.

    IN:
    inFilename: the filename of the FITS file

    OUT:
    objectID: the ID of the astronomical object
    redShift: the redshift of the astronomical object
    dataframe: an Pandas dataframe containing the wavelength and spectral flux density series.
    """

    # See: http://docs.astropy.org/en/stable/io/fits/

    # Open the FITS file read-only and store a list of Header Data Units
    HduList = fits.open(inFilename)

    # Print the HDU info
#    HduList.info()

    # Get the first header and print the keys
    priHeader = HduList[1].header #HduList['PRIMARY'].header
#    print(repr(priHeader))

    # Get spectrum data is containted in the first extension
    spectrumData = HduList[1].data #HduList['COADD'].data
    spectrumColumns = HduList[1].columns.names #HduList['COADD'].columns.names
#    print(spectrumColumns)
    dataframe = pd.DataFrame(spectrumData, columns=spectrumColumns)

    # Get the number of records
    numRecords = len(dataframe.index)

    # Get range of wavelengths contained in the second extension
    spectrumProperties = HduList[2].data #HduList['SPECOBJ'].data
#    print(spectrumProperties.columns.names)
    survey        = spectrumProperties.field(0)[0]
    minWavelength = spectrumProperties.field('WAVEMIN') # in Angstroms
    maxWavelength = spectrumProperties.field('WAVEMAX') # in Angstroms
    covWavelength = spectrumProperties.field('wCoverage') # Coverage in wavelength, in units of log10 wavelength; unsure what this is
    redShift      = spectrumProperties.field('z') # Final redshift

    objectID = ['-1']
    if(survey.upper() == 'SDSS'):
        objectID = spectrumProperties.field('BESTOBJID') # Object ID
    elif(survey.upper() == 'BOSS'):
        objectID = spectrumProperties.field('OBJID') # Object ID
    elif(survey.upper() == 'SEQUELS'):
        objectID = spectrumProperties.field('OBJID') # Object ID
    else:
        print("ERROR in ReadSdssDr12FitsFile: Unknow survey type: " + survey + "...")

    # Add the wavelengths to the dataframe
    wavelengthArray = np.logspace(start=np.log10(minWavelength), stop=np.log10(maxWavelength), num = numRecords, endpoint=True)
    dataframe['wavelength'] = wavelengthArray

    # Close the FITS file
    HduList.close()

    return [objectID, redShift, dataframe]

def ConvertSdssDr12FitsToCsv(inFilename, outFilename):
    # Load data from the FITS file
    objId, z, data = ReadSdssDr12FitsFile(inFilename)
    data.to_csv(outFilename)
    print("Writing " + outFilename)

def ConvertAllSdssDr12FitsToCsv(foldername):
    for inFilename in os.listdir(foldername):
        if inFilename.endswith('.fits'):
            inFilename = foldername + inFilename
            outFilename = inFilename[0:(len(inFilename)-len('.fits'))] + '.csv'
            ConvertSdssDr12FitsToCsv(inFilename, outFilename)

def ToEmittedFrame(z, measuredFrame):
    """
    Converts the wavelengthArray to its emitted spectrum

    IN:
    z: redshift
    measuredFrame: the measured wavelengths

    OUT:
    emittedFrame: the wavelengths in the emitted frame
    """

    emittedFrame = measuredFrame / (1+z)
    return emittedFrame

def ConsiderAndMask(fluxArray, andMaskArray):
    """
    Zeros all fluxes which have an and_mask > 0

    IN:
    fluxArray: array of fluxes
    andMaskArray: array of and_masks

    OUT:
    fluxArray: array of modified fluxes
    """

    if (len(fluxArray) != len(andMaskArray)):
        print("Error in ConsiderAndMask: arrays not compatible.")
    else:
        fluxArray = fluxArray * (andMaskArray==0)

    return fluxArray

def ReBinData(wavelengthArray, fluxArray, ivarArray, binSize, startWavelength, stopWavelength):
    """
    Re-bins data.

    IN:
    wavelengthArray: array with wavelengthes to be re-binned.
    fluxArray: array with fluxes to be re-binned.
    ivarArray: array with values for uncertainty of measurements.
    binSize: new size of bins
    startWavelength: new start wavelength
    end:Wavelength new end wavelenght

    OUT:
    """

    newWavelengthArray = np.arange(start=startWavelength, stop=stopWavelength, step = binSize)
    arrayLength = len(newWavelengthArray)

    newFluxArray = np.zeros(shape=arrayLength)
    newUncFluxArray = np.zeros(shape=arrayLength)
    newNoiseFluxArray =  np.zeros(shape=arrayLength)
    newNumDataArray =  np.zeros(shape=arrayLength)

    # Define indices on the source arrays
    lowerIndex = 0
    upperIndex = 1

    for index in range(arrayLength):
        lowerWavelength = newWavelengthArray[index] - binSize/2.
        upperWavelength = newWavelengthArray[index] + binSize/2.

        if(upperWavelength < wavelengthArray[0]):
            # Out of range for wavelengthArray
            newFluxArray[index] = 0.
            newUncFluxArray[index] = 0.
            newNoiseFluxArray[index] = 0.
            newNumDataArray[index] = 0

            #DEBUG
#            print("upperWavelength = {0:.3G}; wavelengthArray[0] = {1:.3G}".format(upperWavelength, wavelengthArray[0]))

        elif(lowerWavelength > wavelengthArray[len(wavelengthArray)-1]):
            # Out of range for wavelengthArray
            newFluxArray[index] = 0.
            newUncFluxArray[index] = 0.
            newNoiseFluxArray[index] = 0.
            newNumDataArray[index] = 0

            #DEBUG
#            print("lowerWavelength = {0:.3G}; wavelengthArray[max] = {1:.3G}".format(lowerWavelength, wavelengthArray[len(wavelengthArray)-1]))

        else:
            # Point indices to the boundaries of this bin. Approach [lowerWavelength, upperWavelength>
            while (lowerIndex>0) and (wavelengthArray[lowerIndex]>=lowerWavelength):
                lowerIndex -=1

            while (lowerIndex<len(wavelengthArray)-1) and (wavelengthArray[lowerIndex]<lowerWavelength):
                lowerIndex +=1

            # lowerIndex should now point to the sample where the wavelength >= left size of the bin

            while (upperIndex>0) and (wavelengthArray[upperIndex]>upperWavelength):
                upperIndex -=1

            while (upperIndex<len(wavelengthArray)-1) and (wavelengthArray[upperIndex]<=upperWavelength):
                upperIndex +=1

            # upperIndex should now point to the sample where the wavelength is just beyond upperWavelength

            fluxSlice = fluxArray[lowerIndex:upperIndex]
            n = len(fluxSlice) # number of samples
            newNumDataArray[index] = n
            if(n==0):
                # No data points in selected range
                newFluxArray[index] = 0.
                newUncFluxArray[index] = 0.
                newNoiseFluxArray[index] = 0.

                # DEBUG
#                print(lowerWavelength, upperWavelength, lowerIndex, upperIndex)
            else:
                # Calculate the mean flux
                newFluxArray[index] = np.mean(fluxSlice)

                if(n==1):
                    newUncFluxArray[index] = 0.
                else:
                    # Calculate the standard deviation
                    fluxStdev = np.std(a=fluxSlice, ddof=1)

                    # Calculate the uncertainty of the mean value
                    newUncFluxArray[index] = fluxStdev/math.sqrt(float(n))

                # Calculate the noise of the measured values
                ivarSlice = ivarArray[lowerIndex:upperIndex]
                newNoiseFluxArray[index] = math.sqrt(np.sum(1./ivarSlice))/n

        lowerIndex = upperIndex
        if (upperIndex < len(wavelengthArray)-1):
            upperIndex +=1

    spectrumDf = pd.DataFrame({'wavelength':newWavelengthArray, 'mean_f_lambda':newFluxArray, 'noise_f_lambda':newNoiseFluxArray, 'unc_f_lambda':newUncFluxArray, 'n_data':newNumDataArray})

    return spectrumDf

def NormaliseSpectra(spectraDf):
    # Sort spectra by redshift
    spectraDf.sort_values(by='z', inplace=True)
    spectraDf.index = range(len(spectraDf.index)) # Reset the index (was also sorted)

    # Find the normalisation factors
    print("Calculating normalisation factors...")

    normalisationList = []
    for index in range(len(spectraDf.index)-1):
        # Display the redshift
        z = spectraDf['z'][index]
        print("   Processing redshift: z = {0:.3G}".format(z))

        # Get current and next rebinned spectra
        spectrumDf1 = spectraDf['spectrum'][index]
        spectrumDf2 = spectraDf['spectrum'][index+1]

        # Store fluxes in lists
        fluxList1 = []
        fluxList2 = []

        for flux1, flux2 in zip(spectrumDf1['mean_f_lambda'], spectrumDf2['mean_f_lambda']):
            if((flux1 != 0) and (flux2 != 0)):
                fluxList1.append(flux1)
                fluxList2.append(flux2)

        if((len(fluxList1) == 0) or (len(fluxList2) == 0)):
            print("ERROR in CombineSpectra: No overlapping region found.")
            print("   SpecObjectID1 = {0}; SpecObjectID2 = {1}".format(objectIdList[pointer1], objectIdList[pointer2]))

            normalisationFactor = 1.
        else:
            # Calculate the normalisation factor for spectrum 1
            normalisationFactor = np.mean(fluxList2)/np.mean(fluxList1)

        # Multiply all normalisation factors before current with the current normalisation factor
        normalisationList = [x*normalisationFactor for x in normalisationList]

        # Add the current normalisation factor
        normalisationList.append(normalisationFactor)

    # Add the normalisation factor for the last spectrum, which is 1, since all previous spectra are normalised to the last spectrum
    normalisationList.append(1.)

    # Normalise the spectra
    for index in range(len(normalisationList)):
        normalisationFactor = normalisationList[index]

        spectraDf['spectrum'][index]['mean_f_lambda'] = spectraDf['spectrum'][index]['mean_f_lambda']*normalisationFactor
        spectraDf['spectrum'][index]['unc_f_lambda']  = spectraDf['spectrum'][index]['unc_f_lambda']*normalisationFactor
        spectraDf['spectrum'][index]['noise_f_lambda']= spectraDf['spectrum'][index]['noise_f_lambda']*normalisationFactor

    return spectraDf

def CombineSpectra(spectraDf, method = 'GM'):
    # Copy the wavelength array into a new dataframe
    compositeDf = pd.DataFrame()
    compositeDf['wavelength'] = spectraDf['spectrum'][0]['wavelength']
    compositeDf['mean_f_lambda'] = np.zeros(len(compositeDf.index))

    # Loop over all fluxes and combine the spectra to 1
    print("Creating composite spectrum...")

    method = method.upper()

    # Loop over all entries
    for index in range(len(compositeDf.index)):
#        print("Index = {0}".format(index))

        # Create a list of normalised fluxes
        normalisedFluxList = []
        noiseList = []
        uncList = []

        # For each entry, loop over each spectrum
        for spectrum in spectraDf['spectrum']:
            normalisedFlux = spectrum['mean_f_lambda'][index]
            if (normalisedFlux>0):
                normalisedFluxList.append(normalisedFlux)

        flux  = 0.
        noise = 0.
        unc   = 0.
        if (method == 'GM'):
            # Geometric mean
            if(len(normalisedFluxList) == 0):
                flux  = 0.
                noise = 0.
                unc   = 0.
            else:
                flux = sp.gmean(normalisedFluxList)
        elif (method == 'AM'):
            # Arithmetic mean
            if(len(normalisedFluxList) == 0):
                flux  = 0.
                noise = 0.
                unc   = 0.
            else:
                flux = np.mean(normalisedFluxList)
        else:
            print("ERROR in CombineSpectra: unknown method for combing spectra: " + method)

        compositeDf['mean_f_lambda'].iloc[index] = flux

    #
    # What to do with the uncertainties???
    #

    return compositeDf

def CompositeSpectrum(filenameList = [], wavelengthRange = (), binSize = 4., method="GM"):
    """
    Creates a composite spectrum from multiple spectra.

    IN:

    OUT:
    """

    # Open each fits file and retrieve and store redshift
    objectIdList = []
    redshiftList = []
    spectrumList = []

    minWavelengthList = []
    maxWavelengthList = []

    #
    # Load data
    #
    for filename in filenameList:
        print("Processing: " + filename +"...")
        HduList = fits.open(filename)

        # Get spectrum data is containted in the first extension
        spectrumData = HduList[1].data #HduList['COADD'].data
        spectrumColumns = HduList[1].columns.names #HduList['COADD'].columns.names
        dfSpectrum = pd.DataFrame(spectrumData, columns=spectrumColumns)

        # Get the number of records
        numRecords = len(dfSpectrum.index)

        # Get range of wavelengths contained in the second extension
        spectrumProperties = HduList[2].data #HduList['SPECOBJ'].data

        try:
            minWavelength = spectrumProperties.field('WAVEMIN') # in Angstrom
            maxWavelength = spectrumProperties.field('WAVEMAX') # in Angstrom
            covWavelength = spectrumProperties.field('wCoverage') # Coverage in wavelength, in units of log10 wavelength; unsure what this is
            redShift      = spectrumProperties.field('z') # Final redshift
            objectId      = spectrumProperties.field('SPECOBJID') # Object ID
        except KeyError:
            print("ERROR in CombineSpectra: Could not retrieve spectrum property...")
            print(HduList[2].columns.names)
            raise KeyError

        # Close fits file
        HduList.close()

        # Create a wavelength array and add it to the dataframe
        wavelengthArray = np.logspace(start=np.log10(minWavelength), stop=np.log10(maxWavelength), num = numRecords, endpoint=True)
        dfSpectrum['wavelength'] = wavelengthArray

        # Transform min / max wavelengths to emitted frame
        minWavelength = minWavelength/(1+redShift)
        maxWavelength = maxWavelength/(1+redShift)

        # Store all in lists
        minWavelengthList.append(minWavelength)
        maxWavelengthList.append(maxWavelength)

        objectIdList.append(objectId)
        redshiftList.append(float(redShift))
        spectrumList.append(dfSpectrum)

    # Determine minimum and maximum wavelengths to use for rebinning
    if (len(wavelengthRange)<2):
        # Create a proper start and end wavelength based on the info in the FITS files
        minWavelength = min(minWavelengthList)
        maxWavelength = max(maxWavelengthList)

        minWavelength = int(minWavelength/500)*500.0
        maxWavelength = (int(maxWavelength/500)+1)*500.0
    else:
        # Use given range
        minWavelength = min(wavelenghRange)
        maxWavelength = max(wavelenghRange)

    print("Wavelength range: [{0:.3G}, {1:.3G}]".format(minWavelength, maxWavelength))

    #
    # Correct for redshift, remove bad data and rebin spectrum
    #
    print("Correcting for redshift, and rebinning spectra...")

    rebinnedList = []
    for z, spectrumDf in zip(redshiftList, spectrumList):
        # Correct for redshift
        spectrumDf['emitted wavelength'] = ToEmittedFrame(z, spectrumDf['wavelength'])

        # Zero all suspicious data
        spectrumDf['flux'] = ConsiderAndMask(spectrumDf['flux'], spectrumDf['and_mask'])

        # Rebin spectrum
        print("   Processing z = {0:.3G}".format(z))
        rebinnedDf = ReBinData(spectrumDf['emitted wavelength'], spectrumDf['flux'], spectrumDf['ivar'], binSize, minWavelength, maxWavelength)

        # DEBUG
#        spectrumDf.to_csv(str(z)+"_1.csv")
#        rebinnedSpectrumDf.to_csv(str(z)+"_2.csv")

        # Add to rebinnedSpectrumList
        rebinnedList.append(rebinnedDf)

    #
    # Normalise the spectra
    #
    spectraDf = pd.DataFrame({'z':redshiftList, 'objID':objectIdList, 'spectrum':rebinnedList})
    normalisedDf = NormaliseSpectra(spectraDf)

    #
    # Combine the individual spectra to a single composite spectrum
    #
    compositeDf = CombineSpectra(normalisedDf, method)

    # Return the composite spectrum
    return compositeDf


def CalculateRequiredRedshifts(observedWavelengthRange = (), emittedWavelengthRange = ()):
    """
    Calculates a number of redshift for spectra to use in a composite spectrum.

    IN:
    observedWavelengthRange: the tuple with the minimum and maximum wavelength values that the instrument can measure (in observed frame)
    emittedWavelengthRange: the tuple with desired wavelength range in the emitted frame.

    OUT:
    redshiftList: A list containing all advised redshifts to use
    """

    # emitted frame = observed frame / (1+z)

    redshiftList = []

    # First calculate at the minimum wavelength
    leftWavelength = min(emittedWavelengthRange)
    z = min(observedWavelengthRange) / leftWavelength - 1

    # Calculate the right boundary of the first spectrum
    rightWavelength = max(observedWavelengthRange) / (1+z)

    # Add redshift to lsit
    reshiftList.append(z)

    # Repeat while rightWavelength < maximum emittedWavelengthRange
    while rightWavelength < max(emittedWavelengthRange):
        # Calculate the left boundary of the spectrum
        leftWavelength = min(emittedWavelengthRange)
        z = observedWavelengthRange[1] / leftWavelength - 1

        # Calculate the right boundary of the spectrum
        rightWavelength = observedWavelengthRange[2] / (1+z)

        # Add redshift to lsit
        reshiftList.append(z)


    return redshiftList

def CreateSpecCombineParameterFile(binSize = 2., normalisationList = [], wavelengthRange = (), foldername = ".", parameterFilename = 'input.txt', csvFilename = 'output.csv'):
    """
    Create a parameter file for the programme 'SpecCombine.exe'

    IN:
    binSize: the size of the bins in Angstrom. Default = 2A
    normalisationList: list of normalisation factors
    wavelengthRange: tuple of the minimum and maximum wavelength for the composite spectrum.
    foldername: name of the working folder. Default is current folder.
    parameterFilename: the name of the input file for SpecCombine. This file will be generated. Default 'input.txt'
    output.csv: the name of the CSV file that SpecCombine must generate. Default 'output.csv'

    OUT:
    -
    """

    # Store the names of fits-files
    filenameList = []
    for filename in os.listdir(foldername):
        if filename.endswith(".fits"):
            filenameList.append(filename)

    # If the length of the normalisation list < length filename list, then make the normalisation factor 1. for the missing items
    if len(normalisationList) < len(filenameList):
        for index in range(len(normalisationList), len(filenameList)):
            normalisationList.append(1.)

    # Open each fits file and retrieve and store redshift
    minWavelengthList = []
    maxWavelengthList = []
    redshiftList = []

    for filename in filenameList:
        fullname = foldername +filename
        print("Processing: " + fullname +"...")
        HduList = fits.open(fullname)

        # Get range of wavelengths
        spectrumProperties = HduList[2].data # HduList['SPECOBJ'].data
        minWavelength = float(spectrumProperties.field('WAVEMIN')) # in Angstroms
        maxWavelength = float(spectrumProperties.field('WAVEMAX')) # in Angstroms
        redshift      = float(spectrumProperties.field('z')) # Final redshift

        # Transform wavelengths to emitted frame
        minWavelength = minWavelength/(1+redshift)
        maxWavelength = maxWavelength/(1+redshift)

        # Store all in lists
        minWavelengthList.append(minWavelength)
        maxWavelengthList.append(maxWavelength)
        redshiftList.append(redshift)

        # Close fits file
        HduList.close()

    if (len(wavelengthRange)<2):
        # Create a proper start and end wavelength based on the info in the FITS files
        minWavelength = min(minWavelengthList)
        maxWavelength = max(maxWavelengthList)

        minWavelength = int(minWavelength/500)*500.0
        maxWavelength = (int(maxWavelength/500)+1)*500.0
    else:
        # Use given range
        minWavelength = min(wavelenghRange)
        maxWavelength = max(wavelenghRange)

    print("Wavelength range: [{0:.3G}, {1:.3G}]".format(minWavelength, maxWavelength))

    # Store to SpecCombine input file
    filename = foldername + parameterFilename
    with open(filename, 'w') as f:
        # Write the number of files to use
        f.write(str(len(filenameList))+'\n')


        # Write the minimum wavelength, maximum wavelength, bin size
        f.write(str(minWavelength) + ',' + str(maxWavelength) + ',' + str(binSize) + '\n')

        # Write the SpecCombine output filename
        f.write(csvFilename + '\n')

        # Write for each spectrum the redshift, normalisation factor, fits filename
        for z, normalisationFactor, fitsFile in zip(redshiftList, normalisationList, filenameList):
            f.write(str(z) +',' + str(normalisationFactor) + ',' + fitsFile + '\n')

    print('Parameter file written to: ' + filename + '...')

def CreateContinuum(spectrumDf, breakpointList = []):
	"""
	Creates a continuum for spectrumDf.
	The continuum is calculated using a linear regression model.

	IN:
	spectrumDf: the dataframe containing the spectrum.
	breakPointList: the wavelength  where one power-law transitions into another.

	OUT:
	y: the numpy array containing the spectrum, including a column 'continuum'
	"""

	# Calculate the linear pieces
	slopeList = []
	interceptList = []
	stdErrList = []

	condition1 = (spectrumDf['mean_f_lambda'] > 0.)
	for index in range(len(breakpointList)):
		if(index==0):
			# The first piece...
			breakpoint = breakpointList[index]
			condition2 = (spectrumDf['wavelength'] <= breakpoint)

			indexList = spectrumDf[condition1 & condition2].index.tolist()
			x = np.log(spectrumDf['wavelength'].iloc[indexList])
			y = np.log(spectrumDf['mean_f_lambda'].iloc[indexList])
		else:
			# ... the middle piece...
			breakpoint1 = breakpointList[index-1]
			breakpoint2 = breakpointList[index]
			condition2 = (spectrumDf['wavelength']>breakpoint1)
			condition3 = (spectrumDf['wavelength'] <= breakpoint2)

			indexList = spectrumDf[condition1 & condition2 & condition3].index.tolist()
			x = np.log(spectrumDf['wavelength'].iloc[indexList])
			y = np.log(spectrumDf['mean_f_lambda'].iloc[indexList])

		slope, intercept, r_value, p_value, std_err = st.linregress(x, y)

		print("slope = {0:.3G}; intercept = {1:.3G}; std_err = {2:.3G}".format(slope, intercept, std_err))

		slopeList.append(slope)
		interceptList.append(intercept)
		stdErrList.append(std_err)

	# ...and the last piece
	breakpoint = breakpointList[-1]
	condition2 = (spectrumDf['wavelength']>breakpoint)

	indexList = spectrumDf[condition1 & condition2].index.tolist()
	x = np.log(spectrumDf['wavelength'].iloc[indexList])
	y = np.log(spectrumDf['mean_f_lambda'].iloc[indexList])

	slope, intercept, r_value, p_value, std_err = st.linregress(x, y)

	print("slope = {0:.3G}; intercept = {1:.3G}; std_err = {2:.3G}".format(slope, intercept, std_err))

	slopeList.append(slope)
	interceptList.append(intercept)
	stdErrList.append(std_err)

	# Create a new breakpoint list based on interceptions between line elements
	breakpointList = []
	for index in range(1, len(slopeList)):
		slope1 = slopeList[index-1]
		intercept1 = interceptList[index-1]

		slope2 = slopeList[index]
		intercept2 = interceptList[index]

		breakpoint = np.exp((intercept2-intercept1)/(slope1-slope2))
		breakpointList.append(breakpoint)

		print("new breakpoint at {0:.3G} A".format(breakpoint))

	# Create the continuum arrays
	x = np.zeros(len(spectrumDf.index))
	y = np.zeros(len(spectrumDf.index))

	# The first piece..
	index = 0

	breakpoint = breakpointList[index]
	slope = slopeList[index]
	intercept = interceptList[index]

	condition1 = (spectrumDf['mean_f_lambda'] > 0.)
	condition2 = (spectrumDf['wavelength'] <= breakpoint)

	indexList = spectrumDf[condition1 & condition2].index.tolist()

	x[indexList] = spectrumDf['wavelength'].iloc[indexList]
	y[indexList] = np.exp(intercept)*x[indexList]**slope

	# .. the middle pieces
	for index in range(1, len(breakpointList)):
		slope = slopeList[index]
		intercept = interceptList[index]

		breakpoint1 = breakpointList[index-1]
		breakpoint2 = breakpointList[index]

		condition2 = (spectrumDf['wavelength'] > breakpoint1)
		condition3 = (spectrumDf['wavelength'] <= breakpoint2)

		indexList = spectrumDf[condition1 & condition2 & condition3].index.tolist()

		x[indexList] = spectrumDf['wavelength'].iloc[indexList]
		y[indexList] = np.exp(intercept)*x[indexList]**slope

	# ...and the last piece
	index = -1

	slope = slopeList[index]
	intercept = interceptList[index]

	breakpoint = breakpointList[index]

	condition2 = (spectrumDf['wavelength']>breakpoint)

	indexList = spectrumDf[condition1 & condition2].index.tolist()

	x[indexList] = spectrumDf['wavelength'].iloc[indexList]
	y[indexList] = np.exp(intercept)*x[indexList]**slope

	return y

def CalculateFwhm1(spectrumDf, lineNameList = []):
	"""
	Determines the Full-Width at Half-Maximum of one or more spectrum lines.
	Achieves this by searching for troughs on both sides of an emission line.

	IN:
	spectrumDf: the dataframe defining the spectrum; should have columns 'wavelength', 'mean_f_lambda' and 'theoretical'
	lineNameList: a list with laboratory determined wavelengths of spectrum lines.

	OUT:
	FwhmDf: a dataframe with all lines and the associated FWHM values.
	"""

	# Load the table with known emission lines
	emissionLines = pd.read_csv('./hkSpectrumLines.csv', skiprows=5)

	# Create temporary storage
	FwhmLineNameList = []
	FwhmLineWavelengthList = []
	FwhmLineWidth = []

	# Loop over all spectrum lines
	for lineName in lineNameList:

		# We can have multiple lines with the same name, but at different wavelengths
		indexListLab = emissionLines['Line'][emissionLines['Line'] == lineName].index.tolist()

		# So loop over all found wavelengths for this spectrum line
		for indexLab in indexListLab:
			# Retrieve the laboratory wavelength; this is a good starting point to find an emission peak
			wavelengthLab = emissionLines['Wavelength / Å'].iloc[indexLab]

			# Search for the same wavelength in the actual spectrum
			indexSpec = 0 # Spectrum index
			while (spectrumDf['wavelength'].iloc[indexSpec] < wavelengthLab):
				indexSpec = indexSpec + 1

			# IndexSpec now points to the position in SpectrumDf that is associated with the laboratory spectrum line.
			# The actual spectrum line might however be shifted to the left or to the right.
			# Let's find the emission peak by checking neighbouring fluxes.

			# First check to the left
			while (spectrumDf['mean_f_lambda'].iloc[indexSpec-1] > spectrumDf['mean_f_lambda'].iloc[indexSpec]):
				indexSpec = indexSpec - 1

			# Now check to the right
			while (spectrumDf['mean_f_lambda'].iloc[indexSpec+1] > spectrumDf['mean_f_lambda'].iloc[indexSpec]):
				indexSpec = indexSpec + 1

			# indexSpec now points to the emission peak in the actual spectrum
			# Store the emissionline name and the actual wavelength in a list for later use
			FwhmLineNameList.append(lineName)
			FwhmLineWavelengthList.append(spectrumDf['wavelength'].iloc[indexSpec])

#			print("Emission line {0} found at {1:4.2f} Å".format(lineName, spectrumDf['wavelength'].iloc[indexSpec]))

			# Determine the half-maximum
			halfMaximum = spectrumDf['continuum'].iloc[indexSpec] + 0.5*(spectrumDf['mean_f_lambda'].iloc[indexSpec]-spectrumDf['continuum'].iloc[indexSpec])
#			print("Half-maximum = {0:.2f}".format(halfMaximum))

			#
			# Find a minimum flux value to the left or the crossing point with the half-maximum, whichever comes first
			#
			indexSpecLeft = indexSpec
			while (spectrumDf['mean_f_lambda'].iloc[indexSpecLeft-1] < spectrumDf['mean_f_lambda'].iloc[indexSpecLeft])\
			and (spectrumDf['mean_f_lambda'].iloc[indexSpecLeft] > halfMaximum):
				indexSpecLeft = indexSpecLeft - 1

#			print("1: {0:.2f}".format(spectrumDf['wavelength'].iloc[indexSpecLeft]))

			# If the flux is smaller than the half-maximum, then we have found the left wavelength, otherwise extrapolate
			if(spectrumDf['mean_f_lambda'].iloc[indexSpecLeft] > halfMaximum):
				A = (spectrumDf['mean_f_lambda'].iloc[indexSpec] - spectrumDf['mean_f_lambda'].iloc[indexSpecLeft]) / (spectrumDf['wavelength'].iloc[indexSpec] - spectrumDf['wavelength'].iloc[indexSpecLeft])
				B = spectrumDf['mean_f_lambda'].iloc[indexSpec] - A*spectrumDf['wavelength'].iloc[indexSpec]

				indexSpecLeft = indexSpec
				while (A*spectrumDf['wavelength'].iloc[indexSpecLeft]+B > halfMaximum):
					indexSpecLeft = indexSpecLeft - 1

			wavelengthLeft = spectrumDf['wavelength'].iloc[indexSpecLeft]
#			print("2: {0:.2f}".format(spectrumDf['wavelength'].iloc[indexSpecLeft]))

			#
			# Find a minimum flux value to the right or the crossing point with the half-maximum
			#
			indexSpecRight = indexSpec
			while (spectrumDf['mean_f_lambda'].iloc[indexSpecRight+1] < spectrumDf['mean_f_lambda'].iloc[indexSpecRight])\
			and (spectrumDf['mean_f_lambda'].iloc[indexSpecRight] > halfMaximum):
				indexSpecRight = indexSpecRight + 1

#			print("3: {0:.2f}".format(spectrumDf['wavelength'].iloc[indexSpecRight]))

			# If the flux is smaller than the half-maximum, then we have found the right wavelength, otherwise extrapolate
			if(spectrumDf['mean_f_lambda'].iloc[indexSpecRight] > halfMaximum):
				A = (spectrumDf['mean_f_lambda'].iloc[indexSpecRight] - spectrumDf['mean_f_lambda'].iloc[indexSpec]) / (spectrumDf['wavelength'].iloc[indexSpecRight] - spectrumDf['wavelength'].iloc[indexSpec])
				B = spectrumDf['mean_f_lambda'].iloc[indexSpec] - A*spectrumDf['wavelength'].iloc[indexSpec]

				indexSpecRight = indexSpec
				while (A*spectrumDf['wavelength'].iloc[indexSpecRight]+B > halfMaximum):
					indexSpecRight = indexSpecRight + 1

			wavelengthRight = spectrumDf['wavelength'].iloc[indexSpecRight]
#			print("4: {0:.2f}".format(spectrumDf['wavelength'].iloc[indexSpecRight]))

#			print("Wavelength left: {0:4.2f} Å; wavelength right: {1:4.2f} Å".format(wavelengthLeft, wavelengthRight))

			# Now calculate the full width and add to the list
			fwhm = wavelengthRight - wavelengthLeft
			FwhmLineWidth.append(fwhm)

#			print("Full Width at Half Maximum: {0:4.2f} Å".format((wavelengthRight - wavelengthLeft)/2.))

	# Now create the final dataframe
	FwhmDf = pd.DataFrame()
	FwhmDf['Line'] = FwhmLineNameList
	FwhmDf['Wavelength'] = FwhmLineWavelengthList
	FwhmDf['Width'] = FwhmLineWidth

	return FwhmDf

def CalculateFwhm2(spectrumDf, lineNameList = []):
	"""
	Determines the Full-Width at Half-Maximum of one or more spectrum lines.
	Achieves this by searching for crossing points with the continuum.

	IN:
	spectrumDf: the dataframe defining the spectrum; should have columns 'wavelength', 'mean_f_lambda' and 'theoretical'
	lineNameList: a list with laboratory determined wavelengths of spectrum lines.

	OUT:
	FwhmDf: a dataframe with all lines and the associated FWHM values.
	"""

	# Load the table with known emission lines
	emissionLines = pd.read_csv('./hkSpectrumLines.csv', skiprows=5)

	# Create temporary storage
	FwhmLineNameList = []
	FwhmLineWavelengthList = []
	FwhmLineWidth = []

	# Loop over all spectrum lines
	for lineName in lineNameList:

		# We can have multiple lines with the same name, but at different wavelengths
		indexListLab = emissionLines['Line'][emissionLines['Line'] == lineName].index.tolist()

		# So loop over all found wavelengths for this spectrum line
		for indexLab in indexListLab:
			# Retrieve the laboratory wavelength; this is a good starting point to find an emission peak
			wavelengthLab = emissionLines['Wavelength / Å'].iloc[indexLab]

			# Search for the same wavelength in the actual spectrum
			indexSpec = 0 # Spectrum index
			while (spectrumDf['wavelength'].iloc[indexSpec] < wavelengthLab):
				indexSpec = indexSpec + 1

			# IndexSpec now points to the position in SpectrumDf that is associated with the laboratory spectrum line.
			# The actual spectrum line might however be shifted to the left or to the right.
			# Let's find the emission peak by checking neighbouring fluxes.

			# First check to the left
			while (spectrumDf['mean_f_lambda'].iloc[indexSpec-1] > spectrumDf['mean_f_lambda'].iloc[indexSpec]):
				indexSpec = indexSpec - 1

			# Now check to the right
			while (spectrumDf['mean_f_lambda'].iloc[indexSpec+1] > spectrumDf['mean_f_lambda'].iloc[indexSpec]):
				indexSpec = indexSpec + 1

			# indexSpec now points to the emission peak in the actual spectrum
			# Store the emissionline name and the actual wavelength in a list for later use
			FwhmLineNameList.append(lineName)
			FwhmLineWavelengthList.append(spectrumDf['wavelength'].iloc[indexSpec])

#			print("Emission line {0} found at {1:4.2f} Å".format(lineName, spectrumDf['wavelength'].iloc[indexSpec]))

			# Determine the half-maximum
			halfMaximum = spectrumDf['continuum'].iloc[indexSpec] + 0.5*(spectrumDf['mean_f_lambda'].iloc[indexSpec]-spectrumDf['continuum'].iloc[indexSpec])
#			print("Half-maximum = {0:.2f}".format(halfMaximum))

			#
			# Find the left crossing point with the half-maximum
			#
			indexSpecLeft = indexSpec
			while (spectrumDf['mean_f_lambda'].iloc[indexSpecLeft] > halfMaximum):
				indexSpecLeft = indexSpecLeft - 1

			wavelengthLeft = spectrumDf['wavelength'].iloc[indexSpecLeft]
#			print("1: {0:.2f}".format(spectrumDf['wavelength'].iloc[indexSpecLeft]))

			#
			# Find the right crossing point with the half-maximum
			#
			indexSpecRight = indexSpec
			while (spectrumDf['mean_f_lambda'].iloc[indexSpecRight] > halfMaximum):
				indexSpecRight = indexSpecRight + 1

			wavelengthRight = spectrumDf['wavelength'].iloc[indexSpecRight]
#			print("2: {0:.2f}".format(spectrumDf['wavelength'].iloc[indexSpecRight]))

#			print("Wavelength left: {0:4.2f} Å; wavelength right: {1:4.2f} Å".format(wavelengthLeft, wavelengthRight))

			# Now calculate the full width and add to the list
			fwhm = wavelengthRight - wavelengthLeft
			FwhmLineWidth.append(fwhm)

#			print("Full Width at Half Maximum: {0:4.2f} Å".format((wavelengthRight - wavelengthLeft)/2.))

	# Now create the final dataframe
	FwhmDf = pd.DataFrame()
	FwhmDf['Line'] = FwhmLineNameList
	FwhmDf['Wavelength'] = FwhmLineWavelengthList
	FwhmDf['Width'] = FwhmLineWidth

	return FwhmDf

def CalculateFwhm3(spectrumDf, lineNameList = []):
	"""
	Determines the Full-Width at Half-Maximum of one or more spectrum lines.
	Achieves this by approximating an emission line by a Bell curve.

	IN:
	spectrumDf: the dataframe defining the spectrum; should have columns 'wavelength', 'mean_f_lambda' and 'theoretical'
	lineNameList: a list with laboratory determined wavelengths of spectrum lines.

	OUT:
	FwhmDf: a dataframe with all lines and the associated FWHM values.
	"""

	# Load the table with known emission lines
	emissionLines = pd.read_csv('./hkSpectrumLines.csv', skiprows=5)

	# Create temporary storage
	FwhmLineNameList = []
	FwhmLineWavelengthList = []
	FwhmLineWidth = []

	# Loop over all spectrum lines
	for lineName in lineNameList:

		# We can have multiple lines with the same name, but at different wavelengths
		indexListLab = emissionLines['Line'][emissionLines['Line'] == lineName].index.tolist()

		# So loop over all found wavelengths for this spectrum line
		for indexLab in indexListLab:
			# Retrieve the laboratory wavelength; this is a good starting point to find an emission peak
			wavelengthLab = emissionLines['Wavelength / Å'].iloc[indexLab]

			# Search for the same wavelength in the actual spectrum
			indexSpec = 0 # Spectrum index
			while (spectrumDf['wavelength'].iloc[indexSpec] < wavelengthLab):
				indexSpec = indexSpec + 1

			# IndexSpec now points to the position in SpectrumDf that is associated with the laboratory spectrum line.
			# The actual spectrum line might however be shifted to the left or to the right.
			# Let's find the emission peak by checking neighbouring fluxes.

			# First check to the left
			while (spectrumDf['mean_f_lambda'].iloc[indexSpec-1] > spectrumDf['mean_f_lambda'].iloc[indexSpec]):
				indexSpec = indexSpec - 1

			# Now check to the right
			while (spectrumDf['mean_f_lambda'].iloc[indexSpec+1] > spectrumDf['mean_f_lambda'].iloc[indexSpec]):
				indexSpec = indexSpec + 1

			# indexSpec now points to the emission peak in the actual spectrum
			# Store the emissionline name and the actual wavelength in a list for later use
			FwhmLineNameList.append(lineName)
			FwhmLineWavelengthList.append(spectrumDf['wavelength'].iloc[indexSpec])

#			print("Emission line {0} found at {1:4.2f} Å".format(lineName, spectrumDf['wavelength'].iloc[indexSpec]))

			#
			# Find the left crossing point with the continuum
			#
			indexSpecLeft = indexSpec
			while (spectrumDf['mean_f_lambda'].iloc[indexSpecLeft] > spectrumDf['continuum'].iloc[indexSpecLeft]):
				indexSpecLeft = indexSpecLeft - 1

			wavelengthLeft = spectrumDf['wavelength'].iloc[indexSpecLeft]
#			print("1: {0:.2f}".format(spectrumDf['wavelength'].iloc[indexSpecLeft]))

			#
			# Find the right crossing point with the continuum
			#
			indexSpecRight = indexSpec
			while (spectrumDf['mean_f_lambda'].iloc[indexSpecRight] > spectrumDf['continuum'].iloc[indexSpecLeft]):
				indexSpecRight = indexSpecRight + 1

			wavelengthRight = spectrumDf['wavelength'].iloc[indexSpecRight]
#			print("2: {0:.2f}".format(spectrumDf['wavelength'].iloc[indexSpecRight]))

#			print("Wavelength left: {0:4.2f} Å; wavelength right: {1:4.2f} Å".format(wavelengthLeft, wavelengthRight))

			# Now calculate the full width and add to the list
			fwhm = BellCurve(spectrumDf[indexSpecLeft:indexSpecRight])
			FwhmLineWidth.append(fwhm)

#			print("Full Width at Half Maximum: {0:4.2f} Å".format((wavelengthRight - wavelengthLeft)/2.))

	# Now create the final dataframe
	FwhmDf = pd.DataFrame()
	FwhmDf['Line'] = FwhmLineNameList
	FwhmDf['Wavelength'] = FwhmLineWavelengthList
	FwhmDf['Width'] = FwhmLineWidth

	return FwhmDf

def BellCurve(spectrumDf):
	"""
	Creates a Bell curve. This is based on a normal distribution, but used to approximate an emission line.
	Therefore the actual inetgral will not be 1, as in a normal distribution, but scaled such as to fit the peak and width of the eission line.

	IN:

	OUT:
	"""

	width = max(spectrumDf['wavelength']) - min(spectrumDf['wavelength'])
	height = max(spectrumDf['mean_f_lambda']) - min(spectrumDf['mean_f_lambda'])
	offsetX = min(spectrumDf['wavelength']) + width/2.
	offsetY = min(spectrumDf['mean_f_lambda'])

	sigma = width/6. # 99%
	mu = offsetX

#	spectrumDf['bell curve'] = height*np.exp(-1.*(spectrumDf['wavelength'] - mu)**2)/(2*sigma**2)

	xl = -np.sqrt(-1.*np.log(0.5)*2*sigma**2) + mu
	xr = +np.sqrt(-1.*np.log(0.5)*2*sigma**2) + mu

	fwhm = xr - xl
#	print("fwhm = {0:.2f}".format(fwhm))

	return fwhm
