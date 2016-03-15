import numpy as np
import scipy.stats.mstats as sp
import pandas as pd
import math
import os
import pyfits as fits

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
    newFluxArray: array of modified fluxes
    """
    
    newFluxArray = []
    
    if (len(fluxArray) != len(andMaskArray)):
        print("Error in ConsiderAndMask: arrays not compatible.")
    else:
        newFluxArray = fluxArray * (andMaskArray==0)

    return newFluxArray

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

    return [newWavelengthArray, newFluxArray, newUncFluxArray, newNoiseFluxArray, newNumDataArray]

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
    #HduList.info()

    # Get the first header and print the keys
    priHeader = HduList[1].header #HduList['PRIMARY'].header
    #print(repr(priHeader))

    # Get spectrum data is containted in the first extension
    spectrumData = HduList[1].data #HduList['COADD'].data
    spectrumColumns = HduList[1].columns.names #HduList['COADD'].columns.names
    dataframe = pd.DataFrame(spectrumData, columns=spectrumColumns)

    # Get the number of records
    numRecords = len(dataframe.index)

    # Get range of wavelengths contained in the second extension
    spectrumProperties = HduList[2].data #HduList['SPECOBJ'].data
    minWavelength = spectrumProperties.field('WAVEMIN') # in Angstroms
    maxWavelength = spectrumProperties.field('WAVEMAX') # in Angstroms
    covWavelength = spectrumProperties.field('wCoverage') # Coverage in wavelength, in units of log10 wavelength; unsure what this is
    redShift      = spectrumProperties.field('z') # Final redshift
    objectID      = 0 #spectrumProperties.field('BESTOBJID') # Object ID

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
    
def CombineSpectra(filenameList = [], wavelengthRange = (), binSize = 4., method="GM"):
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
    
    rebinnedSpectrumList = []
    for z, spectrumDf in zip(redshiftList, spectrumList):
        # Correct for redshift
        spectrumDf['emitted wavelength'] = ToEmittedFrame(z, spectrumDf['wavelength'])

        # Zero all suspicious data
        spectrumDf['flux'] = ConsiderAndMask(spectrumDf['flux'], spectrumDf['and_mask'])

        # Rebin spectrum
        print("   Processing z = {0:.3G}".format(z))
        newWavelengthArray, newFluxArray, newUncFluxArray, newNoiseFluxArray, newNumDataArray = ReBinData(spectrumDf['emitted wavelength'], spectrumDf['flux'], spectrumDf['ivar'], binSize, minWavelength, maxWavelength)
        
        # Convert to dataframe
        rebinnedSpectrumDf = pd.DataFrame({'wavelength':newWavelengthArray, 'mean_f_lambda':newFluxArray, 'noise_f_lambda':newNoiseFluxArray, 'unc_f_lambda':newUncFluxArray, 'n_data':newNumDataArray})

        # DEBUG
#        spectrumDf.to_csv(str(z)+"_1.csv")
#        rebinnedSpectrumDf.to_csv(str(z)+"_2.csv")
        
        # Add to rebinnedSpectrumList
        rebinnedSpectrumList.append(rebinnedSpectrumDf)
    
    # Sort objectIdList, redshiftList, spectrumList, rebinnedSpectrumList by redshift
    # Rather then sorting 3 lists, create a red-shift sorted pointer array
    print("Sorting arrays by redshift...")
    
    pointerList = range(len(redshiftList))
    pointerDf = pd.DataFrame({'ptr':pointerList, 'z':redshiftList})
    pointerDf.sort_values(by='z', inplace=True)
    pointerDf.index = range(len(pointerDf.index)) # Reset the index (was also sorted)
    
    #
    # Write temporary results to CSV-files
    #
#    for pointer in pointerDf['ptr']:
#        spectrumDf = rebinnedSpectrumList[pointer]
#        z = redshiftList[pointer]
#        
#        filename = 'Z' + str(z).replace('.','')[0:4] + '.csv'
#        spectrumDf.to_csv(filename)
    
    #
    # Normalise the spectra
    #

    # Find the normalisation factors
    print("Calculating normalisation factors...")
    
    normalisationList = []
    for index in range(len(pointerDf.index)-1):
        # Get the pointers
        pointer1 = pointerDf['ptr'][index]
        pointer2 = pointerDf['ptr'][index+1]
        
        # Display the redshift
        z = redshiftList[pointer1]
        print("   Processing redshift: z = {0:.3G}".format(z))
        
        # Get current and next rebinned spectra
        spectrumDf1 = rebinnedSpectrumList[pointer1]
        spectrumDf2 = rebinnedSpectrumList[pointer2]
        
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
    
    print("Normalisation factors: ", normalisationList)
    
    # Normalise the spectra
    for index in range(len(normalisationList)):
        # Get the normalisation factor
        normalisationFactor = normalisationList[index]
                                
        # Get the pointer
        pointer = pointerDf['ptr'][index]
        
        # Normalise the spectrum
        rebinnedSpectrumList[pointer]['normalised flux'] = rebinnedSpectrumList[pointer]['mean_f_lambda']*normalisationFactor
                                
    #
    # Combine the individual spectra to a single composite spectrum
    #
    
    # Copy the wavelength array into a new dataframe
    compositeSpectrumDf = pd.DataFrame()
    compositeSpectrumDf['wavelength'] = rebinnedSpectrumList[0]['wavelength']
    compositeSpectrumDf['flux'] = np.zeros(len(compositeSpectrumDf.index))
    
    # Loop over all fluxes and combine the spectra to 1
    print("Creating composite spectrum...")
    
    method = method.upper()
    
    # Loop over all entries
    for index in range(len(compositeSpectrumDf.index)):
#        print("Index = {0}".format(index))

        # Create a list of normalised fluxes
        normalisedFluxList = []
        
        # For each entry, loop over each spectrum
        for spectrum in rebinnedSpectrumList:
            normalisedFlux = spectrum['normalised flux'][index]
            if (normalisedFlux>0):
                normalisedFluxList.append(normalisedFlux)

        flux  = 0.
        if (method == 'GM'):
            # Geometric mean
            if(len(normalisedFluxList) == 0):
                flux = 0.
            else:
                flux = sp.gmean(normalisedFluxList)
        elif (method == 'AM'):
            # Arithmetic mean
            if(len(normalisedFluxList) == 0):
                flux = 0.
            else:
                flux = np.mean(normalisedFluxList)
        else:
            print("ERROR in CombineSpectra: unknown method for combing spectra: " + method)
        
        compositeSpectrumDf['flux'].iloc[index] = flux
                                
    #
    # What to do with the uncertainties???
    #
                                
    return compositeSpectrumDf


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
        fullname = foldername +'/' + filename
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
    filename = foldername + '/' + parameterFilename
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