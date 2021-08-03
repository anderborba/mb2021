# To call the function
# Input: IM     - (nrows x ncols x nc) Data with one image per channel
IF = fus_dwt(IM, nrows, ncols, nc)
# output IF - Image fusion
def fus_dwt(E, m, n, nc):
    # Autors: Anderson Borba and Maurício Marengoni - Version 1.0 (04/12/2021)
    # Discrete wavelet transform Fusion
    # Input: E     - (m x n x nc) Data with one image per channel
    #        m x n - Image dimension
    #        nc    - Channels number
    # Output: F - Image fusion
    #
    # Calculates DWT to each channel nc
    # Set a list with (mat, tuple) coefficients
    cA = []
    for canal in range(nc):
        cAx, (cHx, cVx, cDx) = pywt.dwt2(E[ :, :, canal], 'db2')
        cA.append([cAx, (cHx, cVx, cDx)])
    #
    # Fusion Method
    # Calculates average to all channels with the coefficients cA from DWT transform
    cAF = 0
    for canal in range(nc):
        cAF = cAF + cA[canal][0]
    cAF = cAF / nc
    #
    # Calculates maximum to all channels with the coefficients cH, cV e Cd from DWT transform
    cHF = np.maximum(cA[0][1][0], cA[1][1][0])
    cVF = np.maximum(cA[0][1][1], cA[1][1][1])
    cDF = np.maximum(cA[0][1][2], cA[1][1][2])
    for canal in range(2, nc):
        cHF = np.maximum(cHF, cA[canal][1][0])
        cVF = np.maximum(cVF, cA[canal][1][1])
        cDF = np.maximum(cDF, cA[canal][1][2])
    #
    # Set the fusion coefficients like (mat, tuple)
    fus_coef = cAF, (cHF, cVF, cDF)
    #
    #Use the transform DWT inverse to obtain the fusion image
    F = pywt.idwt2(fus_coef, 'db2')
    return F
#
