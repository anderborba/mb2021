import pywt
coeffs = pywt.swt2([[1,2,3,4],[5,6,7,8], [9,10,11,12],[13,14,15,16]], 'sym2', level=1)
out = pywt.iswt2(coeffs, 'sym2')
