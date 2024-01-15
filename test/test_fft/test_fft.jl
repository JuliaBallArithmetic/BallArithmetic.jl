dims = (1024, 1)

v = ones(dims)

v_rad = ones(dims)/2^20

v = BallMatrix(v, v_rad)

fft_v = BallArithmetic.fft(v)

@test 1024 in fft_v[1]