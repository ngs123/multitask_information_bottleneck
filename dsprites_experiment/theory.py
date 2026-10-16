"""Exact two-task Gaussian reference, bits throughout. No third-party dependencies."""
import math

LN2 = math.log(2)

def softplus(x):
    return max(x, 0) + math.log1p(math.exp(-abs(x)))

def marginal(r, h, weight):
    x = 2 * r * LN2 - math.log(h)
    return weight * (math.exp(-x) / (1 + math.exp(-x)) if x > 0 else 1 / (1 + math.exp(x)))

def distortion(M, r1, r2):
    p, q = 1 - 1/M, 1/M
    return (p * softplus(2*math.log(M)-2*r1*LN2) + q * softplus(4*math.log(M)-2*r2*LN2))/(2*LN2)

def optimum(M, R):
    if M <= 1 or R < 0:
        raise ValueError('Require M > 1 and R >= 0')
    def derivative(r):
        return marginal(R-r, M**4, 1/M)-marginal(r, M**2, 1-1/M)
    if derivative(0) >= 0:
        r = 0.
    elif derivative(R) <= 0:
        r = R
    else:
        lo, hi = 0., R
        for _ in range(80):
            mid = (lo+hi)/2
            if derivative(mid) < 0: lo = mid
            else: hi = mid
        r = (lo+hi)/2
    return {'rate': R, 'r1': r, 'r2': R-r, 'distortion': distortion(M,r,R-r),
            'equal_distortion': distortion(M,R/2,R/2)}

def predictions(M):
    # At crossover r1=r2=R/2, solve m1=m2 for t=2^-R.
    p, q = 1-1/M, 1/M
    a, b = M**2, M**4
    t = (q*b-p*a)/(a*b*(p-q)) if p != q else -1
    cross = -math.log2(t) if 0 < t < 1 else None
    R1, R2 = math.log2(M), 4*math.log2(M)
    rows = [optimum(M,R) for R in (R1,R2)]
    return {'M':M, 'R1':R1, 'R2':R2, 'crossover':cross,
            'relative_gap_lower_bound':min(x['equal_distortion']/x['distortion']-1 for x in rows),
            'reference_points':rows}

if __name__ == '__main__':
    import json
    print(json.dumps([predictions(M) for M in (4,6,8,256)], indent=2))
